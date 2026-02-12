"""LLM-based SPARQL query generation for DBpedia with hybrid schema discovery."""

import re
import logging
from typing import Optional

from dbpedia_qa.dbpedia.schema_discovery import SchemaDiscovery

logger = logging.getLogger(__name__)


SPARQL_GENERATION_PROMPT = """You are a SPARQL expert. Generate a query to answer the question using the schema below.

{schema}

QUESTION: {question}

RULES:
1. Use the EXACT entity URIs provided in the schema
2. For multi-hop queries, chain patterns (e.g., ?x dbp:father ?father . ?father dbp:father ?grandfather)
3. "paternal grandfather" = father's father (2 hops via father property)
4. "maternal grandmother" = mother's mother (2 hops via mother property)
5. Prefer dbp: properties when available as they have better coverage
6. Use OPTIONAL for labels to get human-readable names

Generate ONLY the SPARQL query, no explanation:
"""


class SPARQLGenerator:
    """Generates SPARQL queries using LLM with hybrid schema discovery."""

    def __init__(self, llm_client=None, sparql_endpoint: str = "https://dbpedia.org/sparql"):
        """Initialize generator.

        Args:
            llm_client: LLM client for query generation (optional)
            sparql_endpoint: DBpedia SPARQL endpoint URL
        """
        self.llm_client = llm_client
        self.schema_discovery = SchemaDiscovery(sparql_endpoint)

    def generate(self, question: str) -> str:
        """Generate SPARQL query for a question.

        Uses hybrid approach:
        1. Extract entities from question
        2. Discover schema for each entity (class + properties)
        3. Give schema to LLM to generate SPARQL

        Args:
            question: Natural language question

        Returns:
            SPARQL query string
        """
        # Extract entities from question
        entities = self._extract_entities(question)
        logger.info(f"Extracted entities: {entities}")

        if self.llm_client and entities:
            return self._generate_with_llm(question, entities)
        else:
            return self._generate_with_templates(question, entities)

    def _generate_with_llm(self, question: str, entities: list[str]) -> str:
        """Generate SPARQL using LLM with dynamic schema discovery."""
        # Get hybrid schema for all entities
        schema = self.schema_discovery.get_schema_for_question(question, entities)
        logger.debug(f"Discovered schema:\n{schema}")

        prompt = SPARQL_GENERATION_PROMPT.format(
            schema=schema,
            question=question
        )

        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=500,
        )

        # Extract SPARQL from response
        return self._extract_sparql(response)

    def _generate_with_templates(self, question: str, entities: list[str]) -> str:
        """Generate SPARQL using pattern matching (fallback when no LLM)."""
        q_lower = question.lower()

        # Get entity URIs using schema discovery
        entity_uris = []
        for entity in entities:
            uri = self.schema_discovery.get_entity_uri(entity)
            if uri:
                entity_uris.append(uri)
            else:
                # Fallback to simple conversion
                entity_uris.append(f"http://dbpedia.org/resource/{entity.replace(' ', '_')}")

        # Pattern: "paternal grandfather of X" → 2-hop father chain
        if "grandfather" in q_lower:
            uri = entity_uris[0] if entity_uris else "http://dbpedia.org/resource/ENTITY"
            return f"""
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?grandfather ?grandfatherLabel WHERE {{
    <{uri}> dbp:father ?father .
    ?father dbp:father ?grandfather .
    OPTIONAL {{ ?grandfather rdfs:label ?grandfatherLabel . FILTER(LANG(?grandfatherLabel) = "en") }}
}}
"""

        # Pattern: "grandmother" → 2-hop mother chain
        if "grandmother" in q_lower:
            uri = entity_uris[0] if entity_uris else "http://dbpedia.org/resource/ENTITY"
            prop = "mother" if "maternal" in q_lower else "father"
            return f"""
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?grandmother ?grandmotherLabel WHERE {{
    <{uri}> dbp:{prop} ?parent .
    ?parent dbp:mother ?grandmother .
    OPTIONAL {{ ?grandmother rdfs:label ?grandmotherLabel . FILTER(LANG(?grandmotherLabel) = "en") }}
}}
"""

        # Pattern: "birthplace of director of X" → 2-hop film→director→birthplace
        if "birthplace" in q_lower and "director" in q_lower:
            uri = entity_uris[0] if entity_uris else "http://dbpedia.org/resource/ENTITY"
            return f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?birthplace ?birthplaceLabel ?director ?directorLabel WHERE {{
    <{uri}> (dbo:director|dbp:director) ?director .
    ?director (dbo:birthPlace|dbp:birthPlace) ?birthplace .
    OPTIONAL {{ ?birthplace rdfs:label ?birthplaceLabel . FILTER(LANG(?birthplaceLabel) = "en") }}
    OPTIONAL {{ ?director rdfs:label ?directorLabel . FILTER(LANG(?directorLabel) = "en") }}
}}
"""

        # Pattern: "born first, X or Y" → comparison query
        if "born first" in q_lower or "born earlier" in q_lower:
            if len(entity_uris) >= 2:
                return f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?entity ?label ?birthDate WHERE {{
    VALUES ?entity {{ <{entity_uris[0]}> <{entity_uris[1]}> }}
    ?entity (dbo:birthDate|dbp:birthDate) ?birthDate .
    OPTIONAL {{ ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
}}
ORDER BY ?birthDate
"""

        # Pattern: "same nationality" → comparison query
        if "same nationality" in q_lower or ("nationality" in q_lower and " and " in q_lower):
            if len(entity_uris) >= 2:
                return f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?nat1 ?nat1Label ?nat2 ?nat2Label WHERE {{
    <{entity_uris[0]}> (dbo:nationality|dbp:nationality) ?nat1 .
    <{entity_uris[1]}> (dbo:nationality|dbp:nationality) ?nat2 .
    OPTIONAL {{ ?nat1 rdfs:label ?nat1Label . FILTER(LANG(?nat1Label) = "en") }}
    OPTIONAL {{ ?nat2 rdfs:label ?nat2Label . FILTER(LANG(?nat2Label) = "en") }}
}}
"""

        # Default: single entity property lookup
        uri = entity_uris[0] if entity_uris else "http://dbpedia.org/resource/ENTITY"
        return f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?property ?value ?valueLabel WHERE {{
    <{uri}> ?property ?value .
    FILTER(
        STRSTARTS(STR(?property), "http://dbpedia.org/ontology/") ||
        STRSTARTS(STR(?property), "http://dbpedia.org/property/")
    )
    FILTER(!CONTAINS(STR(?property), "wikiPage"))
    OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "en") }}
}}
LIMIT 50
"""

    def _extract_entities(self, question: str) -> list[str]:
        """Extract entity names from question."""
        entities = []

        # Check for "X or Y" pattern
        or_match = re.search(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+or\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            question
        )
        if or_match:
            entities.append(or_match.group(1))
            entities.append(or_match.group(2))
            return entities

        # Check for "X and Y" pattern
        and_match = re.search(
            r'(?:^|[^A-Za-z])([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            question
        )
        if and_match:
            entity1 = and_match.group(1)
            entity2 = and_match.group(2)
            skip_words = {"Were", "Was", "Is", "Are", "Did", "Do", "Does", "The", "A", "An"}
            if entity1.split()[0] not in skip_words:
                entities.append(entity1)
            else:
                entity1_words = entity1.split()
                if len(entity1_words) > 1:
                    entities.append(" ".join(entity1_words[1:]))
            entities.append(entity2)
            return entities

        # Check for "of X" pattern (e.g., "grandfather of Prince William")
        of_match = re.search(r'of\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)', question)
        if of_match:
            entity = of_match.group(1).rstrip(',').strip()
            entities.append(entity)

        # Check for film/movie names
        film_match = re.search(r'(?:film|movie)\s+([^?]+?)(?:\s*\?|$)', question, re.IGNORECASE)
        if film_match:
            film_name = film_match.group(1).strip()
            if film_name and film_name not in entities:
                entities.append(film_name)

        return entities

    def _extract_sparql(self, response: str) -> str:
        """Extract SPARQL query from LLM response."""
        # Remove markdown code blocks
        if "```" in response:
            lines = response.split("\n")
            clean_lines = [l for l in lines if not l.strip().startswith("```")]
            response = "\n".join(clean_lines)

        # Look for query between PREFIX and closing brace
        match = re.search(r'(PREFIX[\s\S]+?\})\s*$', response, re.MULTILINE)
        if match:
            return match.group(1)

        # Look for SELECT query
        match = re.search(r'(SELECT[\s\S]+?\})', response)
        if match:
            return match.group(1)

        return response.strip()

    def _to_dbpedia_uri(self, entity: str) -> str:
        """Convert entity name to DBpedia URI format (deprecated, use schema_discovery)."""
        clean = entity.strip()
        uri_name = clean.replace(" ", "_")
        return f"dbr:{uri_name}"


def generate_sparql_for_question(question: str, llm_client=None) -> str:
    """Convenience function to generate SPARQL for a question.

    Args:
        question: Natural language question
        llm_client: Optional LLM client

    Returns:
        SPARQL query string
    """
    generator = SPARQLGenerator(llm_client)
    return generator.generate(question)
