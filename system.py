"""GraphRAG System for 2WikiMultiHopQA.

Pipeline:
    Question → Generate SPARQL → Execute Query → Get Triples →
    Find Wikipedia Sentences → Assemble Context → LLM Answer
"""

import logging
import re
from typing import Optional, Tuple, List

from .config import Config
from .models import (
    Question,
    QuestionType,
    Answer,
    Evidence,
    Entity,
    Triple,
    Subgraph,
)
from .dbpedia.client import DBpediaSPARQLClient
from .dbpedia.cache import CacheManager
from .sparql_generator import SPARQLGenerator
from .wikipedia.client import WikipediaClient
from .llm.generator import AnswerGenerator
from .llm.parser import ResponseParser

logger = logging.getLogger(__name__)


class DBpediaGraphRAGSystem:
    """GraphRAG system for multi-hop question answering.

    Uses LLM to generate SPARQL queries against DBpedia, then retrieves
    Wikipedia text containing the found entities as context for answering.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the GraphRAG system.

        Args:
            config: Optional configuration object
        """
        self.config = config or Config.from_env()

        # Initialize cache
        self.cache = CacheManager(
            use_redis=self.config.redis.enabled,
            redis_host=self.config.redis.host,
            redis_port=self.config.redis.port,
            redis_password=self.config.redis.password,
            default_ttl=self.config.dbpedia.cache_ttl,
        )

        # Initialize SPARQL client
        self.sparql_client = DBpediaSPARQLClient(
            endpoint=self.config.dbpedia.sparql_endpoint,
            timeout=self.config.dbpedia.timeout,
            rate_limit=self.config.dbpedia.rate_limit_per_second,
            cache_manager=self.cache,
        )

        # Initialize SPARQL generator
        self.sparql_generator = SPARQLGenerator()

        # Initialize Wikipedia client
        self.wiki_client = WikipediaClient(cache_manager=self.cache)

        # Initialize LLM components
        self.answer_generator = AnswerGenerator(llm_config=self.config.llm)
        self.parser = ResponseParser()

    def answer_question(
        self,
        question_text: str,
        context_paragraphs: Optional[List[dict]] = None,
    ) -> Tuple[Answer, List[Tuple[str, int]], Optional[Evidence]]:
        """Answer a question using the GraphRAG pipeline.

        Pipeline:
        1. Generate SPARQL query from question
        2. Execute SPARQL against DBpedia → get triples
        3. Find Wikipedia sentences containing entities from triples
        4. Assemble context (KG facts + Wikipedia text)
        5. Generate answer using LLM
        6. Extract supporting facts

        Args:
            question_text: The question to answer
            context_paragraphs: Optional pre-provided context (for benchmark)

        Returns:
            Tuple of (Answer, supporting_facts, evidence)
        """
        logger.info(f"Processing question: {question_text}")

        # Step 1: Generate SPARQL query
        sparql_query = self.sparql_generator.generate(question_text)
        logger.debug(f"Generated SPARQL:\n{sparql_query}")

        # Step 2: Execute SPARQL and extract triples
        triples, entities = self._execute_sparql(sparql_query, question_text)
        logger.debug(f"Found {len(triples)} triples, {len(entities)} entities")

        # Step 3 & 4: Get Wikipedia context
        if context_paragraphs:
            # Use provided context (benchmark mode)
            context = self._format_provided_context(context_paragraphs)
            paragraphs = context_paragraphs
        else:
            # Retrieve Wikipedia sentences for entities
            wiki_sentences = self._get_wikipedia_sentences(entities, triples)
            context = self._assemble_context(question_text, triples, wiki_sentences)
            paragraphs = self._sentences_to_paragraphs(wiki_sentences)

        logger.debug(f"Context length: {len(context)} chars")

        # Step 5: Generate answer
        if self.answer_generator:
            answer = self.answer_generator.generate_answer(
                question=question_text,
                context=context,
                question_type=QuestionType.UNKNOWN,
                triples=triples,
            )
        else:
            # Rule-based answer extraction (fallback)
            answer = self._extract_answer_from_triples(question_text, triples)

        logger.debug(f"Answer: {answer.text}")

        # Step 6: Find supporting facts
        supporting_facts = self._find_supporting_facts(paragraphs, triples)

        # Build evidence
        evidence = Evidence(
            triples=triples,
            supporting_sentences=[],
            reasoning_path=[f"SPARQL: {sparql_query}"],
        )

        return answer, supporting_facts, evidence

    def answer_with_context(
        self,
        question_text: str,
        context: str,
        question_type: QuestionType,
    ) -> Tuple[Answer, List[Tuple[str, int]]]:
        """Answer using pre-formatted context (for evaluation).

        Args:
            question_text: The question
            context: Pre-formatted context string
            question_type: Question type (ignored, kept for compatibility)

        Returns:
            Tuple of (Answer, supporting_facts)
        """
        answer = self.answer_generator.generate_answer(
            question=question_text,
            context=context,
            question_type=question_type,
        )

        parsed = self.parser.parse(answer.reasoning or "")
        supporting_facts = parsed.get("supporting_facts", [])

        return answer, supporting_facts

    def _execute_sparql(
        self,
        sparql_query: str,
        question: str,
    ) -> Tuple[List[Triple], List[Entity]]:
        """Execute SPARQL query and extract triples and entities.

        Args:
            sparql_query: SPARQL query string
            question: Original question (for context)

        Returns:
            Tuple of (triples, entities)
        """
        triples = []
        entities = []
        seen_entities = set()

        try:
            results = self.sparql_client.query(sparql_query)

            for row in results:
                # Handle different query result patterns
                row_triples, row_entities = self._parse_sparql_row(row, question)
                triples.extend(row_triples)

                for entity in row_entities:
                    if entity.uri not in seen_entities:
                        entities.append(entity)
                        seen_entities.add(entity.uri)

        except Exception as e:
            logger.warning(f"SPARQL execution failed: {e}")
            # Try to extract entities from question as fallback
            entities = self._extract_entities_fallback(question)

        return triples, entities

    def _parse_sparql_row(
        self,
        row: dict,
        question: str,
    ) -> Tuple[List[Triple], List[Entity]]:
        """Parse a SPARQL result row into triples and entities.

        Args:
            row: Result row dict
            question: Original question

        Returns:
            Tuple of (triples, entities)
        """
        triples = []
        entities = []
        q_lower = question.lower()

        # Pattern: grandfather query (father chain)
        if "father" in row and "answer" in row:
            father = row.get("father", "")
            answer = row.get("answer", "")
            father_label = row.get("fatherLabel", self._uri_to_label(father))
            answer_label = row.get("answerLabel", self._uri_to_label(answer))

            if father:
                triples.append(Triple(
                    subject="",
                    predicate="dbo:father",
                    obj=father,
                    obj_label=father_label,
                ))
                entities.append(Entity(uri=father, label=father_label))

            if answer:
                triples.append(Triple(
                    subject=father,
                    subject_label=father_label,
                    predicate="dbo:father",
                    obj=answer,
                    obj_label=answer_label,
                ))
                entities.append(Entity(uri=answer, label=answer_label))

        # Pattern: comparison query (birthDate)
        elif "entity" in row and "birthDate" in row:
            entity = row.get("entity", "")
            birth_date = row.get("birthDate", "")
            label = row.get("label", self._uri_to_label(entity))

            if entity and birth_date:
                triples.append(Triple(
                    subject=entity,
                    subject_label=label,
                    predicate="dbo:birthDate",
                    obj=birth_date,
                ))
                entities.append(Entity(uri=entity, label=label))

        # Pattern: director-birthplace query
        elif "director" in row and "answer" in row:
            director = row.get("director", "")
            answer = row.get("answer", "")
            director_label = row.get("directorLabel", self._uri_to_label(director))
            answer_label = row.get("answerLabel", self._uri_to_label(answer))

            if director:
                triples.append(Triple(
                    subject="",
                    predicate="dbo:director",
                    obj=director,
                    obj_label=director_label,
                ))
                entities.append(Entity(uri=director, label=director_label))

            if answer:
                triples.append(Triple(
                    subject=director,
                    subject_label=director_label,
                    predicate="dbo:birthPlace",
                    obj=answer,
                    obj_label=answer_label,
                ))
                entities.append(Entity(uri=answer, label=answer_label))

        # Pattern: nationality comparison
        elif "nat1" in row and "nat2" in row:
            nat1 = row.get("nat1", "")
            nat2 = row.get("nat2", "")
            nat1_label = row.get("nat1Label", self._uri_to_label(nat1))
            nat2_label = row.get("nat2Label", self._uri_to_label(nat2))

            triples.append(Triple(
                subject="Entity1",
                predicate="dbo:nationality",
                obj=nat1,
                obj_label=nat1_label,
            ))
            triples.append(Triple(
                subject="Entity2",
                predicate="dbo:nationality",
                obj=nat2,
                obj_label=nat2_label,
            ))

        # Generic pattern: property-value
        elif "property" in row and "value" in row:
            prop = row.get("property", "")
            value = row.get("value", "")
            value_label = row.get("valueLabel", self._uri_to_label(value))

            triples.append(Triple(
                subject="",
                predicate=prop,
                obj=value,
                obj_label=value_label,
            ))

        return triples, entities

    def _get_wikipedia_sentences(
        self,
        entities: List[Entity],
        triples: List[Triple],
    ) -> dict[str, List[Tuple[int, str]]]:
        """Get Wikipedia sentences containing entities from triples.

        Args:
            entities: List of entities to fetch
            triples: Triples (for finding relevant content)

        Returns:
            Dict mapping entity title to list of (index, sentence) tuples
        """
        wiki_sentences = {}

        # Collect all entity/value names to search for
        search_terms = set()
        for e in entities:
            if e.label:
                search_terms.add(e.label.lower())
        for t in triples:
            if t.subject_label:
                search_terms.add(t.subject_label.lower())
            if t.obj_label:
                search_terms.add(t.obj_label.lower())

        # Get Wikipedia content for each entity
        for entity in entities:
            title = entity.label
            if not title:
                continue

            try:
                sentences = self.wiki_client.get_sentences(title, max_sentences=10)

                if sentences:
                    relevant = []
                    for i, sent in enumerate(sentences):
                        sent_lower = sent.lower()

                        # Include if mentions other entities or has relation keywords
                        is_relevant = any(term in sent_lower for term in search_terms)
                        has_keywords = any(kw in sent_lower for kw in [
                            "son of", "daughter of", "father", "mother",
                            "born", "died", "directed", "founded", "married"
                        ])

                        if is_relevant or has_keywords:
                            relevant.append((i, sent))

                    if relevant:
                        wiki_sentences[title] = relevant

            except Exception as e:
                logger.warning(f"Failed to get Wikipedia for {title}: {e}")

        return wiki_sentences

    def _assemble_context(
        self,
        question: str,
        triples: List[Triple],
        wiki_sentences: dict[str, List[Tuple[int, str]]],
    ) -> str:
        """Assemble context from triples and Wikipedia sentences.

        Args:
            question: Original question
            triples: Knowledge graph triples
            wiki_sentences: Wikipedia sentences by entity

        Returns:
            Formatted context string
        """
        parts = []

        # Knowledge graph facts
        if triples:
            parts.append("Knowledge Graph Facts:")
            for t in triples:
                subj = t.subject_label or self._uri_to_label(t.subject)
                pred = self._uri_to_label(t.predicate)
                obj = t.obj_label or self._uri_to_label(t.obj)
                if subj and obj:
                    parts.append(f"  - {subj} → {pred} → {obj}")
                elif obj:
                    parts.append(f"  - {pred}: {obj}")
            parts.append("")

        # Wikipedia evidence
        if wiki_sentences:
            parts.append("Wikipedia Context:")
            for title, sentences in wiki_sentences.items():
                parts.append(f"\nParagraph: {title}")
                for idx, sent in sentences[:5]:
                    parts.append(f"  [{idx}] {sent}")

        return "\n".join(parts)

    def _extract_answer_from_triples(
        self,
        question: str,
        triples: List[Triple],
    ) -> Answer:
        """Extract answer from triples without LLM (fallback).

        Args:
            question: Original question
            triples: Retrieved triples

        Returns:
            Answer object
        """
        if not triples:
            return Answer(text="Could not find answer", confidence=0.0)

        q_lower = question.lower()

        # Grandfather questions → last entity in chain
        if "grandfather" in q_lower:
            if len(triples) >= 2:
                answer_text = triples[-1].obj_label or self._uri_to_label(triples[-1].obj)
                return Answer(
                    text=answer_text,
                    confidence=0.9,
                    reasoning=f"Chain: {' → '.join([t.obj_label or t.obj for t in triples])}",
                )

        # "Born first" questions → compare dates
        if "born first" in q_lower or "born earlier" in q_lower:
            dated = []
            for t in triples:
                if "birthDate" in t.predicate:
                    label = t.subject_label or self._uri_to_label(t.subject)
                    dated.append((label, t.obj))

            if dated:
                dated.sort(key=lambda x: x[1])
                return Answer(
                    text=dated[0][0],
                    confidence=0.9,
                    reasoning=f"Birth dates: {dated}",
                )

        # "Same nationality" questions → compare values
        if "same nationality" in q_lower:
            nationalities = [t.obj_label or t.obj for t in triples if "nationality" in t.predicate]
            if len(nationalities) >= 2:
                same = nationalities[0] == nationalities[1]
                return Answer(
                    text="yes" if same else "no",
                    confidence=0.9,
                    reasoning=f"Nationalities: {nationalities}",
                )

        # Birthplace questions
        if "birthplace" in q_lower:
            for t in triples:
                if "birthPlace" in t.predicate:
                    return Answer(
                        text=t.obj_label or self._uri_to_label(t.obj),
                        confidence=0.9,
                    )

        # Default: return last triple's object
        last = triples[-1]
        return Answer(
            text=last.obj_label or self._uri_to_label(last.obj),
            confidence=0.7,
        )

    def _find_supporting_facts(
        self,
        paragraphs: List[dict],
        triples: List[Triple],
    ) -> List[Tuple[str, int]]:
        """Find supporting fact indices in paragraphs.

        Args:
            paragraphs: Context paragraphs
            triples: KG triples used

        Returns:
            List of (title, sentence_index) tuples
        """
        supporting = []

        # Collect object values from triples
        obj_values = set()
        for t in triples:
            if t.obj_label:
                obj_values.add(t.obj_label.lower())
            obj_values.add(self._uri_to_label(t.obj).lower())

        for para in paragraphs:
            title = para.get("title", "")
            sentences = para.get("sentences", [])

            for i, sent in enumerate(sentences):
                sent_lower = sent.lower()

                # Check if sentence contains any triple object
                for obj in obj_values:
                    if obj and obj in sent_lower:
                        supporting.append((title, i))
                        break

        return list(set(supporting))

    def _sentences_to_paragraphs(
        self,
        wiki_sentences: dict[str, List[Tuple[int, str]]],
    ) -> List[dict]:
        """Convert wiki_sentences dict to paragraph list format.

        Args:
            wiki_sentences: Dict of title -> [(idx, sentence)]

        Returns:
            List of paragraph dicts
        """
        paragraphs = []
        for title, sentences in wiki_sentences.items():
            paragraphs.append({
                "title": title,
                "sentences": [sent for _, sent in sentences],
            })
        return paragraphs

    def _format_provided_context(self, paragraphs: List[dict]) -> str:
        """Format provided context paragraphs.

        Args:
            paragraphs: List of paragraph dicts

        Returns:
            Formatted context string
        """
        parts = []
        for para in paragraphs:
            title = para.get("title", "")
            sentences = para.get("sentences", [])

            parts.append(f"Paragraph: {title}")
            for i, sent in enumerate(sentences):
                parts.append(f"[{i}] {sent}")
            parts.append("")

        return "\n".join(parts)

    def _extract_entities_fallback(self, question: str) -> List[Entity]:
        """Extract entities from question text as fallback.

        Args:
            question: Question text

        Returns:
            List of entities
        """
        entities = []

        # Extract capitalized names
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Multi-word names
            r'([A-Z][a-z]+)\s+(?:or|and)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # X or/and Y
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m and m not in ["Who", "What", "Where", "When", "Which"]:
                            entities.append(Entity(uri="", label=m))
                elif match not in ["Who", "What", "Where", "When", "Which"]:
                    entities.append(Entity(uri="", label=match))

        return entities

    def _uri_to_label(self, uri: str) -> str:
        """Extract human-readable label from URI.

        Args:
            uri: DBpedia URI or label

        Returns:
            Human-readable label
        """
        if not uri:
            return ""

        if "dbpedia.org/resource/" in uri:
            label = uri.split("/")[-1]
        elif "dbpedia.org/ontology/" in uri:
            label = uri.split("/")[-1]
        elif "dbpedia.org/property/" in uri:
            label = uri.split("/")[-1]
        else:
            label = uri

        return label.replace("_", " ")

    def get_entity_info(self, entity_name: str) -> Optional[Entity]:
        """Get information about an entity from DBpedia.

        Args:
            entity_name: Name of the entity

        Returns:
            Entity object or None
        """
        try:
            entity = self.sparql_client.get_entity_by_label(entity_name)
            return entity
        except Exception as e:
            logger.warning(f"Failed to get entity {entity_name}: {e}")
            return None

    def get_wikipedia_text(self, entity: Entity, max_sentences: int = 10) -> List[str]:
        """Get Wikipedia text for an entity.

        Args:
            entity: The entity
            max_sentences: Max sentences to return

        Returns:
            List of Wikipedia sentences
        """
        return self.wiki_client.get_content_for_dbpedia_entity(
            entity.uri, max_sentences
        )


def create_system(config: Optional[Config] = None) -> DBpediaGraphRAGSystem:
    """Factory function to create a GraphRAG system.

    Args:
        config: Optional configuration

    Returns:
        Configured DBpediaGraphRAGSystem instance
    """
    return DBpediaGraphRAGSystem(config)
