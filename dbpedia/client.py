"""DBpedia SPARQL client with caching and rate limiting."""

import logging
import time
from typing import Any, Optional

from SPARQLWrapper import JSON, SPARQLWrapper
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models import Entity, Triple
from .cache import CacheManager
from .sparql_utils import (
    build_bif_expression,
    escape_sparql_string,
    escape_sparql_uri,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float = 10.0):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()

    def acquire(self) -> None:
        """Wait until a token is available."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 0
        else:
            self.tokens -= 1


class DBpediaSPARQLClient:
    """SPARQL client for querying DBpedia with caching and rate limiting."""

    PREFIXES = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    """

    def __init__(
        self,
        endpoint: str = "https://dbpedia.org/sparql",
        timeout: int = 30,
        rate_limit: float = 10.0,
        cache_manager: Optional[CacheManager] = None,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self.rate_limiter = RateLimiter(rate_limit)
        self.cache = cache_manager or CacheManager()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _execute_query(self, query: str) -> dict[str, Any]:
        """Execute SPARQL query with retry logic."""
        self.rate_limiter.acquire()
        full_query = f"{self.PREFIXES}\n{query}"
        self.sparql.setQuery(full_query)

        try:
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            raise

    def query(self, query: str, use_cache: bool = True) -> dict[str, Any]:
        """Execute SPARQL query with optional caching."""
        if use_cache:
            cache_key = f"sparql:{query}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        results = self._execute_query(query)

        if use_cache:
            self.cache.set(cache_key, results)

        return results

    def get_entity_types(self, entity_uri: str) -> list[str]:
        """Get dbo: rdf:type values for an entity."""
        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT ?type WHERE {{
            <{safe_uri}> rdf:type ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        """
        try:
            raw = self.query(query)
            bindings = raw.get("results", {}).get("bindings", [])
            return [b["type"]["value"].split("/")[-1] for b in bindings]
        except Exception as e:
            logger.warning(f"Type query failed for {entity_uri}: {e}")
            return []

    def get_entity_by_label(self, label: str, lang: str = "en") -> Optional[Entity]:
        """Find entity by exact label match."""
        query = f"""
        SELECT ?uri ?abstract
        WHERE {{
            ?uri rdfs:label "{escape_sparql_string(label)}"@{lang} .
            OPTIONAL {{ ?uri dbo:abstract ?abstract . FILTER(LANG(?abstract) = "{lang}") }}
        }}
        LIMIT 1
        """
        results = self.query(query)
        bindings = results.get("results", {}).get("bindings", [])

        if bindings:
            binding = bindings[0]
            return Entity(
                uri=binding["uri"]["value"],
                label=label,
                abstract=binding.get("abstract", {}).get("value"),
                source="sparql",
            )
        return None

    def get_entity_by_uri(self, uri: str) -> Optional[Entity]:
        """Get entity details by URI."""
        # Normalize URI
        if not uri.startswith("http"):
            uri = f"http://dbpedia.org/resource/{uri.replace(' ', '_')}"

        safe_uri = escape_sparql_uri(uri)
        query = f"""
        SELECT ?label ?abstract ?type
        WHERE {{
            <{safe_uri}> rdfs:label ?label .
            FILTER(LANG(?label) = "en")
            OPTIONAL {{ <{safe_uri}> dbo:abstract ?abstract . FILTER(LANG(?abstract) = "en") }}
            OPTIONAL {{ <{safe_uri}> rdf:type ?type . }}
        }}
        LIMIT 10
        """
        results = self.query(query)
        bindings = results.get("results", {}).get("bindings", [])

        if bindings:
            types = list(
                set(b.get("type", {}).get("value", "") for b in bindings if "type" in b)
            )
            return Entity(
                uri=uri,
                label=bindings[0].get("label", {}).get("value", ""),
                abstract=bindings[0].get("abstract", {}).get("value"),
                types=types,
                source="sparql",
            )
        return None

    def get_entity_properties(
        self, entity_uri: str, properties: Optional[list[str]] = None
    ) -> list[Triple]:
        """Get all or specific properties of an entity."""
        if not entity_uri.startswith("http"):
            entity_uri = f"http://dbpedia.org/resource/{entity_uri.replace(' ', '_')}"

        if properties:
            prop_filter = " || ".join(f"?p = {p}" for p in properties)
            filter_clause = f"FILTER({prop_filter})"
        else:
            filter_clause = ""

        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT ?p ?o ?pLabel ?oLabel
        WHERE {{
            <{safe_uri}> ?p ?o .
            {filter_clause}
            OPTIONAL {{ ?p rdfs:label ?pLabel . FILTER(LANG(?pLabel) = "en") }}
            OPTIONAL {{ ?o rdfs:label ?oLabel . FILTER(LANG(?oLabel) = "en") }}
        }}
        LIMIT 100
        """
        results = self.query(query)
        bindings = results.get("results", {}).get("bindings", [])

        triples = []
        for binding in bindings:
            triple = Triple(
                subject=entity_uri,
                predicate=binding["p"]["value"],
                obj=binding["o"]["value"],
                predicate_label=binding.get("pLabel", {}).get("value"),
                obj_label=binding.get("oLabel", {}).get("value"),
                source="dbpedia",
            )
            triples.append(triple)

        return triples

    def get_neighbors(
        self,
        entity_uri: str,
        relation_types: Optional[set[str]] = None,
        direction: str = "both",
        limit: int = 50,
    ) -> list[tuple[str, str, float]]:
        """Get neighboring entities connected by relations.

        Returns: List of (relation, neighbor_uri, confidence) tuples.
        """
        if not entity_uri.startswith("http"):
            entity_uri = f"http://dbpedia.org/resource/{entity_uri.replace(' ', '_')}"

        safe_uri = escape_sparql_uri(entity_uri)
        results = []

        # Outgoing relations
        if direction in ("out", "both"):
            query = f"""
            SELECT DISTINCT ?p ?neighbor ?neighborLabel
            WHERE {{
                <{safe_uri}> ?p ?neighbor .
                FILTER(isIRI(?neighbor))
                FILTER(STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/"))
                OPTIONAL {{ ?neighbor rdfs:label ?neighborLabel . FILTER(LANG(?neighborLabel) = "en") }}
            }}
            LIMIT {limit}
            """
            query_results = self.query(query)
            for binding in query_results.get("results", {}).get("bindings", []):
                pred = binding["p"]["value"]
                if relation_types is None or any(rt in pred for rt in relation_types):
                    results.append(
                        (pred, binding["neighbor"]["value"], 1.0)
                    )

        # Incoming relations
        if direction in ("in", "both"):
            query = f"""
            SELECT DISTINCT ?p ?neighbor ?neighborLabel
            WHERE {{
                ?neighbor ?p <{safe_uri}> .
                FILTER(isIRI(?neighbor))
                FILTER(STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/"))
                OPTIONAL {{ ?neighbor rdfs:label ?neighborLabel . FILTER(LANG(?neighborLabel) = "en") }}
            }}
            LIMIT {limit}
            """
            query_results = self.query(query)
            for binding in query_results.get("results", {}).get("bindings", []):
                pred = binding["p"]["value"]
                if relation_types is None or any(rt in pred for rt in relation_types):
                    results.append(
                        (pred, binding["neighbor"]["value"], 1.0)
                    )

        return results

    def get_neighbors_with_types(
        self,
        entity_uri: str,
        direction: str = "out",
        limit: int = 50,
    ) -> list[tuple[str, str, set[str]]]:
        """Get neighboring entities with their ontology types.

        Uses GROUP_CONCAT to aggregate multiple rdf:type values per neighbor
        into a single row, avoiding row explosion from the OPTIONAL join.

        Returns: List of (relation_uri, neighbor_uri, neighbor_types) tuples
                 where neighbor_types is a set of dbo: class local names.
        """
        if not entity_uri.startswith("http"):
            entity_uri = f"http://dbpedia.org/resource/{entity_uri.replace(' ', '_')}"

        safe_uri = escape_sparql_uri(entity_uri)
        results = []

        if direction in ("out", "both"):
            query = f"""
            SELECT ?p ?neighbor
                   (GROUP_CONCAT(DISTINCT STR(?nt); separator=",") AS ?neighborTypes)
            WHERE {{
                <{safe_uri}> ?p ?neighbor .
                FILTER(isIRI(?neighbor))
                FILTER(STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/"))
                OPTIONAL {{
                    ?neighbor a ?nt .
                    FILTER(STRSTARTS(STR(?nt), "http://dbpedia.org/ontology/"))
                }}
            }}
            GROUP BY ?p ?neighbor
            LIMIT {limit}
            """
            query_results = self.query(query)
            for binding in query_results.get("results", {}).get("bindings", []):
                pred = binding["p"]["value"]
                neighbor = binding["neighbor"]["value"]
                types_str = binding.get("neighborTypes", {}).get("value", "")
                types = set()
                if types_str:
                    for t in types_str.split(","):
                        # Extract local name from full URI
                        local = t.split("/")[-1] if "/" in t else t
                        if local:
                            types.add(local)
                results.append((pred, neighbor, types))

        if direction in ("in", "both"):
            query = f"""
            SELECT ?p ?neighbor
                   (GROUP_CONCAT(DISTINCT STR(?nt); separator=",") AS ?neighborTypes)
            WHERE {{
                ?neighbor ?p <{safe_uri}> .
                FILTER(isIRI(?neighbor))
                FILTER(STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/"))
                OPTIONAL {{
                    ?neighbor a ?nt .
                    FILTER(STRSTARTS(STR(?nt), "http://dbpedia.org/ontology/"))
                }}
            }}
            GROUP BY ?p ?neighbor
            LIMIT {limit}
            """
            query_results = self.query(query)
            for binding in query_results.get("results", {}).get("bindings", []):
                pred = binding["p"]["value"]
                neighbor = binding["neighbor"]["value"]
                types_str = binding.get("neighborTypes", {}).get("value", "")
                types = set()
                if types_str:
                    for t in types_str.split(","):
                        local = t.split("/")[-1] if "/" in t else t
                        if local:
                            types.add(local)
                results.append((pred, neighbor, types))

        return results

    def get_property_value(
        self, entity_uri: str, property_uri: str
    ) -> Optional[str]:
        """Get a specific property value for an entity."""
        if not entity_uri.startswith("http"):
            entity_uri = f"http://dbpedia.org/resource/{entity_uri.replace(' ', '_')}"

        safe_uri = escape_sparql_uri(entity_uri)
        safe_prop = escape_sparql_uri(property_uri)
        query = f"""
        SELECT ?value ?valueLabel
        WHERE {{
            <{safe_uri}> <{safe_prop}> ?value .
            OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "en") }}
        }}
        LIMIT 1
        """
        results = self.query(query)
        bindings = results.get("results", {}).get("bindings", [])

        if bindings:
            if "valueLabel" in bindings[0]:
                return bindings[0]["valueLabel"]["value"]
            return bindings[0]["value"]["value"]
        return None

    def search_entities(
        self, search_term: str, entity_type: Optional[str] = None, limit: int = 10
    ) -> list[Entity]:
        """Search for entities by name using fuzzy matching."""
        type_filter = f"?uri rdf:type {entity_type} ." if entity_type else ""

        query = f"""
        SELECT DISTINCT ?uri ?label ?abstract
        WHERE {{
            ?uri rdfs:label ?label .
            {type_filter}
            FILTER(LANG(?label) = "en")
            FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{escape_sparql_string(search_term)}")))
            OPTIONAL {{ ?uri dbo:abstract ?abstract . FILTER(LANG(?abstract) = "en") }}
        }}
        LIMIT {limit}
        """
        results = self.query(query)
        bindings = results.get("results", {}).get("bindings", [])

        entities = []
        for binding in bindings:
            entity = Entity(
                uri=binding["uri"]["value"],
                label=binding["label"]["value"],
                abstract=binding.get("abstract", {}).get("value"),
                source="sparql",
            )
            entities.append(entity)

        return entities

    def search_entities_fulltext(
        self,
        search_term: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities using Virtuoso bif:contains full-text index.

        Much faster than FILTER(CONTAINS(...)) because it uses Virtuoso's
        built-in text index. Supports multi-word AND queries.
        """
        bif_expr = build_bif_expression(search_term)
        if bif_expr is None:
            return []

        type_clause = f"?uri a {entity_type} ." if entity_type else ""
        query = f"""
        SELECT DISTINCT ?uri ?label WHERE {{
            ?uri rdfs:label ?label .
            ?label bif:contains "{bif_expr}" .
            {type_clause}
            FILTER(LANG(?label) = "en")
            FILTER(STRSTARTS(STR(?uri), "http://dbpedia.org/resource/"))
            FILTER(!STRSTARTS(STR(?uri), "http://dbpedia.org/resource/Category:"))
        }} LIMIT {int(limit)}
        """
        try:
            results = self.query(query)
        except Exception as e:
            logger.warning(f"bif:contains search failed for '{search_term}': {e}")
            return []

        entities = []
        for binding in results.get("results", {}).get("bindings", []):
            entities.append(Entity(
                uri=binding["uri"]["value"],
                label=binding["label"]["value"],
                source="sparql_fulltext",
            ))
        return entities

    def search_entity_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None,
    ) -> Optional[Entity]:
        """Find the best DBpedia entity for a name string.

        Cascade: exact label match -> bif:contains fulltext search.
        Disambiguation: exact label > type match > shortest URI.
        """
        # Step 1: exact match
        exact = self.get_entity_by_label(name)
        if exact:
            return exact

        # Step 2: fulltext search
        candidates = self.search_entities_fulltext(name, entity_type, limit=5)
        if not candidates:
            return None

        # Step 3: disambiguate
        name_lower = name.lower().strip()
        # Prefer exact label match among candidates
        for c in candidates:
            if c.label.lower().strip() == name_lower:
                return c
        # When entity_type is specified, prefer type-matching candidates
        if entity_type:
            type_short = entity_type.replace('dbo:', '')
            for c in candidates:
                try:
                    types = self.get_entity_types(c.uri)
                    if any(type_short in t for t in types):
                        return c
                except Exception:
                    continue
        # Fallback: shortest URI (less likely to be a disambiguation/list page)
        return min(candidates, key=lambda e: len(e.uri))

    def get_wikipedia_page_id(self, entity_uri: str) -> Optional[str]:
        """Get Wikipedia page ID for a DBpedia entity."""
        if not entity_uri.startswith("http"):
            entity_uri = f"http://dbpedia.org/resource/{entity_uri.replace(' ', '_')}"

        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT ?wikiPageID
        WHERE {{
            <{safe_uri}> dbo:wikiPageID ?wikiPageID .
        }}
        LIMIT 1
        """
        results = self.query(query)
        bindings = results.get("results", {}).get("bindings", [])

        if bindings:
            return bindings[0]["wikiPageID"]["value"]
        return None
