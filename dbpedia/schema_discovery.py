"""Hybrid schema discovery for DBpedia entities.

Combines pre-computed ontology cache (734 classes from rdfs:domain/range/subClassOf)
with dynamic property discovery for instance-level data.
"""

import logging
from typing import Optional

from SPARQLWrapper import JSON, SPARQLWrapper

from .ontology_loader import (
    _RAW_CACHE,
    build_cached_class_schemas,
    build_property_ranges,
    build_property_to_classes,
    load_superclass_map,
)
from .sparql_utils import escape_sparql_string, escape_sparql_uri

logger = logging.getLogger(__name__)

# Auto-generated from data/ontology_cache.json (built by scripts/build-ontology-cache.py)
# Same shape callers expect: {cls: {"dbo": [...], "dbp": [], "descriptions": {}, "dbo_2hop": {...}}}
CACHED_CLASS_SCHEMAS: dict[str, dict] = build_cached_class_schemas(_RAW_CACHE)

# "dbo:director" -> "Person", auto-derived from ontology ranges
PROPERTY_RANGES: dict[str, str] = build_property_ranges(_RAW_CACHE)

# "director" -> ["Film", ...], auto-derived from CACHED_CLASS_SCHEMAS
PROPERTY_TO_CLASSES: dict[str, list[str]] = build_property_to_classes(CACHED_CLASS_SCHEMAS)

# Fine-grained class -> canonical superclass (e.g., "Actor" -> "Person")
SUPERCLASS_MAP: dict[str, str] = load_superclass_map(_RAW_CACHE)

# Classes too generic to be useful as primary type or range target
GENERIC_CLASSES: frozenset[str] = frozenset({
    "Agent", "Thing", "TopicalConcept", "Species", "Eukaryote",
})


def format_2hop_lines(cached: dict, max_chains: int = 5, max_props: int = 4) -> list[str]:
    """Format 2-hop chain info from a cached class schema entry.

    Shared by _build_schema_text() and sparql_evaluation.get_schema_for_uris().
    Returns list of lines (without trailing newline).
    """
    dbo_2hop = cached.get("dbo_2hop")
    if not dbo_2hop:
        return []
    lines = ["  2-Hop chains:"]
    for prop, (target_cls, target_props) in list(dbo_2hop.items())[:max_chains]:
        lines.append(
            f"    - dbo:{prop} -> {target_cls}: "
            f"{', '.join(target_props[:max_props])}"
        )
    return lines


class SchemaDiscovery:
    """Hybrid schema discovery combining cached ontology with dynamic lookup.

    NOT thread-safe: self.sparql is a shared SPARQLWrapper instance whose
    setQuery() mutates internal state. All methods must be called sequentially.
    """

    def __init__(self, sparql_endpoint: str = "https://dbpedia.org/sparql"):
        self.endpoint = sparql_endpoint
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", "DBpediaQA/1.0 (research)")
        self._schema_cache: dict[str, dict] = {}
        self._range_cache: dict[str, dict[str, str]] = {}

    def get_entity_class(self, entity_uri: str) -> Optional[str]:
        """Get the primary DBpedia class for an entity."""
        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT ?type WHERE {{
            <{safe_uri}> a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        """
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            for row in bindings:
                class_name = row["type"]["value"].split("/")[-1]
                if class_name in CACHED_CLASS_SCHEMAS:
                    return class_name
            if bindings:
                return bindings[0]["type"]["value"].split("/")[-1]
        except Exception as e:
            logger.warning(f"Error getting entity class: {e}")
        return None

    def get_all_cached_classes(self, entity_uri: str) -> tuple[list[str], list[str]]:
        """Get ALL dbo: types and filter to cached classes.

        Returns:
            (all_dbo_classes, cached_classes) where cached_classes
            is the subset present in CACHED_CLASS_SCHEMAS.
        """
        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT ?type WHERE {{
            <{safe_uri}> a ?type .
            FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
        }}
        """
        all_classes = []
        cached_classes = []
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            for row in bindings:
                class_name = row["type"]["value"].split("/")[-1]
                if class_name not in all_classes:
                    all_classes.append(class_name)
                if class_name in CACHED_CLASS_SCHEMAS and class_name not in cached_classes:
                    cached_classes.append(class_name)
        except Exception as e:
            logger.warning(f"Error getting all cached classes: {e}")
        return all_classes, cached_classes

    def get_actual_properties(
        self, entity_uri: str, property_filter: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """Query DBpedia for actual properties an entity has."""
        filter_clause = ""
        if property_filter:
            conditions = " || ".join([
                f'CONTAINS(LCASE(STR(?property)), "{escape_sparql_string(p.lower())}")'
                for p in property_filter
            ])
            filter_clause = f"FILTER({conditions})"

        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT ?property (SAMPLE(?value) AS ?sampleValue) WHERE {{
            <{safe_uri}> ?property ?value .
            FILTER(
                STRSTARTS(STR(?property), "http://dbpedia.org/ontology/") ||
                STRSTARTS(STR(?property), "http://dbpedia.org/property/")
            )
            FILTER(!CONTAINS(STR(?property), "wikiPage"))
            FILTER(!CONTAINS(STR(?property), "abstract"))
            FILTER(!CONTAINS(STR(?property), "thumbnail"))
            FILTER(!CONTAINS(STR(?property), "isPrimaryTopicOf"))
            FILTER(!CONTAINS(STR(?property), "description"))
            {filter_clause}
        }}
        GROUP BY ?property
        LIMIT 100
        """
        result = {"dbo": [], "dbp": [], "values": {}}
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            for row in results.get("results", {}).get("bindings", []):
                prop_uri = row["property"]["value"]
                prop_name = prop_uri.split("/")[-1]
                value = row["sampleValue"]["value"]
                if row["sampleValue"]["type"] == "uri":
                    value = value.split("/")[-1].replace("_", " ")
                if "/ontology/" in prop_uri:
                    if prop_name not in result["dbo"]:
                        result["dbo"].append(prop_name)
                        result["values"][f"dbo:{prop_name}"] = value
                else:
                    if prop_name not in result["dbp"]:
                        result["dbp"].append(prop_name)
                        result["values"][f"dbp:{prop_name}"] = value
        except Exception as e:
            logger.warning(f"Error getting actual properties: {e}")
        return result

    def discover_property_ranges(self, entity_uri: str) -> dict[str, str]:
        """Discover target class types for entity-valued dbo: properties.

        Rare fallback — only fires when entity has object properties with no
        range in the ontology cache. Most types now covered by 734-class cache.
        """
        if entity_uri in self._range_cache:
            return self._range_cache[entity_uri]

        safe_uri = escape_sparql_uri(entity_uri)
        query = f"""
        SELECT DISTINCT ?prop ?targetType WHERE {{
            <{safe_uri}> ?prop ?target .
            ?target a ?targetType .
            FILTER(STRSTARTS(STR(?prop), "http://dbpedia.org/ontology/"))
            FILTER(STRSTARTS(STR(?targetType), "http://dbpedia.org/ontology/"))
            FILTER(!CONTAINS(STR(?prop), "wikiPage"))
        }}
        LIMIT 200
        """
        prop_types: dict[str, list[str]] = {}
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            for row in results.get("results", {}).get("bindings", []):
                prop_name = row["prop"]["value"].split("/")[-1]
                type_name = row["targetType"]["value"].split("/")[-1]
                prop_types.setdefault(prop_name, []).append(type_name)
        except Exception as e:
            logger.warning("Error discovering property ranges for %s: %s", entity_uri, e)
            self._range_cache[entity_uri] = {}
            return {}

        ranges: dict[str, str] = {}
        for prop, types in prop_types.items():
            cached_hit = [t for t in types if t in CACHED_CLASS_SCHEMAS]
            if cached_hit:
                ranges[prop] = cached_hit[0]
                continue
            specific = [t for t in types if t not in GENERIC_CLASSES]
            if specific:
                ranges[prop] = specific[0]

        self._range_cache[entity_uri] = ranges
        return ranges
