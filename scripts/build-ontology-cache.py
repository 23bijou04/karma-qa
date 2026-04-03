#!/usr/bin/env python3
"""Build ontology cache from DBpedia SPARQL endpoint.

Queries rdfs:subClassOf, rdfs:domain, rdfs:range for all dbo: classes/properties,
resolves inheritance transitively, flattens range types to canonical superclasses,
and outputs data/ontology_cache.json.

Usage:
    python scripts/build-ontology-cache.py [--endpoint URL] [--output PATH]
"""

import argparse
import json
import logging
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from SPARQLWrapper import JSON, SPARQLWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Classes we treat as "anchor" types for bridge detection and range flattening.
# Fine-grained classes (Actor, SoccerPlayer, City) map to their nearest
# canonical ancestor via rdfs:subClassOf BFS.
CANONICAL_CLASSES = {
    "Person", "Place", "Organisation", "Film", "Book",
    "MusicalWork", "Event", "Species", "Device",
}

DEFAULT_ENDPOINT = "https://dbpedia.org/sparql"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "ontology_cache.json"


def _query(sparql: SPARQLWrapper, query_text: str) -> list[dict]:
    """Execute SPARQL query and return bindings list."""
    sparql.setQuery(query_text)
    results = sparql.query().convert()
    return results.get("results", {}).get("bindings", [])


def fetch_subclass_edges(sparql: SPARQLWrapper) -> dict[str, list[str]]:
    """Fetch all rdfs:subClassOf edges between dbo: classes.

    Returns:
        parent_map: {child_class: [parent_class, ...]}
    """
    query = """
    SELECT ?child ?parent WHERE {
        ?child rdfs:subClassOf ?parent .
        FILTER(STRSTARTS(STR(?child), "http://dbpedia.org/ontology/"))
        FILTER(STRSTARTS(STR(?parent), "http://dbpedia.org/ontology/"))
    }
    """
    parent_map: dict[str, list[str]] = {}
    for row in _query(sparql, query):
        child = row["child"]["value"].split("/")[-1]
        parent = row["parent"]["value"].split("/")[-1]
        if child != parent:  # skip self-loops
            parent_map.setdefault(child, []).append(parent)
    logger.info("Fetched %d subClassOf edges", sum(len(v) for v in parent_map.values()))
    return parent_map


def fetch_property_definitions(sparql: SPARQLWrapper) -> list[dict]:
    """Fetch rdfs:domain, rdfs:range, rdf:type for all dbo: properties.

    Returns list of dicts with keys: property, domain (optional), range (optional),
    prop_type ("object" or "datatype").
    """
    # Query object properties (range is a dbo: class)
    obj_query = """
    SELECT ?prop ?domain ?range WHERE {
        ?prop a owl:ObjectProperty .
        FILTER(STRSTARTS(STR(?prop), "http://dbpedia.org/ontology/"))
        OPTIONAL {
            ?prop rdfs:domain ?domain .
            FILTER(STRSTARTS(STR(?domain), "http://dbpedia.org/ontology/"))
        }
        OPTIONAL {
            ?prop rdfs:range ?range .
            FILTER(STRSTARTS(STR(?range), "http://dbpedia.org/ontology/"))
        }
    }
    """
    # Query datatype properties (range is xsd: or literal)
    dt_query = """
    SELECT ?prop ?domain WHERE {
        ?prop a owl:DatatypeProperty .
        FILTER(STRSTARTS(STR(?prop), "http://dbpedia.org/ontology/"))
        OPTIONAL {
            ?prop rdfs:domain ?domain .
            FILTER(STRSTARTS(STR(?domain), "http://dbpedia.org/ontology/"))
        }
    }
    """
    props = []
    for row in _query(sparql, obj_query):
        entry = {
            "property": row["prop"]["value"].split("/")[-1],
            "prop_type": "object",
        }
        if "domain" in row:
            entry["domain"] = row["domain"]["value"].split("/")[-1]
        if "range" in row:
            entry["range"] = row["range"]["value"].split("/")[-1]
        props.append(entry)

    for row in _query(sparql, dt_query):
        entry = {
            "property": row["prop"]["value"].split("/")[-1],
            "prop_type": "datatype",
        }
        if "domain" in row:
            entry["domain"] = row["domain"]["value"].split("/")[-1]
        props.append(entry)

    logger.info("Fetched %d property definitions", len(props))
    return props


def get_all_ancestors(cls: str, parent_map: dict[str, list[str]]) -> list[str]:
    """Get all transitive ancestors of a class via BFS (avoids cycles)."""
    ancestors = []
    visited = {cls}
    queue = deque(parent_map.get(cls, []))
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        ancestors.append(current)
        queue.extend(parent_map.get(current, []))
    return ancestors


def find_canonical_superclass(
    cls: str, parent_map: dict[str, list[str]], canonical: set[str],
) -> str | None:
    """BFS up the class hierarchy to find nearest canonical ancestor."""
    if cls in canonical:
        return cls
    visited = set()
    queue = deque([cls])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current in canonical:
            return current
        queue.extend(parent_map.get(current, []))
    return None


def build_ontology_cache(endpoint: str) -> dict:
    """Build complete ontology cache from DBpedia SPARQL endpoint."""
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "DBpediaQA/1.0 (research)")
    sparql.setTimeout(30)

    # Step 1: Fetch class hierarchy
    parent_map = fetch_subclass_edges(sparql)

    # Step 2: Fetch property definitions (domain, range, type)
    prop_defs = fetch_property_definitions(sparql)

    # Step 3: Group properties by domain class
    # Compact format: {class: {prop: "RangeClass" or null}}
    # - "Person" means object property with range Person
    # - null means datatype property (no entity range)
    class_own_props: dict[str, dict[str, str | None]] = {}
    for pdef in prop_defs:
        domain = pdef.get("domain")
        if not domain:
            continue  # orphan property — no domain, handled by instance merge
        prop_name = pdef["property"]
        if pdef["prop_type"] == "object":
            prop_info = pdef.get("range")  # class name or None
        else:
            prop_info = None
        class_own_props.setdefault(domain, {})[prop_name] = prop_info

    # Collect all classes that appear in hierarchy or have properties
    all_classes = set(parent_map.keys())
    for parents in parent_map.values():
        all_classes.update(parents)
    all_classes.update(class_own_props.keys())
    # Remove overly generic classes
    all_classes.discard("Thing")
    logger.info("Total classes discovered: %d", len(all_classes))

    # Step 4: Build superclass_map (fine-grained → canonical)
    superclass_map: dict[str, str] = {}
    for cls in sorted(all_classes):
        canonical = find_canonical_superclass(cls, parent_map, CANONICAL_CLASSES)
        if canonical and canonical != cls:
            superclass_map[cls] = canonical

    # Step 5: Resolve inheritance — each class gets own + ancestor properties
    classes_output: dict[str, dict] = {}
    for cls in sorted(all_classes):
        ancestors = get_all_ancestors(cls, parent_map)
        # Merge properties: own first, then ancestors (own wins on conflict)
        merged_props: dict[str, str | None] = {}
        # Ancestors first (lower priority)
        for ancestor in reversed(ancestors):
            if ancestor in class_own_props:
                merged_props.update(class_own_props[ancestor])
        # Own properties override
        if cls in class_own_props:
            merged_props.update(class_own_props[cls])

        if not merged_props:
            continue  # skip classes with no properties

        # Flatten range types to canonical superclasses
        for prop_name in merged_props:
            raw_range = merged_props[prop_name]
            if raw_range:
                canonical = find_canonical_superclass(
                    raw_range, parent_map, CANONICAL_CLASSES,
                )
                if canonical:
                    merged_props[prop_name] = canonical

        # Direct parents (not full ancestor chain)
        direct_parents = sorted(set(parent_map.get(cls, [])) - {"Thing"})

        classes_output[cls] = {
            "parents": direct_parents,
            "properties": dict(sorted(merged_props.items())),
        }

    logger.info("Classes with properties: %d", len(classes_output))

    # Count unique properties
    all_prop_names = set()
    for cls_data in classes_output.values():
        all_prop_names.update(cls_data["properties"].keys())
    logger.info("Unique properties: %d", len(all_prop_names))

    return {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endpoint": endpoint,
            "class_count": len(classes_output),
            "property_count": len(all_prop_names),
        },
        "superclass_map": dict(sorted(superclass_map.items())),
        "classes": classes_output,
    }


def validate_cache(cache: dict) -> bool:
    """Validate ontology cache has expected content."""
    ok = True
    classes = cache["classes"]

    if len(classes) < 100:
        logger.error("Only %d classes — expected ≥100", len(classes))
        ok = False

    # Check key classes have expected properties
    # Note: some properties (e.g. country) have domain on subclasses
    # (PopulatedPlace, not Place) — warn but don't fail.
    checks = {
        "Film": ["director", "starring"],
        "Person": ["birthDate", "birthPlace"],
    }
    for cls, expected_props in checks.items():
        if cls not in classes:
            logger.error("Missing class: %s", cls)
            ok = False
            continue
        props = set(classes[cls]["properties"].keys())
        for p in expected_props:
            if p not in props:
                logger.error("%s missing property: %s", cls, p)
                ok = False

    # Check Film.director range is Person (compact format: value is string)
    director_range = classes.get("Film", {}).get("properties", {}).get("director")
    if director_range != "Person":
        logger.warning("Film.director range = %s, expected Person", director_range)

    # Check superclass_map
    smap = cache["superclass_map"]
    expected_maps = {"Actor": "Person", "City": "Place"}
    for child, expected_parent in expected_maps.items():
        if child in smap and smap[child] != expected_parent:
            logger.warning(
                "superclass_map[%s] = %s, expected %s",
                child, smap[child], expected_parent,
            )

    return ok


def main():
    parser = argparse.ArgumentParser(description="Build DBpedia ontology cache")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    logger.info("Querying %s ...", args.endpoint)
    cache = build_ontology_cache(args.endpoint)

    if not validate_cache(cache):
        logger.error("Validation failed — output may be incomplete")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=False)

    size_kb = output_path.stat().st_size / 1024
    logger.info("Written %s (%.0f KB, %d classes)", output_path, size_kb, len(cache["classes"]))


if __name__ == "__main__":
    main()
