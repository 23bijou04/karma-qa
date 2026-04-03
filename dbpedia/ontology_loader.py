"""Load pre-computed ontology cache and build backward-compatible schema dicts.

Converts data/ontology_cache.json (compact format) into the same shape
callers expect from the old CACHED_CLASS_SCHEMAS:
  {cls: {"dbo": [...], "dbp": [], "descriptions": {}, "dbo_2hop": {...}}}

Also auto-generates PROPERTY_RANGES and PROPERTY_TO_CLASSES.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the pre-computed ontology cache (built by scripts/build-ontology-cache.py)
_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "ontology_cache.json"

# Common bridging properties sorted by priority for 2-hop chain display.
# These appear first in chain property lists so format_2hop_lines shows
# useful properties (birthDate, country) instead of alphabetical noise.
_PRIORITY_PROPS = [
    "birthDate", "deathDate", "birthPlace", "deathPlace", "nationality",
    "foundingDate", "foundedBy", "country", "capital", "isPartOf",
    "populationTotal", "locationCity", "locationCountry",
    "director", "author", "starring", "producer", "spouse",
    "parent", "father", "mother", "child",
    "genre", "releaseDate", "recordLabel",
]
_PRIORITY_SET = set(_PRIORITY_PROPS)


def _priority_sort_props(props: list[str]) -> list[str]:
    """Sort properties: priority props first (in priority order), then rest alphabetically."""
    priority = [p for p in _PRIORITY_PROPS if p in props]
    rest = sorted(p for p in props if p not in _PRIORITY_SET)
    return priority + rest


def _load_raw_cache(path: Path = _CACHE_PATH) -> dict | None:
    """Load raw JSON cache. Returns None if file missing."""
    if not path.exists():
        logger.warning("Ontology cache not found at %s — using empty cache", path)
        return None
    with open(path) as f:
        return json.load(f)


def _enrich_class_properties(raw_classes: dict) -> dict[str, dict[str, str | None]]:
    """Build enriched property dicts: ontology properties + orphan supplements.

    Returns: {class_name: {prop_name: range_class_or_none}}
    Used for both dbo list building and 2-hop chain construction.
    """
    enriched = {}
    for cls_name, cls_data in raw_classes.items():
        props = dict(cls_data.get("properties", {}))
        # Add orphan properties that lack rdfs:domain in ontology
        for prop in _ORPHAN_CLASS_PROPERTIES.get(cls_name, []):
            if prop not in props:
                range_key = f"dbo:{prop}"
                props[prop] = _ORPHAN_PROPERTY_RANGES.get(range_key)
        # Override ontology ranges that are too broad for QA (e.g. Agent → Person)
        for prop in list(props):
            range_key = f"dbo:{prop}"
            if range_key in _ORPHAN_PROPERTY_RANGES:
                props[prop] = _ORPHAN_PROPERTY_RANGES[range_key]
        enriched[cls_name] = props
    return enriched


def _build_2hop_chains(
    cls_props: dict[str, str | None],
    enriched_classes: dict[str, dict[str, str | None]],
) -> dict:
    """Build dbo_2hop dict for a class from its properties.

    For each object property with a known range class, look up that class's
    enriched properties (ontology + orphans) and include as chain target.
    """
    chains = {}
    for prop_name, range_cls in cls_props.items():
        if not range_cls:
            continue  # datatype property, no chain
        target_props_dict = enriched_classes.get(range_cls, {})
        target_props = _priority_sort_props(list(target_props_dict.keys()))[:15]
        if target_props:
            chains[prop_name] = (range_cls, target_props)
    return chains


def build_cached_class_schemas(raw_cache: dict | None) -> dict:
    """Convert ontology cache to CACHED_CLASS_SCHEMAS-compatible dict.

    Each entry has: {"dbo": [...], "dbp": [], "descriptions": {}, "dbo_2hop": {...}}
    """
    if not raw_cache:
        return {}

    raw_classes = raw_cache.get("classes", {})
    # Enrich all class properties with orphan supplements BEFORE chain building,
    # so chain targets (e.g., Place) include orphan props (country, isPartOf)
    enriched = _enrich_class_properties(raw_classes)

    schemas = {}
    for cls_name, cls_props in enriched.items():
        dbo_props = _priority_sort_props(list(cls_props.keys()))
        dbo_2hop = _build_2hop_chains(cls_props, enriched)
        schemas[cls_name] = {
            "dbo": dbo_props,
            "dbp": [],  # ontology has no dbp: info; filled by instance data
            "descriptions": {},  # dropped — uniform treatment for all classes
            "dbo_2hop": dbo_2hop,
        }
    return schemas


# Critical properties that lack rdfs:domain/range in the DBpedia ontology
# but are widely used in QA. Without these, bridge detection and routing break.
_ORPHAN_PROPERTY_RANGES: dict[str, str] = {
    "dbo:country": "Place",
    "dbo:isPartOf": "Place",
    "dbo:leader": "Person",
    "dbo:foundedBy": "Person",
    "dbo:leaderName": "Person",
    "dbo:father": "Person",
    "dbo:mother": "Person",
    "dbo:distributor": "Organisation",
    # Ontology declares Agent range but QA needs the specific subclass
    "dbo:producer": "Person",
    "dbo:artist": "Person",
    "dbo:associatedMusicalArtist": "MusicalArtist",
}

# Orphan properties per class: widely used in QA but lack rdfs:domain in ontology.
# These are merged into CACHED_CLASS_SCHEMAS dbo lists and affect PROPERTY_TO_CLASSES,
# _score_class(), and bridge detection.
_ORPHAN_CLASS_PROPERTIES: dict[str, list[str]] = {
    "Person": ["occupation", "religion", "father", "mother"],
    "Film": ["releaseDate", "distributor", "country", "budget", "gross"],
    "MusicalArtist": ["genre", "recordLabel", "associatedMusicalArtist",
                       "activeYearsStartYear", "instrument"],
    "MusicalWork": ["genre", "recordLabel", "releaseDate", "album"],
    "Place": ["country", "isPartOf", "populationTotal", "areaTotal",
              "leader", "leaderName", "capital", "largestCity"],
    "Book": ["publicationDate", "isbn", "literaryGenre", "language"],
    "Company": ["foundedBy", "foundingDate", "industry", "numberOfEmployees"],
}


def build_property_ranges(raw_cache: dict | None) -> dict[str, str]:
    """Auto-derive PROPERTY_RANGES from ontology ranges + orphan supplements.

    Returns: {"dbo:director": "Person", "dbo:birthPlace": "Place", ...}
    """
    if not raw_cache:
        return dict(_ORPHAN_PROPERTY_RANGES)

    ranges: dict[str, str] = {}
    for cls_data in raw_cache.get("classes", {}).values():
        for prop_name, range_cls in cls_data.get("properties", {}).items():
            if range_cls:
                key = f"dbo:{prop_name}"
                if key not in ranges:
                    ranges[key] = range_cls

    # Apply orphan ranges — these override ontology when set (e.g. Agent → Person)
    for key, cls in _ORPHAN_PROPERTY_RANGES.items():
        ranges[key] = cls

    return dict(sorted(ranges.items()))


def build_property_to_classes(schemas: dict) -> dict[str, list[str]]:
    """Auto-derive PROPERTY_TO_CLASSES from schemas.

    Returns: {"director": ["Film", ...], "birthDate": ["Person", ...]}
    """
    mapping: dict[str, list[str]] = {}
    for cls_name, schema in schemas.items():
        for prop in schema.get("dbo", []):
            mapping.setdefault(prop, []).append(cls_name)
    return {k: sorted(set(v)) for k, v in sorted(mapping.items())}


def load_superclass_map(raw_cache: dict | None) -> dict[str, str]:
    """Load fine-grained → canonical class mapping."""
    if not raw_cache:
        return {}
    return raw_cache.get("superclass_map", {})


# Module-level loading — runs once at import time
_RAW_CACHE = _load_raw_cache()
