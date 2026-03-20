"""QuestionAnalysis: deterministic keyword matching for question routing.

Consolidates 6 scattered keyword-to-property maps into KEYWORD_TO_PROPERTIES
and provides analyze_question() for structured question metadata.
No LLM calls — purely regex + dictionary lookup.
"""

import re
from dataclasses import dataclass, field

from dbpedia_qa.dbpedia.schema_discovery import (
    CACHED_CLASS_SCHEMAS,
    PROPERTY_RANGES,
    PROPERTY_TO_CLASSES,
)

# Consolidated keyword -> property names (without prefix).
KEYWORD_TO_PROPERTIES: dict[str, list[str]] = {
    # Roles (bridge patterns)
    "director": ["director"], "directed": ["director"], "direct": ["director"],
    "producer": ["producer"], "produced": ["producer"],
    "writer": ["writer", "author"], "wrote": ["writer", "author"],
    "written": ["writer", "author"],
    "author": ["author"],
    "composer": ["musicComposer"],
    "starring": ["starring"], "actor": ["starring"], "actress": ["starring"],
    "founded": ["foundedBy", "foundingDate", "foundingYear"],
    "founder": ["foundedBy"],
    "performer": ["artist"], "performed": ["artist"],
    # Family
    "father": ["father", "parent"], "mother": ["mother", "parent"],
    "spouse": ["spouse"], "married": ["spouse"],
    "wife": ["spouse"], "husband": ["spouse"],
    "grandfather": ["father", "parent"], "grandmother": ["mother", "parent"],
    "parent": ["parent"], "paternal": ["parent"], "maternal": ["parent"],
    # Life events
    "born": ["birthDate", "birthPlace"],
    "birth": ["birthDate", "birthPlace"],
    "died": ["deathDate", "deathPlace"],
    "death": ["deathDate", "deathPlace"],
    "cause": ["deathCause"],
    # Publication/release
    "published": ["publicationDate"],
    "publication": ["publicationDate"],
    "released": ["releaseDate", "publicationDate"],
    # Geography
    "capital": ["capital"],
    "population": ["populationTotal"],
    "nationality": ["nationality", "citizenship"],
    "country": ["country", "nationality"],
    "citizen": ["nationality", "citizenship"],
    "citizenship": ["nationality", "citizenship"],
    "origin": ["country"],
    # Education
    "education": ["almaMater"],
    "study": ["almaMater"], "studied": ["almaMater"],
    "attended": ["almaMater"],
    "educated": ["almaMater"],
    # Career/recognition
    "award": ["award"], "awarded": ["award"],
    "occupation": ["occupation"],
    "genre": ["genre"],
    "language": ["language"],
}

COMPARISON_WORDS: set[str] = {
    "earlier", "later", "older", "younger", "first", "before", "after",
    "same", "both", "more", "less", "longer", "shorter",
}

# Bridge keywords that can serve as hop-1 in a 2-hop chain.
_BRIDGE_KEYWORDS: set[str] = {
    "director", "directed", "direct", "producer", "produced",
    "writer", "wrote", "written", "author", "composer",
    "starring", "actor", "actress",
    "founded", "founder",
    "father", "mother", "parent", "paternal", "maternal",
    "performer",
    "spouse", "married",
    "wife", "husband",
    "grandfather", "grandmother",
}

# Hop-2 keywords that indicate the question asks about a property of the hop-1 target.
_HOP2_KEYWORDS: set[str] = {
    "born", "birth", "died", "death", "country", "nationality", "capital",
    "citizenship", "citizen",
    "education", "studied", "attended", "educated",
    "spouse", "wife", "husband",
}


@dataclass
class QuestionAnalysis:
    """Per-question analysis: what properties/types are needed."""

    target_keywords: list[str]
    target_properties: frozenset[str]  # {"dbo:director", "dbp:director", ...}
    expected_target_type: str | None  # "Person" (from PROPERTY_TO_CLASSES)
    hop_structure: str  # "direct" | "bridge" | "comparison" | "compositional"
    is_comparison: bool
    bridge_keyword: str | None  # "director" for "who directed X"
    bridge_property: str | None  # "director" (property name, no prefix)
    hop2_keyword: str | None  # "born" for "where was director born"
    hop2_properties: frozenset[str] | None  # {"dbo:birthDate", "dbo:birthPlace"}
    target_keywords_set: frozenset[str] = field(default_factory=frozenset)


def analyze_question(
    question: str,
    entity_profiles: list | None = None,
) -> QuestionAnalysis:
    """Analyze question to extract target properties and hop structure.

    No LLM calls — purely deterministic keyword matching + regex.

    Args:
        question: Raw question text
        entity_profiles: Optional EntityProfiles (reserved for future use)

    Returns:
        QuestionAnalysis with structured question metadata.
    """
    q_lower = question.lower()

    # 1. Keyword extraction (longest-first to avoid partial matches)
    sorted_keywords = sorted(KEYWORD_TO_PROPERTIES.keys(), key=len, reverse=True)
    matched_keywords: list[str] = []
    for kw in sorted_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", q_lower):
            matched_keywords.append(kw)

    # Collapse compound keywords to avoid misclassification
    # "cause of death" → keep only "cause" (maps to deathCause)
    if "cause" in matched_keywords and "death" in matched_keywords:
        matched_keywords = [kw for kw in matched_keywords if kw != "death"]

    # 2. Property expansion
    all_property_names: set[str] = set()
    target_props: set[str] = set()
    for kw in matched_keywords:
        for prop_name in KEYWORD_TO_PROPERTIES[kw]:
            all_property_names.add(prop_name)
            target_props.add(f"dbo:{prop_name}")
            target_props.add(f"dbp:{prop_name}")

    # 3. Expected target type via PROPERTY_TO_CLASSES
    type_votes: dict[str, int] = {}
    for prop_name in all_property_names:
        for cls in PROPERTY_TO_CLASSES.get(prop_name, []):
            type_votes[cls] = type_votes.get(cls, 0) + 1
    expected_type = max(type_votes, key=type_votes.get) if type_votes else None

    # 4. Hop structure detection
    is_comparison = any(re.search(rf"\b{w}\b", q_lower) for w in COMPARISON_WORDS)

    # Bridge detection: keyword in _BRIDGE_KEYWORDS AND a hop2 keyword also matched
    bridge_keyword = None
    bridge_property = None
    hop2_keyword = None
    hop2_properties = None

    matched_bridge = [kw for kw in matched_keywords if kw in _BRIDGE_KEYWORDS]
    matched_hop2 = [kw for kw in matched_keywords if kw in _HOP2_KEYWORDS]

    if matched_bridge and matched_hop2:
        # Try each bridge candidate; pick the first that leaves valid hop2 candidates
        # (avoids self-matching when a keyword is in both _BRIDGE and _HOP2)
        for candidate in matched_bridge:
            hop2_candidates = [kw for kw in matched_hop2 if kw != candidate]
            if hop2_candidates:
                bridge_keyword = candidate
                bridge_props = KEYWORD_TO_PROPERTIES[bridge_keyword]
                bridge_property = bridge_props[0]
                hop2_keyword = hop2_candidates[0]
                h2_props: set[str] = set()
                for prop_name in KEYWORD_TO_PROPERTIES[hop2_keyword]:
                    h2_props.add(f"dbo:{prop_name}")
                    h2_props.add(f"dbp:{prop_name}")
                hop2_properties = frozenset(h2_props)
                break

    # For bridge questions, override expected_type with the PROPERTY_RANGES target
    # of the bridge property. E.g., "When was the director born?" -> bridge is
    # "director", PROPERTY_RANGES["dbo:director"] = "Person", so target = Person
    # (not MusicalArtist from raw vote count).
    if bridge_keyword and hop2_keyword:
        bridge_range_key = f"dbo:{bridge_props[0]}" if bridge_props else None
        if bridge_range_key and bridge_range_key in PROPERTY_RANGES:
            expected_type = PROPERTY_RANGES[bridge_range_key]

    # Determine hop_structure
    if bridge_keyword and hop2_keyword:
        hop_structure = "bridge"
    elif is_comparison:
        hop_structure = "comparison"
    elif len(matched_keywords) <= 1:
        hop_structure = "direct"
    else:
        hop_structure = "compositional"

    return QuestionAnalysis(
        target_keywords=matched_keywords,
        target_properties=frozenset(target_props),
        expected_target_type=expected_type,
        hop_structure=hop_structure,
        is_comparison=is_comparison,
        bridge_keyword=bridge_keyword,
        bridge_property=bridge_property,
        hop2_keyword=hop2_keyword,
        hop2_properties=hop2_properties,
        target_keywords_set=frozenset(matched_keywords),
    )


def enrich_analysis(
    base: QuestionAnalysis,
    llm_properties: list[str],
) -> QuestionAnalysis:
    """Create enriched QuestionAnalysis with LLM-extracted properties.

    When keyword matching misses the right properties (closed vocabulary),
    the LLM extracts property names from the question + schema. This function
    merges those into the base analysis and detects bridge structure.

    Args:
        base: Original keyword-based QuestionAnalysis
        llm_properties: Property names from LLM (no prefix), e.g. ["director", "birthDate"]

    Returns:
        New QuestionAnalysis with LLM properties merged in.
    """
    if not llm_properties:
        return base

    # Expand to prefixed property set
    new_props: set[str] = set(base.target_properties)
    for prop in llm_properties:
        new_props.add(f"dbo:{prop}")
        new_props.add(f"dbp:{prop}")

    # Detect bridge structure from LLM properties
    bridge_keyword = base.bridge_keyword
    bridge_property = base.bridge_property
    hop2_keyword = base.hop2_keyword
    hop2_properties = base.hop2_properties
    hop_structure = base.hop_structure
    expected_type = base.expected_target_type

    if not bridge_property and len(llm_properties) >= 2:
        # Score each candidate by how well its target class covers remaining props.
        # E.g., ["director","birthDate"]: director→Person has birthDate ✓ (score=1)
        #        birthPlace→Place doesn't have director (score=0) → director wins.
        best_candidate = None
        best_score = -1
        for prop in llm_properties:
            range_key = f"dbo:{prop}"
            if range_key not in PROPERTY_RANGES:
                continue
            target_cls = PROPERTY_RANGES[range_key]
            target_schema = CACHED_CLASS_SCHEMAS.get(target_cls, {})
            target_props = set(target_schema.get("dbo", []))
            remaining = [p for p in llm_properties if p != prop]
            score = sum(1 for p in remaining if p in target_props)
            if score > best_score:
                best_score = score
                best_candidate = prop
        if best_candidate and best_score > 0:
            bridge_keyword = best_candidate
            bridge_property = best_candidate
            expected_type = PROPERTY_RANGES[f"dbo:{best_candidate}"]
            remaining = [p for p in llm_properties if p != best_candidate]
            hop2_keyword = remaining[0]
            h2_set: set[str] = set()
            for p in remaining:
                h2_set.add(f"dbo:{p}")
                h2_set.add(f"dbp:{p}")
            hop2_properties = frozenset(h2_set)
            hop_structure = "bridge"

    return QuestionAnalysis(
        target_keywords=base.target_keywords,
        target_properties=frozenset(new_props),
        expected_target_type=expected_type,
        hop_structure=hop_structure,
        is_comparison=base.is_comparison,
        bridge_keyword=bridge_keyword,
        bridge_property=bridge_property,
        hop2_keyword=hop2_keyword,
        hop2_properties=hop2_properties,
        target_keywords_set=base.target_keywords_set,
    )
