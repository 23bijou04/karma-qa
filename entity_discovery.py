"""Type-guided entity discovery for TEXT route.

Deterministic SPARQL-based discovery using wikiPageWikiLink + type filtering.
No LLM calls. Used when structured predicates are missing (TEXT route scenario).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dbpedia.sparql_utils import escape_sparql_uri
from models import Entity

if TYPE_CHECKING:
    from schema_projection import SchemaProjection

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """Discovery candidate with relevance score."""

    uri: str
    label: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _discover_linked_entities(
    seed_uri: str,
    expected_type: str | None,
    kg_client,
    limit: int = 30,
) -> list[dict]:
    """Fetch wikiPageWikiLink neighbors, optionally filtered by type."""
    safe_uri = escape_sparql_uri(seed_uri)
    type_filter = ""
    if expected_type:
        type_filter = f"?entity a <http://dbpedia.org/ontology/{expected_type}> ."

    query = f"""
    SELECT DISTINCT ?entity ?label WHERE {{
        <{safe_uri}> <http://dbpedia.org/ontology/wikiPageWikiLink> ?entity .
        {type_filter}
        ?entity rdfs:label ?label .
        FILTER(LANG(?label) = "en")
        FILTER(STRSTARTS(STR(?entity), "http://dbpedia.org/resource/"))
        FILTER(!STRSTARTS(STR(?entity), "http://dbpedia.org/resource/Category:"))
    }} LIMIT {limit}
    """
    try:
        raw = kg_client.query(query)
        return raw.get("results", {}).get("bindings", [])
    except Exception as e:
        logger.warning("Discovery query failed for %s: %s", seed_uri, e)
        return []


def _score_candidates(
    candidates: list[dict],
    seed_uri: str,
    question: str,
    hop2_property: str | None,
    actual_uris: set[str],
    kg_client,
    ask_limit: int = 10,
) -> list[ScoredCandidate]:
    """Score discovery candidates by relevance signals.

    Two-pass: cheap signals first, then SPARQL ASK only on top candidates.
    """
    q_words = set(re.findall(r'\w+', question.lower()))
    stop_words = {"the", "of", "a", "an", "in", "is", "was", "who", "what", "where", "when"}
    scored = []

    # Pass 1: cheap signals
    for cand in candidates:
        uri = cand["entity"]["value"]
        label = cand.get("label", {}).get("value", "")
        sc = ScoredCandidate(uri=uri, label=label)

        label_words = set(re.findall(r'\w+', label.lower()))
        overlap = q_words & label_words - stop_words
        if overlap:
            sc.score += 1
            sc.reasons.append(f"label_overlap:{overlap}")

        if uri in actual_uris:
            sc.score -= 2
            sc.reasons.append("already_known")

        scored.append(sc)

    # Pass 2: SPARQL ASK on top candidates only
    scored.sort(key=lambda s: s.score, reverse=True)
    for sc in scored[:ask_limit]:
        safe_uri = escape_sparql_uri(sc.uri)
        safe_seed = escape_sparql_uri(seed_uri)

        try:
            ask = (
                f"ASK {{ <{safe_uri}> "
                f"<http://dbpedia.org/ontology/wikiPageWikiLink> <{safe_seed}> }}"
            )
            if kg_client.query(ask).get("boolean", False):
                sc.score += 3
                sc.reasons.append("backlink")
        except Exception:
            pass

        if hop2_property:
            try:
                ask = (
                    f"ASK {{ <{safe_uri}> "
                    f"<http://dbpedia.org/ontology/{hop2_property}> ?x }}"
                )
                if kg_client.query(ask).get("boolean", False):
                    sc.score += 2
                    sc.reasons.append(f"has_{hop2_property}")
            except Exception:
                pass

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def discover_context_entities(
    seed_entities: list[Entity],
    projections: list[SchemaProjection],
    kg_client,
    question: str = "",
    max_per_seed: int = 5,
) -> list[Entity]:
    """Discover context entities via type-guided wikiPageWikiLink.

    Deterministic — no LLM calls. Uses SchemaProjection's expected_target_type
    and hop2 properties for type filtering and scoring.

    Fallback chain:
      1. wikiPageWikiLink + type filter
      2. wikiPageWikiLink (no type)
      3. Return seed entities only

    Returns: seed_entities + discovered entities (deduplicated by URI).
    """
    if not projections:
        return list(seed_entities)

    analysis = projections[0].question_analysis
    expected_type = analysis.expected_target_type
    hop2_property = analysis.bridge_property

    # Collect known URIs for penalty
    actual_uris: set[str] = set()
    for proj in projections:
        for val in proj.entity_profile.actual_properties.get("values", {}).values():
            if isinstance(val, str) and val.startswith("http://"):
                actual_uris.add(val)

    all_entities = list(seed_entities)
    seen_uris = {e.uri for e in seed_entities if e.uri}

    for entity in seed_entities:
        if not entity.uri:
            continue

        # Try type-filtered discovery
        candidates = _discover_linked_entities(entity.uri, expected_type, kg_client)

        # Fallback: unfiltered
        if not candidates and expected_type:
            candidates = _discover_linked_entities(entity.uri, None, kg_client)

        if not candidates:
            continue

        scored = _score_candidates(
            candidates, entity.uri, question, hop2_property, actual_uris, kg_client,
        )

        for sc in scored[:max_per_seed]:
            if sc.uri in seen_uris:
                continue
            seen_uris.add(sc.uri)
            top_reason = sc.reasons[0] if sc.reasons else "wikiPageWikiLink"
            all_entities.append(Entity(
                uri=sc.uri,
                label=sc.label,
                source="sparql_neighbor",
                discovery_relation=top_reason,
            ))

    return all_entities
