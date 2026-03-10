"""Routing logic for hybrid QA: rule-based route + SPARQL ASK verification.

Extracted from hybrid_qa.py to break circular dependency with logicrag_decomposition.

Key functions:
  route_question()                    — Main entry: rules + ASK verification + LLM fallback
  _rule_based_route()                 — Deterministic property/chain matching (structured)
  verify_sparql_feasibility()         — SPARQL ASK to verify triples exist
  _is_comparison_question()           — Detect comparison questions
  _has_incomplete_entity_coverage()   — Detect missing entities in comparisons
  entities_to_schema_dicts()          — Convert Entity list to schema dicts
  _sparql_results_to_entities()       — Convert SPARQL result rows to Entity objects
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import unquote

from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS, PROPERTY_RANGES
from dbpedia_qa.models import Entity

if TYPE_CHECKING:
    from question_analysis import QuestionAnalysis
    from schema_projection import SchemaProjection

logger = logging.getLogger(__name__)

# Mapping from question keywords to DBpedia ontology property names.
# Used by verify_sparql_feasibility() (ASK).
BRIDGE_WORD_TO_PROP = {
    "director": "director", "directed": "director",
    "producer": "producer", "produced": "producer",
    "writer": "writer", "wrote": "writer",
    "author": "author",
    "composer": "musicComposer",
    "starring": "starring", "actor": "starring", "actress": "starring",
    "founded": "foundedBy", "founder": "foundedBy",
    "father": "parent", "mother": "parent",
}

ROUTE_PROMPT = """You are a knowledge graph query router for DBpedia. \
Decide if this question can be answered via SPARQL.

{schema}

QUESTION: {question}
{hint}
Reasoning steps:
1. What information does this question need? \
(a date, a name, a place, a comparison, etc.)
2. What property or relationship provides that information? \
(e.g., birthDate, director, foundedBy, nationality)
3. Does the schema above contain that property, \
or a 2-hop chain that leads to it?
   - Direct: the property exists on the entity \
(e.g., dbo:birthDate in "Actual properties")
   - 2-hop: a chain exists \
(e.g., "dbo:director -> Person: birthDate")
4. If a complete property path exists: SPARQL
5. If the required property is absent or the chain is incomplete: TEXT

Reply with EXACTLY one word: SPARQL or TEXT"""


# Property sets for comparison routing
_DATE_PROPERTIES = {
    "dbo:birthDate", "dbp:birthDate", "dbo:deathDate", "dbp:deathDate",
    "dbo:activeYearsStartYear", "dbp:established", "dbo:foundingYear",
    "dbp:foundingDate", "dbo:releaseDate", "dbp:released",
}
_COUNTRY_PROPERTIES = {
    "dbo:nationality", "dbp:nationality", "dbo:country", "dbp:country",
    "dbo:citizenship", "dbp:citizenship", "dbo:birthPlace", "dbp:birthPlace",
    "dbo:stateOfOrigin", "dbp:placeOfBirth",
}
_PARENT_PROPS = {"dbo:parent", "dbp:father", "dbp:mother"}


# ---------------------------------------------------------------------------
# Rule-based routing (structured — no regex parsing)
# ---------------------------------------------------------------------------

def _rule_based_route(
    analysis: QuestionAnalysis,
    projections: list[SchemaProjection],
    debug: bool = False,
) -> tuple[str, str] | None:
    """Return ('SPARQL', reason) if rules match, None if inconclusive."""
    # Collect actual properties across all entity projections
    all_actual: set[str] = set()
    for proj in projections:
        all_actual |= proj.entity_profile.actual_property_set

    # 1. Comparison + date properties -> SPARQL
    if analysis.is_comparison:
        if all_actual & _DATE_PROPERTIES:
            if debug:
                print(f"  [DEBUG] Rule matched: direct_date_comparison "
                      f"(have {sorted(all_actual & _DATE_PROPERTIES)})")
            return ("SPARQL", "rule:direct_date_comparison")

        country_keywords = {"country", "nationality", "citizen"}
        if (analysis.target_keywords_set & country_keywords) and (
            all_actual & _COUNTRY_PROPERTIES
        ):
            if debug:
                print(f"  [DEBUG] Rule matched: direct_country_comparison "
                      f"(have {sorted(all_actual & _COUNTRY_PROPERTIES)})")
            return ("SPARQL", "rule:direct_country_comparison")

    # 2. Comparison bridge chain: bridge keyword + date chain in schema
    if analysis.is_comparison and analysis.bridge_keyword:
        for proj in projections:
            if analysis.bridge_property and analysis.bridge_property in proj.relevant_chains:
                target_cls, target_props = proj.relevant_chains[analysis.bridge_property]
                if any(p in target_props for p in [
                    "birthDate", "deathDate", "foundingDate", "activeYearsStartYear",
                ]):
                    if debug:
                        print(f"  [DEBUG] Rule matched: bridge_comparison "
                              f"(chain: {analysis.bridge_property} -> {target_cls})")
                    return ("SPARQL", "rule:bridge_comparison")
            # Fallback: check PROPERTY_RANGES directly
            if analysis.bridge_property:
                range_key = f"dbo:{analysis.bridge_property}"
                if range_key in PROPERTY_RANGES:
                    target_cls = PROPERTY_RANGES[range_key]
                    if target_cls in CACHED_CLASS_SCHEMAS:
                        target_schema = CACHED_CLASS_SCHEMAS[target_cls]
                        if any(p in target_schema.get("dbo", []) for p in [
                            "birthDate", "deathDate", "foundingDate",
                        ]):
                            if debug:
                                print(f"  [DEBUG] Rule matched: bridge_comparison "
                                      f"(PROPERTY_RANGES: {range_key} -> {target_cls})")
                            return ("SPARQL", "rule:bridge_comparison")

    # 3. Non-comparison bridge: bridge_keyword + hop2_keyword
    if analysis.bridge_keyword and analysis.hop2_keyword:
        if analysis.bridge_property:
            bridge_full = f"dbo:{analysis.bridge_property}"
            if bridge_full in all_actual:
                if debug:
                    print(f"  [DEBUG] Rule matched: bridge_hop2 "
                          f"({analysis.bridge_keyword} -> {analysis.hop2_keyword})")
                return ("SPARQL", "rule:bridge_hop2")
            # Also check via PROPERTY_RANGES (schema may have the chain)
            if bridge_full in PROPERTY_RANGES:
                target_cls = PROPERTY_RANGES[bridge_full]
                if target_cls in CACHED_CLASS_SCHEMAS:
                    if debug:
                        print(f"  [DEBUG] Rule matched: bridge_hop2 "
                              f"({analysis.bridge_keyword} -> {target_cls})")
                    return ("SPARQL", "rule:bridge_hop2")

    # 4. Parent chain
    if analysis.bridge_keyword in ("father", "mother", "parent", "paternal", "maternal"):
        if all_actual & _PARENT_PROPS:
            if debug:
                print(f"  [DEBUG] Rule matched: parent_chain "
                      f"(have {sorted(all_actual & _PARENT_PROPS)})")
            return ("SPARQL", "rule:parent_chain")

    # 5. Direct property match
    if analysis.target_properties & all_actual:
        if debug:
            print(f"  [DEBUG] Rule matched: direct_property "
                  f"(matched={sorted(analysis.target_properties & all_actual)})")
        return ("SPARQL", "rule:direct_property")

    if debug:
        print("\n  [DEBUG] SPARQL Gap Analysis")
        print(f"    Actual properties: {sorted(all_actual)}")
        print(f"    Target properties: {sorted(analysis.target_properties)}")
        print("    Route: None (inconclusive -> LLM decides)")

    return None


# ---------------------------------------------------------------------------
# Comparison / coverage helpers
# ---------------------------------------------------------------------------

def _is_comparison_question(question: str, entities: list[Entity]) -> bool:
    """Check if this is a comparison question with 2+ entities."""
    q_lower = question.lower()
    comparison_words = [
        "earlier", "later", "older", "younger", "first", "before", "after",
        "more recently", "longer", "shorter", "same", "both",
    ]
    return len(entities) >= 2 and any(w in q_lower for w in comparison_words)


def _has_incomplete_entity_coverage(
    question: str, entities: list[Entity], schema_ctx: str,
) -> bool:
    """Check if a comparison question mentions entities not in the schema.

    Detects 'X or Y' / 'X and Y' patterns and checks if all mentioned names
    have corresponding Entity blocks in the schema context.
    """
    q_lower = question.lower()
    comparison_words = [
        "earlier", "later", "older", "younger", "first", "before", "after",
        "more recently", "longer", "shorter", "same", "both",
    ]
    if not any(w in q_lower for w in comparison_words):
        return False

    m = re.search(r',\s*(.+?)\s+or\s+(.+?)\s*\?', question)
    if not m:
        m = re.search(r',\s*(.+?)\s+and\s+(.+?)\s*\?', question)
    if not m:
        m = re.search(r'\bboth\s+(.+?)\s+and\s+(.+?)[\s?]', question)
    if not m:
        return False

    candidates = [m.group(1).strip(), m.group(2).strip()]
    linked_labels = {e.label.lower() for e in entities}

    for name in candidates:
        if name.lower() not in linked_labels:
            return True
    return False


# ---------------------------------------------------------------------------
# Entity conversion helpers
# ---------------------------------------------------------------------------

def entities_to_schema_dicts(entities: list[Entity]) -> list[dict]:
    """Convert Spotlight Entity objects to dicts for get_schema_for_uris()."""
    return [
        {"uri": e.uri, "name": e.label}
        for e in entities if e.uri
    ]


def _sparql_results_to_entities(results: list[dict],
                                seed_uris: set[str]) -> list[Entity]:
    """Convert SPARQL context query results to Entity objects."""
    entities = []
    seen = set()
    for row in results:
        uri = None
        label = None
        for key in ("entity", "related", "neighbor", "uri", "s", "o"):
            if key in row and row[key]:
                val = row[key]
                if val.startswith("http://dbpedia.org/resource/"):
                    uri = val
                elif "/" not in val and val.strip():
                    local_name = val.replace(" ", "_")
                    uri = f"http://dbpedia.org/resource/{local_name}"
                if uri:
                    break
        for key in ("label", "entityLabel", "name"):
            if key in row and row[key]:
                label = row[key]
                break

        if not uri or uri in seen:
            continue
        seen.add(uri)

        if not label:
            label = unquote(uri.split("/")[-1]).replace("_", " ")

        relation = None
        for key in ("p", "relation", "predicate", "property"):
            if key in row and row[key]:
                val = row[key]
                relation = val.split("/")[-1] if "/" in val else val
                break

        source = "seed" if uri in seed_uris else "sparql_context"
        entities.append(Entity(
            uri=uri, label=label, source=source,
            discovery_relation=relation,
        ))

    return entities


# ---------------------------------------------------------------------------
# SPARQL ASK verification
# ---------------------------------------------------------------------------

def verify_sparql_feasibility(
    analysis: QuestionAnalysis,
    projections: list[SchemaProjection],
    entities: list[Entity],
    kg_client,
    rule_result: tuple[str, str] | None,
) -> tuple[str, str] | None:
    """Verify route via SPARQL ASK queries.

    Uses SchemaProjection completeness where possible, falling back to ASK
    only for bridge hops that need ground-truth verification.
    """
    if not rule_result:
        return None

    route, reason = rule_result

    # Direct rules: already verified by actual_property_set
    if "direct_property" in reason or "direct_date_comparison" in reason:
        return (route, reason)
    if "direct_country_comparison" in reason:
        return (route, reason)

    # Bridge routes: use projection completeness + ASK verification
    if "bridge" in reason or "hop2" in reason or "parent" in reason:
        # Check projection completeness first
        for proj in projections:
            if proj.completeness == "complete":
                return ("SPARQL", f"{reason}+ask_verified")

        # Completeness not "complete" -- verify via ASK
        hop1_prop = analysis.bridge_property
        if not hop1_prop:
            # Fallback to BRIDGE_WORD_TO_PROP
            q_lower = analysis.target_keywords[0] if analysis.target_keywords else ""
            hop1_prop = BRIDGE_WORD_TO_PROP.get(q_lower)

        if not hop1_prop or not entities:
            return (route, reason)

        verified = False
        for entity in entities:
            if not entity.uri:
                continue
            from dbpedia_qa.dbpedia.sparql_utils import escape_sparql_uri
            safe_uri = escape_sparql_uri(entity.uri)
            ask_query = (
                f"ASK {{ <{safe_uri}> <http://dbpedia.org/ontology/{hop1_prop}> ?x }}"
            )
            try:
                raw = kg_client.query(ask_query)
                if raw.get("boolean", False):
                    verified = True
                    break
            except Exception as e:
                logger.debug("ASK query failed for %s: %s", entity.label, e)
                continue

        if verified:
            return ("SPARQL", f"{reason}+ask_verified")
        return ("DECOMPOSE", f"{reason}+ask_missing")

    return (route, reason)


# ---------------------------------------------------------------------------
# Main route_question entry point
# ---------------------------------------------------------------------------

def route_question(
    question: str,
    entities: list[Entity],
    kg_client,
    llm_client,
    model: str,
    schema_disc=None,
    projections: list[SchemaProjection] | None = None,
    schema_ctx: str = "",
    debug: bool = False,
) -> tuple[str, str]:
    """Determine route for a question: SPARQL, DECOMPOSE, or TEXT.

    If projections not provided, builds them from entities + schema_disc.
    schema_ctx is optional backward-compat for incomplete entity coverage check.
    """
    from entity_profile import build_entity_profile
    from question_analysis import analyze_question
    from schema_projection import project
    from sparql_evaluation import call_llm

    # Build projections if not provided
    if projections is None and schema_disc is not None:
        profiles = [build_entity_profile(e.uri, e.label, schema_disc) for e in entities if e.uri]
        analysis = analyze_question(question, profiles)
        projections = [project(p, analysis) for p in profiles]
    elif projections is not None:
        analysis = analyze_question(question)
    else:
        return ("TEXT", "no_schema_fallback")

    if not projections:
        return ("TEXT", "no_schema_fallback")

    # Guard: comparison questions with incomplete entity coverage → DECOMPOSE
    if schema_ctx and _has_incomplete_entity_coverage(question, entities, schema_ctx):
        if debug:
            print("  [DEBUG] Incomplete entity coverage → DECOMPOSE")
        return ("DECOMPOSE", "incomplete_coverage")

    # Single call to _rule_based_route()
    rule_result = _rule_based_route(analysis, projections, debug=debug)

    # Rule-based + ASK verification
    ask_result = verify_sparql_feasibility(
        analysis, projections, entities, kg_client, rule_result,
    )
    if ask_result:
        route, reason = ask_result
        if debug:
            print(f"  [DEBUG] ASK verification: {route} ({reason})")
        if route in ("SPARQL", "DECOMPOSE"):
            return (route, reason)

    # LLM fallback with render_for_routing() (~150 tokens)
    routing_context = "\n\n".join(proj.render_for_routing() for proj in projections)
    hint = ""
    if rule_result:
        hint = f"\nSchema analysis suggests: {rule_result[1]} (SPARQL likely feasible)\n"

    safe_schema = routing_context.replace("{", "{{").replace("}", "}}")
    safe_question = question.replace("{", "{{").replace("}", "}}")
    route_prompt = ROUTE_PROMPT.format(
        schema=safe_schema, question=safe_question, hint=hint,
    )
    route_resp = call_llm(llm_client, route_prompt, model).strip()
    route_match = re.search(r'\b(SPARQL|TEXT)\b', route_resp, re.IGNORECASE)
    route = route_match.group(1).upper() if route_match else "TEXT"

    # Preserve rule route_source when LLM agrees
    if rule_result and route == "SPARQL":
        route_source = rule_result[1]
    else:
        route_source = "llm"

    if debug:
        print(f"  [DEBUG] LLM route: {route} ({route_source})")

    return (route, route_source)
