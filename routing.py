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

from dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS, PROPERTY_RANGES
from models import Entity
from question_analysis import COMPARISON_WORDS

if TYPE_CHECKING:
    from question_analysis import QuestionAnalysis
    from schema_projection import SchemaProjection

logger = logging.getLogger(__name__)

_last_route_meta: dict = {}

PROPERTY_EXTRACTION_PROMPT = """Given this question and DBpedia schema, \
which properties are needed to answer it?

{schema}

QUESTION: {question}

Output a comma-separated list of property names (no dbo:/dbp: prefix).
Only output properties listed above. For 2-hop paths (e.g., "director's birthdate"),
list both: director, birthDate
If no properties can help, output: NONE

Properties:"""


# Property sets for comparison routing
_DATE_PROPERTIES = {
    "dbo:birthDate", "dbp:birthDate", "dbo:deathDate", "dbp:deathDate",
    "dbo:activeYearsStartYear", "dbp:established", "dbo:foundingYear",
    "dbp:foundingDate", "dbo:releaseDate", "dbp:released",
    "dbo:publicationDate", "dbp:publicationDate", "dbp:published",
}
_COUNTRY_PROPERTIES = {
    "dbo:nationality", "dbp:nationality", "dbo:country", "dbp:country",
    "dbo:citizenship", "dbp:citizenship", "dbo:birthPlace", "dbp:birthPlace",
    "dbo:stateOfOrigin", "dbp:placeOfBirth",
}
_PARENT_PROPS = {"dbo:parent", "dbo:father", "dbo:mother", "dbp:father", "dbp:mother"}


# ---------------------------------------------------------------------------
# Rule-based routing (structured — no regex parsing)
# ---------------------------------------------------------------------------

def _rule_based_route(
    analysis: QuestionAnalysis,
    projections: list[SchemaProjection],
    debug: bool = False,
    extra_ranges: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """Return ('SPARQL', reason) if rules match, None if inconclusive."""
    # Merge static + dynamic property ranges for bridge detection
    effective_ranges = dict(PROPERTY_RANGES)
    if extra_ranges:
        effective_ranges.update(extra_ranges)

    # Collect actual properties across all entity projections
    all_actual: set[str] = set()
    for proj in projections:
        all_actual |= proj.entity_profile.actual_property_set

    # 1. Comparison + date/country properties -> SPARQL
    # Require ALL entities to have the property (intersection), not just ANY (union).
    # If only one entity has dbo:birthDate, SPARQL returns partial data and the
    # LLM has no basis for comparison — fall through to DECOMPOSE instead.
    if analysis.is_comparison:
        per_entity_props = [
            proj.entity_profile.actual_property_set for proj in projections
        ]
        # Intersection: only properties present on every linked entity
        # Use reduce-style intersection to support both set and frozenset inputs.
        common_props: set[str] = (
            set(per_entity_props[0]).intersection(*per_entity_props[1:])
            if per_entity_props else set()
        )

        if common_props & _DATE_PROPERTIES:
            if debug:
                print(f"  [DEBUG] Rule matched: direct_date_comparison "
                      f"(have {sorted(common_props & _DATE_PROPERTIES)})")
            return ("SPARQL", "rule:direct_date_comparison")

        country_keywords = {"country", "nationality", "citizen"}
        if (analysis.target_keywords_set & country_keywords) and (
            common_props & _COUNTRY_PROPERTIES
        ):
            if debug:
                print(f"  [DEBUG] Rule matched: direct_country_comparison "
                      f"(have {sorted(common_props & _COUNTRY_PROPERTIES)})")
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
            # Fallback: check property ranges directly (static + dynamic)
            if analysis.bridge_property:
                range_key = f"dbo:{analysis.bridge_property}"
                if range_key in effective_ranges:
                    target_cls = effective_ranges[range_key]
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
            # Also check via property ranges (static + dynamic)
            if bridge_full in effective_ranges:
                target_cls = effective_ranges[bridge_full]
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
    # For comparison questions, require the matched property on ALL entities
    # (intersection), not just any — partial data yields wrong comparisons.
    # `common_props` is already computed in step 1 when is_comparison is True.
    if analysis.is_comparison:
        matched = analysis.target_properties & common_props
    else:
        matched = analysis.target_properties & all_actual
    if matched:
        if debug:
            print(f"  [DEBUG] Rule matched: direct_property "
                  f"(matched={sorted(matched)})")
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
    return len(entities) >= 2 and any(w in q_lower for w in COMPARISON_WORDS)


def _has_incomplete_entity_coverage(
    question: str, entities: list[Entity],
) -> bool:
    """Check if a comparison question mentions entities not in linked set.

    Detects 'X or Y' / 'X and Y' patterns and checks if all mentioned names
    have corresponding Entity objects.
    """
    q_lower = question.lower()
    if not any(w in q_lower for w in COMPARISON_WORDS):
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
        # Scan ALL row values for DBpedia resource URIs (LLMs use arbitrary variable names)
        for val in row.values():
            if isinstance(val, str) and val.startswith("http://dbpedia.org/resource/"):
                uri = val
                break
        # Fallback: check for local names (non-URI strings without slashes)
        if not uri:
            for val in row.values():
                if isinstance(val, str) and "/" not in val and val.strip():
                    local_name = val.replace(" ", "_")
                    uri = f"http://dbpedia.org/resource/{local_name}"
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

    Direct property routes are already verified by actual_property_set.
    Bridge/hop2/parent routes always run ASK for ground-truth verification.
    """
    if not rule_result:
        return None

    route, reason = rule_result

    # Direct rules: already verified by actual_property_set
    if "direct_property" in reason or "direct_date_comparison" in reason:
        return (route, reason)
    if "direct_country_comparison" in reason:
        return (route, reason)

    # Bridge routes: always verify via ASK query.
    # Don't shortcut on projection completeness — dbp-only properties
    # (e.g., dbp:father without dbo:father) can inflate completeness.
    if "bridge" in reason or "hop2" in reason or "parent" in reason:
        hop1_prop = analysis.bridge_property

        # Safety net: reject property names that aren't valid identifiers
        # (prevents natural language from LLM leaking into SPARQL URIs)
        if hop1_prop and not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', hop1_prop):
            logger.debug("Invalid bridge property %r, skipping ASK", hop1_prop)
            return (route, reason)

        if not hop1_prop or not entities:
            return (route, reason)

        # Determine hop-2 property for 2-hop ASK verification.
        # Scan ALL dbo: candidates — don't break on the first one if it fails
        # validation, otherwise we silently degrade to 1-hop verification.
        hop2_dbo_prop = None
        if analysis.hop2_properties:
            for p in analysis.hop2_properties:
                if not p.startswith("dbo:"):
                    continue
                candidate = p[4:].strip()
                if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', candidate):
                    hop2_dbo_prop = candidate
                    break

        from dbpedia.sparql_utils import escape_sparql_uri

        def _ask_entity(entity: Entity) -> bool:
            """Run ASK query for a single entity. Returns True if chain exists."""
            safe_uri = escape_sparql_uri(entity.uri)
            hop1_pattern = (
                f"<{safe_uri}> <http://dbpedia.org/ontology/{hop1_prop}> ?x"
            )
            if hop2_dbo_prop:
                ask_query = (
                    f"ASK {{ {hop1_pattern} . "
                    f"?x <http://dbpedia.org/ontology/{hop2_dbo_prop}> ?y }}"
                )
            else:
                ask_query = f"ASK {{ {hop1_pattern} }}"
            try:
                raw = kg_client.query(ask_query)
                return raw.get("boolean", False)
            except Exception as e:
                logger.debug("ASK query failed for %s: %s", entity.label, e)
                # Exception counts as verification failure
                return False

        uri_entities = [e for e in entities if e.uri]
        if not uri_entities:
            return (route, reason)

        # Comparison questions need ALL entities verified (partial data = wrong comparison).
        # Non-comparison questions need ANY entity verified (one source suffices).
        if analysis.is_comparison:
            all_verified = all(_ask_entity(e) for e in uri_entities)
            if all_verified:
                return ("SPARQL", f"{reason}+ask_verified")
            return ("DECOMPOSE", f"{reason}+ask_partial")
        else:
            any_verified = any(_ask_entity(e) for e in uri_entities)
            if any_verified:
                return ("SPARQL", f"{reason}+ask_verified")
            return ("DECOMPOSE", f"{reason}+ask_missing")

    return (route, reason)


# ---------------------------------------------------------------------------
# LLM property extraction (replaces binary LLM routing)
# ---------------------------------------------------------------------------

def _build_property_schema(projections: list) -> str:
    """Build available-property summary for LLM property extraction (~200 tokens).

    Shows actual properties (from SPARQL instance query) + ontology schema (capped)
    + 2-hop chains (already instance-filtered in entity_profile).
    """
    lines = []
    for proj in projections:
        profile = proj.entity_profile
        lines.append(f"Entity: {profile.label}")
        if proj.primary_class:
            lines.append(f"Class: {proj.primary_class}")
            schema = CACHED_CLASS_SCHEMAS.get(proj.primary_class, {})
            if schema.get("dbo"):
                lines.append(f"  dbo properties: {', '.join(schema['dbo'][:15])}")
        actual_dbo = profile.actual_properties.get("dbo", [])
        actual_dbp = profile.actual_properties.get("dbp", [])
        if actual_dbo:
            lines.append(f"  Actual dbo: {', '.join(actual_dbo)}")
        dbp_only = [p for p in actual_dbp if p not in (actual_dbo or [])]
        if dbp_only:
            lines.append(f"  Actual dbp: {', '.join(dbp_only)}")
        if profile.all_2hop_chains:
            for prop, (target, tprops) in profile.all_2hop_chains.items():
                lines.append(f"  Chain: {prop} -> {target}: {', '.join(tprops[:6])}")
    return "\n".join(lines)


def _extract_properties_llm(
    question: str,
    property_schema: str,
    llm_client,
    model: str,
    debug: bool = False,
) -> list[str]:
    """Use LLM to extract needed properties from question + schema.

    Called when keyword matching is inconclusive. The LLM sees available
    schema properties and picks which ones the question needs.

    Returns property names without prefix, e.g. ["director", "birthDate"].
    """
    from llm_provider import call_llm

    safe_schema = property_schema.replace("{", "{{").replace("}", "}}")
    safe_question = question.replace("{", "{{").replace("}", "}}")
    prompt = PROPERTY_EXTRACTION_PROMPT.format(
        schema=safe_schema, question=safe_question,
    )

    response = call_llm(llm_client, prompt, model).strip()
    _last_route_meta["llm_property_prompt"] = prompt
    _last_route_meta["llm_property_response"] = response

    if debug:
        print(f"  [DEBUG] LLM property extraction: {response}")

    if not response or response.upper() == "NONE":
        return []

    # Parse comma-separated property names, strip prefixes
    properties = []
    for part in response.split(","):
        prop = part.strip().split(":")[-1].strip()  # strip dbo:/dbp: prefix + whitespace
        if prop and prop.upper() != "NONE":
            # Validate: property names are camelCase identifiers (no spaces, no natural language)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', prop):
                properties.append(prop)
            else:
                logger.debug("Rejected invalid LLM property: %r", prop)
    _last_route_meta["llm_extracted_properties"] = properties

    return properties


# ---------------------------------------------------------------------------
# Multi-hop guard for direct_property routes
# ---------------------------------------------------------------------------

def _is_likely_multihop(question: str, entities: list[Entity], reason: str) -> bool:
    """Detect direct_property SPARQL routes that should be DECOMPOSE instead.

    All MuSiQue questions are multi-hop (3-4 hops), but some get routed to
    single-hop SPARQL via direct_property match. 3+ linked entities strongly
    indicates a multi-hop chain that direct SPARQL cannot handle.

    Only fires on direct_property routes — bridge/comparison routes are
    already handled correctly by the routing pipeline.
    """
    if "direct_property" not in reason:
        return False
    # 3+ entities means the question references multiple things in a chain
    return len(entities) >= 3


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

    Three-stage routing:
      1. Keyword matching → deterministic rules → ASK verification
      2. LLM property extraction → re-run rules → ASK verification
      3. TEXT fallback

    The LLM is used for open-vocabulary property extraction (mapping
    question words like "helmed" to schema properties like "director"),
    NOT for binary SPARQL/TEXT classification. All routing decisions
    are deterministic once properties are identified.
    """
    global _last_route_meta
    _last_route_meta = {}
    from entity_profile import build_entity_profile
    from question_analysis import analyze_question, enrich_analysis
    from schema_projection import project

    # Guard: if no entities have URIs, SPARQL is impossible
    if not any(e.uri for e in entities):
        return ("TEXT", "no_entity_uris")

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

    # Collect dynamic ranges from all entity profiles (uncached type discovery)
    dyn_ranges: dict[str, str] = {}
    for proj in projections:
        dyn_ranges.update(getattr(proj.entity_profile, "dynamic_ranges", {}))

    # Guard: comparison questions with incomplete entity coverage → DECOMPOSE
    if _has_incomplete_entity_coverage(question, entities):
        if debug:
            print("  [DEBUG] Incomplete entity coverage → DECOMPOSE")
        return ("DECOMPOSE", "incomplete_coverage")

    # Stage 1: Keyword-based rules + ASK verification
    rule_result = _rule_based_route(
        analysis, projections, debug=debug, extra_ranges=dyn_ranges,
    )
    _last_route_meta["rule_result"] = rule_result
    ask_result = verify_sparql_feasibility(
        analysis, projections, entities, kg_client, rule_result,
    )
    if ask_result:
        route, reason = ask_result
        _last_route_meta["ask_result"] = reason
        if debug:
            print(f"  [DEBUG] Stage 1 (keywords): {route} ({reason})")
        if route == "SPARQL" and _is_likely_multihop(question, entities, reason):
            if debug:
                print(f"  [DEBUG] Multi-hop guard: SPARQL→DECOMPOSE ({reason})")
            return ("DECOMPOSE", f"multihop_guard:{reason}")
        if route in ("SPARQL", "DECOMPOSE"):
            return (route, reason)

    # Stage 2: LLM property extraction → re-run deterministic routing
    try:
        property_schema = _build_property_schema(projections)
        llm_properties = _extract_properties_llm(
            question, property_schema, llm_client, model, debug=debug,
        )

        if llm_properties:
            enriched = enrich_analysis(analysis, llm_properties, extra_ranges=dyn_ranges)
            enriched_profiles = [proj.entity_profile for proj in projections]
            enriched_projections = [project(p, enriched) for p in enriched_profiles]
            enriched_rule = _rule_based_route(
                enriched, enriched_projections, debug=debug, extra_ranges=dyn_ranges,
            )
            enriched_ask = verify_sparql_feasibility(
                enriched, enriched_projections, entities, kg_client, enriched_rule,
            )
            _last_route_meta["enriched_rule_result"] = enriched_ask
            if enriched_ask:
                route, reason = enriched_ask
                if debug:
                    print(f"  [DEBUG] Stage 2 (LLM props={llm_properties}): "
                          f"{route} ({reason})")
                if route == "SPARQL" and _is_likely_multihop(
                    question, entities, reason
                ):
                    if debug:
                        print(f"  [DEBUG] Multi-hop guard: SPARQL→DECOMPOSE "
                              f"(llm_prop:{reason})")
                    return ("DECOMPOSE", f"multihop_guard:llm_prop:{reason}")
                if route in ("SPARQL", "DECOMPOSE"):
                    return (route, f"llm_prop:{reason}")
    except Exception as e:
        logger.warning("Stage 2 LLM property extraction failed: %s", e)

    # Stage 3: TEXT fallback
    if debug:
        print("  [DEBUG] No property match → TEXT")
    return ("TEXT", "llm_prop:no_match")
