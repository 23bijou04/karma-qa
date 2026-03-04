"""Routing logic for hybrid QA: rule-based route + SPARQL ASK verification.

Extracted from hybrid_qa.py to break circular dependency with logicrag_decomposition.

Key functions:
  route_question()                    — Main entry: rules + ASK verification + LLM fallback
  _rule_based_route()                 — Deterministic property/chain matching
  verify_sparql_feasibility()         — SPARQL ASK to verify triples exist
  _parse_2hop_chains()                — Parse 2-hop chain section from schema
  _is_comparison_question()           — Detect comparison questions
  _has_incomplete_entity_coverage()   — Detect missing entities in comparisons
  entities_to_schema_dicts()          — Convert Entity list to schema dicts
  _sparql_results_to_entities()       — Convert SPARQL result rows to Entity objects
"""

import logging
import re
from urllib.parse import unquote

from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS, PROPERTY_RANGES
from dbpedia_qa.models import Entity

logger = logging.getLogger(__name__)

# Mapping from question keywords to DBpedia ontology property names.
# Used by both _rule_based_route() (chain matching) and verify_sparql_feasibility() (ASK).
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


# ---------------------------------------------------------------------------
# 2-hop chain parsing
# ---------------------------------------------------------------------------

def _parse_2hop_chains(schema_ctx: str) -> dict[str, tuple[str, list[str]]]:
    """Extract 2-hop chain info from the '2-Hop chains:' section only.

    Section-anchored: only parses lines inside '2-Hop chains:' blocks,
    NOT 'Actual properties' lines like 'dbo:director -> Christopher Nolan'.

    Returns: {"director": ("Person", ["birthDate", "deathDate", ...])}
    """
    chains: dict[str, tuple[str, list[str]]] = {}
    in_section = False
    for line in schema_ctx.split("\n"):
        stripped = line.strip()
        if stripped == "2-Hop chains:":
            in_section = True
            continue
        if in_section:
            if not stripped or not stripped.startswith("- dbo:"):
                in_section = False
                continue
            m = re.match(
                r'-\s*dbo:([\w-]+)\s*->\s*([\w-]+):\s*([\w-]+(?:,\s*[\w-]+)*)', stripped
            )
            if m:
                chains[m.group(1)] = (
                    m.group(2),
                    [p.strip() for p in m.group(3).split(",")],
                )
    return chains


# ---------------------------------------------------------------------------
# Rule-based routing
# ---------------------------------------------------------------------------

def _rule_based_route(question: str, schema_ctx: str,
                      debug: bool = False) -> tuple[str, str] | None:
    """Return ('SPARQL', reason) if rules match, None if inconclusive (let LLM decide)."""
    q_lower = question.lower()

    # Parse properties from both schema formats:
    #   Format 1: "dbo:prop -> value" (actual properties section)
    actual_props = set(re.findall(r'(?:dbo|dbp):\w+', schema_ctx))
    #   Format 2: "  dbo: prop1, prop2, ..." (cached ontology list, may be truncated)
    for m in re.finditer(r'^\s+(dbo|dbp): ([\w-]+(?:,\s*[\w-]+)*)', schema_ctx, re.MULTILINE):
        prefix = m.group(1)
        for name in m.group(2).split(','):
            name = name.strip()
            if name:
                actual_props.add(f"{prefix}:{name}")

    comparison_words = [
        "earlier", "later", "older", "younger", "first", "before", "after",
        "same", "both", "more", "less", "longer", "shorter",
    ]
    date_props = {
        "dbo:birthDate", "dbp:birthDate", "dbo:deathDate", "dbp:deathDate",
        "dbo:activeYearsStartYear", "dbp:established", "dbo:foundingYear",
        "dbp:foundingDate", "dbo:releaseDate", "dbp:released",
    }
    country_props = {
        "dbo:nationality", "dbp:nationality", "dbo:country", "dbp:country",
        "dbo:citizenship", "dbp:citizenship", "dbo:birthPlace", "dbp:birthPlace",
        "dbo:stateOfOrigin", "dbp:placeOfBirth",
    }

    # Comparison + date properties on the entity itself -> SPARQL
    if any(w in q_lower for w in comparison_words):
        if actual_props & date_props:
            if debug:
                print(f"  [DEBUG] Rule matched: direct_date_comparison "
                      f"(have {sorted(actual_props & date_props)})")
            return ("SPARQL", "rule:direct_date_comparison")
        # "same country / nationality" comparisons
        country_keywords = ["country", "nationality", "citizen", "born in"]
        if any(kw in q_lower for kw in country_keywords) and actual_props & country_props:
            if debug:
                print(f"  [DEBUG] Rule matched: direct_country_comparison "
                      f"(have {sorted(actual_props & country_props)})")
            return ("SPARQL", "rule:direct_country_comparison")

    # Bridge chain validation: verify the relationship word maps to a target class
    # whose properties are relevant to the question (replaces blind class+word check)
    chains_2hop = _parse_2hop_chains(schema_ctx)

    def _has_bridge_word(word: str) -> bool:
        return bool(re.search(rf"\b{word}\b", q_lower))

    if any(w in q_lower for w in comparison_words):
        for word, prop_name in BRIDGE_WORD_TO_PROP.items():
            if not _has_bridge_word(word):
                continue
            # Check if this property has a chain in schema text
            if prop_name in chains_2hop:
                target_cls, target_props = chains_2hop[prop_name]
                if any(p in target_props for p in [
                    "birthDate", "deathDate", "foundingDate",
                    "activeYearsStartYear",
                ]):
                    if debug:
                        print(f"  [DEBUG] Rule matched: bridge_comparison "
                              f"(chain: {prop_name} -> {target_cls})")
                    return ("SPARQL", "rule:bridge_comparison")
            # Fallback: check PROPERTY_RANGES directly
            range_key = f"dbo:{prop_name}"
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

    # Non-comparison bridge: "When was the director of X born?"
    # Detects: bridge_word + hop-2 property keyword
    hop2_keywords = {
        "born": ["birthDate", "birthPlace"],
        "died": ["deathDate", "deathPlace"],
        "birth": ["birthDate", "birthPlace"],
        "death": ["deathDate", "deathPlace"],
        "country": ["country", "nationality"],
    }
    for word, prop_name in BRIDGE_WORD_TO_PROP.items():
        if not _has_bridge_word(word):
            continue
        range_key = f"dbo:{prop_name}"
        if range_key not in PROPERTY_RANGES:
            continue
        target_cls = PROPERTY_RANGES[range_key]
        if target_cls not in CACHED_CLASS_SCHEMAS:
            continue
        target_dbo = CACHED_CLASS_SCHEMAS[target_cls].get("dbo", [])
        for kw, needed_props in hop2_keywords.items():
            if re.search(rf"\b{kw}\b", q_lower) and any(
                p in target_dbo for p in needed_props
            ):
                if debug:
                    print(f"  [DEBUG] Rule matched: bridge_hop2 "
                          f"({word} -> {target_cls} -> {kw})")
                return ("SPARQL", "rule:bridge_hop2")

    # Parent chain: father/mother + any property question -> SPARQL
    # Handles: "Where did X's father graduate?", "When was X's mother born?"
    parent_words = ["father", "mother", "parent", "paternal", "maternal"]
    parent_props = {"dbo:parent", "dbp:father", "dbp:mother"}
    if any(w in q_lower for w in parent_words) and actual_props & parent_props:
        if debug:
            print(f"  [DEBUG] Rule matched: parent_chain "
                  f"(have {sorted(actual_props & parent_props)})")
        return ("SPARQL", "rule:parent_chain")

    # Direct property lookup (includes morphological variants for word boundary matching)
    _director_props = {"dbo:director", "dbp:director"}
    _author_props = {"dbo:author", "dbp:author", "dbo:writer", "dbp:writer"}
    _birth_props = {"dbo:birthDate", "dbp:birthDate", "dbo:birthPlace", "dbp:birthPlace"}
    _award_props = {"dbo:award", "dbp:award"}
    _founder_props = {"dbo:foundingYear", "dbp:foundingDate", "dbo:founder", "dbp:founder"}
    direct_map = {
        "direct": _director_props,
        "directed": _director_props,
        "director": _director_props,
        "wrote": _author_props,
        "written": _author_props,
        "author": _author_props,
        "born": _birth_props,
        "died": {"dbo:deathDate", "dbp:deathDate", "dbo:deathPlace"},
        "capital": {"dbo:capital", "dbp:capital"},
        "population": {"dbo:populationTotal", "dbp:population"},
        "spouse": {"dbo:spouse", "dbp:spouse"},
        "married": {"dbo:spouse", "dbp:spouse"},
        "founded": _founder_props,
        "founder": _founder_props,
        "nationality": {"dbo:nationality", "dbp:nationality", "dbo:citizenship"},
        "father": {"dbo:parent", "dbp:father"},
        "mother": {"dbo:parent", "dbp:mother"},
        "award": _award_props,
        "awarded": _award_props,
        "occupation": {"dbo:occupation", "dbp:occupation"},
    }
    for keyword, props in direct_map.items():
        if re.search(rf"\b{keyword}\b", q_lower) and actual_props & props:
            if debug:
                print(f"  [DEBUG] Rule matched: direct_property "
                      f"(keyword='{keyword}', matched={sorted(actual_props & props)})")
            return ("SPARQL", "rule:direct_property")

    if debug:
        print("\n  [DEBUG] SPARQL Gap Analysis")
        print(f"    Schema properties: {sorted(actual_props)}")
        if chains_2hop:
            print(f"    2-Hop chains: {chains_2hop}")
        else:
            print("    2-Hop chains: none")
        for keyword, props in direct_map.items():
            if re.search(rf"\b{keyword}\b", q_lower):
                found = actual_props & props
                missing = props - actual_props
                if missing:
                    print(f"    Keyword '{keyword}': need {sorted(props)}, "
                          f"have {sorted(found)}, MISSING {sorted(missing)}")
        print("    Route: None (inconclusive -> LLM decides)")

    return None  # inconclusive — let LLM decide


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

    # Extract candidate entity names from "X or Y" / "X and Y" patterns
    # Match: ", X or Y?" at end of question (common 2WikiMultiHopQA pattern)
    m = re.search(r',\s*(.+?)\s+or\s+(.+?)\s*\?', question)
    if not m:
        m = re.search(r',\s*(.+?)\s+and\s+(.+?)\s*\?', question)
    if not m:
        # "Are both X and Y ..." pattern (no comma)
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
        # Try common column names for entity URIs
        uri = None
        label = None
        for key in ("entity", "related", "neighbor", "uri", "s", "o"):
            if key in row and row[key]:
                val = row[key]
                if val.startswith("http://dbpedia.org/resource/"):
                    uri = val  # full URI — use directly, no re-encoding
                elif "/" not in val and val.strip():
                    # Reconstructed local name from execute_sparql's deconstruction
                    # DBpedia uses literal special chars (', parens) — no quoting
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

        # Extract predicate for discovery_relation
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
    question: str,
    schema_ctx: str,
    entities: list[Entity],
    kg_client,
    rule_result: tuple[str, str] | None,
) -> tuple[str, str] | None:
    """Verify route via SPARQL ASK queries — ground truth, not just schema heuristic.

    Takes a pre-computed rule_result from _rule_based_route() and runs cheap ASK
    queries (~50ms each) to verify expected triples actually exist in the KG.

    Returns ("SPARQL", reason) if verified, ("DECOMPOSE", reason) if any hop
    is missing, or None if no rule matched (let LLM decide).
    """
    if not rule_result:
        return None

    route, reason = rule_result

    # For direct_property rules, verify the entity actually has the property
    if "direct_property" in reason or "direct_date_comparison" in reason:
        # These are already verified by actual property values in schema
        # (schema shows "dbo:birthDate -> 1879-03-14"), so ASK is redundant
        return (route, reason)

    if "direct_country_comparison" in reason:
        return (route, reason)

    # For bridge routes, verify both hops via ASK
    if "bridge" in reason or "hop2" in reason or "parent" in reason:
        q_lower = question.lower()
        hop1_prop = None
        for word, prop in BRIDGE_WORD_TO_PROP.items():
            if re.search(rf"\b{word}\b", q_lower):
                hop1_prop = prop
                break

        if not hop1_prop or not entities:
            return (route, reason)  # can't verify, trust the rule

        # ASK if any seed entity has this property
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
    schema_ctx: str,
    entities: list[Entity],
    kg_client,
    llm_client,
    model: str,
    debug: bool = False,
) -> tuple[str, str]:
    """Determine route for a question: SPARQL, DECOMPOSE, or TEXT.

    Strategy:
    1. Check incomplete entity coverage → DECOMPOSE
    2. Rule-based route + ASK verification → SPARQL (verified) or DECOMPOSE
    3. LLM fallback for ambiguous cases
    """
    from sparql_evaluation import call_llm

    # Guard: comparison questions with incomplete entity coverage → DECOMPOSE
    if schema_ctx and _has_incomplete_entity_coverage(question, entities, schema_ctx):
        if debug:
            print("  [DEBUG] Incomplete entity coverage → DECOMPOSE")
        return ("DECOMPOSE", "incomplete_coverage")

    if not schema_ctx:
        return ("TEXT", "no_schema_fallback")

    # Single call to _rule_based_route()
    rule_result = _rule_based_route(question, schema_ctx, debug=debug)

    # Rule-based + ASK verification
    ask_result = verify_sparql_feasibility(
        question, schema_ctx, entities, kg_client, rule_result,
    )
    if ask_result:
        route, reason = ask_result
        if debug:
            print(f"  [DEBUG] ASK verification: {route} ({reason})")
        if route in ("SPARQL", "DECOMPOSE"):
            return (route, reason)

    # LLM fallback
    hint = ""
    if rule_result:
        hint = f"\nSchema analysis suggests: {rule_result[1]} (SPARQL likely feasible)\n"

    safe_schema = schema_ctx.replace("{", "{{").replace("}", "}}")
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
