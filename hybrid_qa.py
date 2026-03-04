#!/usr/bin/env python3
"""Schema-Routed Hybrid QA: uses KG schema to route SPARQL vs text.

Three-path architecture:
  1. EntityLink(q)            — DBpedia Spotlight (spaCy fallback)
  2. SchemaDiscovery(E)       — entity classes, properties, sample values
  3. RouteDecision(q, S)      — rule-based first, LLM fallback -> SPARQL | TEXT
  4. Direct SPARQL            — single-hop, generate + execute with retry
  5. LogicRAG decomposition   — all multi-hop (bridge, comparison, compositional)
  6. Text fallback            — neighbors + Wikipedia + LLM reasoning

Usage:
  python3 hybrid_qa.py "Who directed Inception?" --verbose
  python3 hybrid_qa.py "Who destroys the last Horcrux?" --verbose
  python3 hybrid_qa.py "When was Albert Einstein born?" --api-key $GROQ_API_KEY
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from groq import Groq

from ablation_study_v3.wiki_utils import uri_to_wiki_title
from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery
from dbpedia_qa.dbpedia.schema_discovery import CACHED_CLASS_SCHEMAS, PROPERTY_RANGES
from dbpedia_qa.models import Entity, HybridResult, extract_final_answer
from graphrag_wiki import (
    assemble_context,
    link_entities_to_dbpedia,
    retrieve_wikipedia,
)
from logicrag_decomposition import _try_logicrag_decomposition
from sparql_evaluation import (
    ANSWER_PROMPT,
    SPARQL_GEN_PROMPT,
    SPARQL_RETRY_PROMPT,
    call_llm,
    execute_sparql,
    extract_sparql,
    get_schema_for_uris,
    sanitize_sparql_uris,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_REASON_PROMPT = """Answer this question using the sources below.

Question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data — trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Show brief step-by-step reasoning
- End with exactly: FINAL ANSWER: <answer>
"""

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
# Route decision
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
    bridge_word_to_prop = {
        "director": "director", "directed": "director",
        "producer": "producer", "produced": "producer",
        "writer": "writer", "wrote": "writer",
        "author": "author",
        "composer": "musicComposer",
        "starring": "starring", "actor": "starring", "actress": "starring",
        "founded": "foundedBy", "founder": "foundedBy",
    }

    chains_2hop = _parse_2hop_chains(schema_ctx)

    def _has_bridge_word(word: str) -> bool:
        return bool(re.search(rf"\b{word}\b", q_lower))

    if any(w in q_lower for w in comparison_words):
        for word, prop_name in bridge_word_to_prop.items():
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
    for word, prop_name in bridge_word_to_prop.items():
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
# Helpers
# ---------------------------------------------------------------------------

def entities_to_schema_dicts(entities: list[Entity]) -> list[dict]:
    """Convert Spotlight Entity objects to dicts for get_schema_for_uris()."""
    return [
        {"uri": e.uri, "name": e.label}
        for e in entities if e.uri
    ]


def _is_comparison_question(question: str, entities: list[Entity]) -> bool:
    """Check if this is a comparison question with 2+ entities."""
    q_lower = question.lower()
    # Superset of comparison_words in _rule_based_route — intentionally broader
    # to catch comparison patterns (same/both/longer/shorter) that may not require
    # date-based routing but do benefit from per-entity decomposition.
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


def _sparql_neighbors(entities: list[Entity],
                      kg_client: DBpediaSPARQLClient,
                      limit_per_entity: int = 10,
                      verbose: bool = False) -> list[Entity]:
    """Get direct dbo:/dbp: neighbors via a simple SPARQL query.

    Much simpler than explore_kg: no schema lookup, no type blocklist.
    Just gets the first N structured neighbors for each seed entity.
    """
    from dbpedia_qa.dbpedia.sparql_utils import escape_sparql_uri

    all_entities = list(entities)
    seen_uris = {e.uri for e in entities if e.uri}

    for entity in entities:
        if not entity.uri:
            continue
        safe_uri = escape_sparql_uri(entity.uri)
        query = f"""
        SELECT DISTINCT ?p ?neighbor ?label WHERE {{
            <{safe_uri}> ?p ?neighbor .
            FILTER(isIRI(?neighbor))
            FILTER(
                STRSTARTS(STR(?p), "http://dbpedia.org/ontology/")
                || STRSTARTS(STR(?p), "http://dbpedia.org/property/")
            )
            FILTER(?p != <http://dbpedia.org/ontology/wikiPageWikiLink>)
            FILTER(STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/"))
            FILTER(!STRSTARTS(STR(?neighbor), "http://dbpedia.org/resource/Category:"))
            ?neighbor rdfs:label ?label .
            FILTER(LANG(?label) = "en")
        }} LIMIT {limit_per_entity}
        """
        try:
            raw = kg_client.query(query)
            bindings = raw.get("results", {}).get("bindings", [])
        except Exception as e:
            logger.warning(f"Neighbor query failed for {entity.label}: {e}")
            continue

        for b in bindings:
            uri = b["neighbor"]["value"]
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            label = b.get("label", {}).get("value", "")
            if not label:
                label = unquote(uri.split("/")[-1]).replace("_", " ")
            prop = b["p"]["value"].split("/")[-1]
            all_entities.append(Entity(
                uri=uri, label=label, source="sparql_neighbor",
                discovery_relation=prop,
            ))
            if verbose:
                print(f"    {entity.label} --{prop}--> {label}")

    return all_entities


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


def _format_kg_facts(results: list[dict], max_chars: int = 800) -> str:
    """Format SPARQL result rows as 'Known facts from KG' text block."""
    if not results:
        return ""
    lines = []
    header = "Known facts from knowledge graph:\n"
    total = len(header)
    for row in results:
        parts = []
        for key, val in row.items():
            if val and not key.startswith("_"):
                if isinstance(val, str) and val.startswith("http://"):
                    val = val.split("/")[-1].replace("_", " ")
                parts.append(f"{key}: {val}")
        if parts:
            line = "- " + ", ".join(parts)
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
    if not lines:
        return ""
    return header + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def hybrid_answer(question: str, api_key: str,
                  model: str = "llama-3.3-70b-versatile",
                  verbose: bool = False, debug: bool = False,
                  llm_client=None, linker=None,
                  kg_client=None, schema_disc=None) -> HybridResult:
    """Schema-routed hybrid QA with 3-path architecture.

    Paths:
      1. Direct SPARQL — single-hop, schema matches confidently
      2. LogicRAG decomposition — all multi-hop (bridge, comparison, compositional)
      3. Simple text fallback — neighbors + Wikipedia + LLM reasoning

    Returns: HybridResult with answer, method, route_source, and sparql_fallback.
    """
    if debug:
        verbose = True
    sep = "=" * 70
    llm_client = llm_client or Groq(api_key=api_key)
    linker = linker or DBpediaEntityLinker(min_confidence=0.5)
    kg_client = kg_client or DBpediaSPARQLClient()
    schema_disc = schema_disc or SchemaDiscovery()

    # === Step 1: Entity Linking ===
    entities = link_entities_to_dbpedia(question, linker)
    if verbose:
        print(f"\n{sep}")
        print("STEP 1: Entity Linking (Spotlight + spaCy fallback)")
        print(sep)
        for e in entities:
            print(f"  - {e.label}  [{e.source or 'unknown'}]")
            if e.uri:
                print(f"    URI: {e.uri}")
        if not entities:
            print("  (none found)")

    # === Step 2: Schema Discovery ===
    schema_dicts = entities_to_schema_dicts(entities)
    schema_ctx = get_schema_for_uris(schema_disc, schema_dicts, question=question) if schema_dicts else ""
    if verbose:
        print(f"\n{sep}")
        print("STEP 2: Schema Discovery")
        print(sep)
        if schema_ctx:
            for line in schema_ctx.split("\n")[:20]:
                print(f"  {line}")
            if schema_ctx.count("\n") > 20:
                print(f"  ... ({schema_ctx.count(chr(10))} lines total)")
        else:
            print("  (no URIs to query schema for)")

    # === Guard: comparison questions with incomplete entity coverage → LogicRAG ===
    if schema_ctx and _has_incomplete_entity_coverage(question, entities, schema_ctx):
        if verbose:
            print(f"\n{sep}")
            print("STEP 3: Route Decision → LogicRAG (incomplete entity coverage)")
            print(sep)
            linked = [e.label for e in entities]
            print(f"  Linked entities: {linked}")
            print("  Question mentions entities not in schema — skipping SPARQL")
        logicrag_answer = _try_logicrag_decomposition(
            question, entities, schema_ctx, llm_client, model,
            linker, kg_client, schema_disc, verbose, debug=debug,
        )
        if logicrag_answer:
            return HybridResult(
                logicrag_answer, "logicrag_decomposed", "incomplete_coverage",
                sparql_query="[logicrag_decomposed]",
            )
        # LogicRAG failed — fall through to normal TEXT fallback below

    # === Step 3: Route Decision — can SPARQL answer this? ===
    sparql_answer = ""
    route = "TEXT"
    route_source = "no_schema_fallback"
    if schema_ctx:
        # Rule analysis as hint (not hard decision — LLM always makes final call)
        rule_result = _rule_based_route(question, schema_ctx, debug=debug)
        if rule_result:
            hint = f"\nSchema analysis suggests: {rule_result[1]} (SPARQL likely feasible)\n"
        else:
            hint = ""

        # Escape curly braces in dynamic content to avoid .format() crashes
        safe_schema = schema_ctx.replace("{", "{{").replace("}", "}}")
        safe_question = question.replace("{", "{{").replace("}", "}}")
        route_prompt = ROUTE_PROMPT.format(
            schema=safe_schema, question=safe_question, hint=hint,
        )
        route_resp = call_llm(llm_client, route_prompt, model).strip()
        route_match = re.search(r'\b(SPARQL|TEXT)\b', route_resp, re.IGNORECASE)
        route = route_match.group(1).upper() if route_match else "TEXT"

        # Preserve rule route_source for bridge budget detection when LLM agrees
        if rule_result and route == "SPARQL":
            route_source = rule_result[1]
        else:
            route_source = "llm"

        if verbose:
            print(f"\n{sep}")
            print(f"STEP 3: Route Decision → {route} ({route_source})")
            print(sep)
            if rule_result:
                print(f"  Rule hint: {rule_result[1]}")
            print(f"  LLM response: {route_resp[:120]}")

    # === Step 4: SPARQL Generation + Execution (only if routed) ===
    executed_sparql_query = ""  # provenance: the SPARQL query that produced the answer
    if route == "SPARQL" and schema_ctx:
        if verbose:
            print(f"\n{sep}")
            print("STEP 4: SPARQL Generation + Execution")
            print(sep)

        sparql_prompt = SPARQL_GEN_PROMPT.format(schema=schema_ctx, question=question)
        sparql_resp = call_llm(llm_client, sparql_prompt, model)
        sparql_query = sanitize_sparql_uris(extract_sparql(sparql_resp))
        executed_sparql_query = sparql_query
        if verbose:
            print(f"  Generated SPARQL:\n    {sparql_query[:200]}")

        results, err = execute_sparql(sparql_query)
        if verbose:
            print(f"  Results: {len(results)} rows, error: '{err[:80]}'" if err else f"  Results: {len(results)} rows")

        if not results or err:
            if debug:
                print(f"  [DEBUG] SPARQL error: {err or 'empty results'}")
                print(f"  [DEBUG] Full query:\n    {sparql_query}")
            problem = f"an error: {err}" if err else "empty results (no rows)"
            retry_prompt = SPARQL_RETRY_PROMPT.format(
                problem=problem, sparql=sparql_query,
                schema=schema_ctx, question=question,
            )
            retry_resp = call_llm(llm_client, retry_prompt, model)
            retry_query = sanitize_sparql_uris(extract_sparql(retry_resp))
            if verbose:
                print(f"  Retry SPARQL:\n    {retry_query[:200]}")
            results2, err2 = execute_sparql(retry_query)
            if verbose:
                print(f"  Retry results: {len(results2)} rows")
            if results2:
                results = results2
                executed_sparql_query = retry_query
            elif debug:
                print(f"  [DEBUG] Retry also failed: {err2 or 'empty results'}")
                print(f"  [DEBUG] Retry query:\n    {retry_query}")
                print("  [DEBUG] -> Falling through to decomposition / TEXT")

        if results:
            fmt_results = json.dumps(results[:20], indent=2)
            answer_prompt = ANSWER_PROMPT.format(
                question=question, results=fmt_results,
            )
            sparql_answer = extract_final_answer(
                call_llm(llm_client, answer_prompt, model)
            )
            if verbose:
                print(f"  SPARQL answer: {sparql_answer}")

    # === Return SPARQL answer or fall back to text ===
    if sparql_answer:
        if verbose:
            print(f"\n{sep}")
            print(f"RESULT (method=sparql, route={route_source})")
            print(sep)
            print(f"  Question: {question}")
            print(f"  Answer:   {sparql_answer}")
        return HybridResult(
            sparql_answer, "sparql", route_source,
            sparql_query=executed_sparql_query,
        )

    # === Step 5: LogicRAG Decomposition (all multi-hop) ===
    sparql_fallback = route == "SPARQL"  # originally routed SPARQL but it failed
    logicrag_answer = _try_logicrag_decomposition(
        question, entities, schema_ctx, llm_client, model,
        linker, kg_client, schema_disc, verbose, debug=debug,
    )
    if logicrag_answer:
        return HybridResult(
            logicrag_answer, "logicrag_decomposed", route_source,
            sparql_fallback=sparql_fallback,
            sparql_query="[logicrag_decomposed]",
        )

    # === Step 6: Text Fallback — neighbors + Wikipedia + LLM (last resort) ===
    method_label = "text"

    if verbose:
        print(f"\n{sep}")
        print("STEP 6a: Context entities — SPARQL neighbors")
        print(sep)
    all_entities = _sparql_neighbors(entities, kg_client, verbose=verbose)
    if verbose:
        print(f"  Seed: {len(entities)}, Total: {len(all_entities)}")

    # Build minimal KG facts from neighbor relations
    kg_lines = []
    for e in all_entities:
        if e.discovery_relation and e.source == "sparql_neighbor":
            kg_lines.append(f"- {e.discovery_relation}: {e.label}")
    kg_facts = (
        "Known facts from knowledge graph:\n" + "\n".join(kg_lines) + "\n\n"
        if kg_lines else ""
    )

    # Build relation hints for section boosting
    relation_hints = {}
    for e in all_entities:
        if e.discovery_relation and e.uri and "/resource/" in e.uri:
            title = uri_to_wiki_title(e.uri)
            relation_hints[title] = e.discovery_relation

    # === Wikipedia retrieval ===
    if verbose:
        print(f"\n{sep}")
        print("STEP 6b: Wikipedia retrieval")
        print(sep)
        if kg_facts:
            print(f"  KG facts ({len(kg_facts)} chars) will be injected into prompt")
    raw_texts = retrieve_wikipedia(all_entities, question, verbose=verbose)
    if not raw_texts:
        if verbose:
            print("  No Wikipedia text retrieved.")
        return HybridResult(
            "Could not find answer", method_label, route_source,
            sparql_fallback, kg_context=kg_facts,
        )

    # Section extraction and context assembly (seed entities get budget priority)
    if route_source in ("rule:bridge_comparison", "rule:bridge_hop2", "rule:parent_chain"):
        # Bridge/parent questions: answer is on discovered entities (directors,
        # parents), not the Spotlight entities (films). Prioritize bridge pages.
        spotlight_uris = {e.uri for e in entities if e.uri}
        # When no bridge entities found, seed_titles is empty (falsy) ->
        # assemble_context uses uniform budget (intended fallback).
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in all_entities
            if e.uri and "/resource/" in e.uri and e.uri not in spotlight_uris
        }
    else:
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in entities if e.uri and "/resource/" in e.uri
        }
    context, filtered_texts = assemble_context(
        raw_texts, question, seed_titles=seed_titles,
        kg_facts=kg_facts, relation_hints=relation_hints, verbose=verbose,
    )

    # LLM reasoning
    if verbose:
        print(f"\n{sep}")
        print("STEP 6c: LLM Reasoning over text")
        print(sep)
        print(f"  Context: {len(context)} chars from {len(filtered_texts)} pages")
        if kg_facts:
            print(f"  KG facts: {len(kg_facts)} chars injected")

    prompt = TEXT_REASON_PROMPT.format(
        question=question, context=context, kg_facts=kg_facts,
    )
    response = call_llm(llm_client, prompt, model)

    answer = extract_final_answer(response) if response else "Could not generate answer"

    if verbose:
        fb_tag = " [sparql_fallback]" if sparql_fallback else ""
        print(f"\n{sep}")
        print(f"RESULT (method={method_label}, route={route_source}{fb_tag})")
        print(sep)
        print(f"  Question: {question}")
        print(f"  Answer:   {answer}")
        print(f"  Pages:    {len(filtered_texts)} ({', '.join(filtered_texts.keys())})")

    return HybridResult(
        answer, method_label, route_source, sparql_fallback,
        kg_context=kg_facts,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Schema-Routed Hybrid QA: SPARQL-first + text fallback"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GROQ_API_KEY"),
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="LLM model (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show pipeline steps")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show SPARQL gap analysis and decomposition traces (implies --verbose)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY env var")
        sys.exit(1)

    result = hybrid_answer(
        args.question, args.api_key, model=args.model,
        verbose=args.verbose, debug=args.debug,
    )
    if not args.verbose:
        fb = " [sparql_fallback]" if result.sparql_fallback else ""
        print(f"Answer ({result.method}, route={result.route_source}{fb}): {result.answer}")


if __name__ == "__main__":
    main()
