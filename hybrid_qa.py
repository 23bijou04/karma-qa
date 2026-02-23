#!/usr/bin/env python3
"""Schema-Routed Hybrid QA: uses KG schema to route SPARQL vs text.

Pipeline:
  1. EntityLink(q)            — DBpedia Spotlight (spaCy fallback)
  2. SchemaDiscovery(E)       — entity classes, properties, sample values
  3. RouteDecision(q, S)      — rule-based first, LLM fallback -> SPARQL | TEXT
  4. if SPARQL: GenerateSPARQL(q, S) -> Execute with retry -> answer
  5. if TEXT (or SPARQL fail): context entity discovery:
     a) LLM-generated SPARQL  — targeted context entities
     b) SPARQL neighbor query  — reliable fallback if a finds nothing new
  6. WikipediaRetrieval       — fetch pages for context entities
  7. SectionExtraction        — sentence-level keyword scoring
  8. LLM reads passages       -> FINAL ANSWER

Usage:
  python3 hybrid_qa.py "Who directed Inception?" --verbose
  python3 hybrid_qa.py "Who destroys the last Horcrux?" --verbose
  python3 hybrid_qa.py "When was Albert Einstein born?" --api-key $GROQ_API_KEY
"""

import json
import os
import re
import sys
import logging
import argparse
from pathlib import Path
from urllib.parse import unquote

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from groq import Groq

from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery
from dbpedia_qa.dbpedia.schema_discovery import PROPERTY_RANGES, CACHED_CLASS_SCHEMAS
from dbpedia_qa.models import Entity, HybridResult

from graphrag_wiki import (
    link_entities_to_dbpedia,
    retrieve_wikipedia,
    assemble_context,
)
from sparql_evaluation import (
    get_schema_for_uris,
    call_llm,
    extract_sparql,
    execute_sparql,
    sanitize_sparql_uris,
    SPARQL_GEN_PROMPT,
    SPARQL_RETRY_PROMPT,
    ANSWER_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_REASON_PROMPT = """Answer this question using ONLY the Wikipedia text below.

Question: {question}

Wikipedia text:
{context}

Instructions:
- Use ONLY facts from the text above
- Show brief step-by-step reasoning
- End with exactly: FINAL ANSWER: <answer>
"""

ROUTE_PROMPT = """Given a question and DBpedia entity schemas, decide the best route.

{schema}

QUESTION: {question}

DECISION RULES:
- SPARQL if the schema has properties RELATED to the question (exact match not required)
- Comparison questions (older/younger/earlier/later/first/last) with ANY date property -> SPARQL
- Relationship properties (director/author/spouse/parent) with matching property -> SPARQL
- If the schema shows "2-Hop chains", use them:
  e.g., dbo:director -> Person: birthDate means you can compare directors' birth dates via SPARQL
- Questions about a person's relative (father/mother) + any property -> SPARQL
- Choose TEXT only if NO relevant properties exist in the schema at all

Reply with EXACTLY one word: SPARQL or TEXT"""

CONTEXT_SPARQL_PROMPT = """You are a SPARQL expert for DBpedia. The KG schema does not have enough properties to answer this question directly via SPARQL. Instead, we need to find related DBpedia entities whose Wikipedia pages likely contain the answer.

Generate a SPARQL query that finds these related entities. Return entity URIs and their labels.

{schema}

QUESTION: {question}

RULES:
1. Use the EXACT entity URIs from the schema above
2. Only follow properties that are RELEVANT to the question topic — use the schema to pick the right properties
3. Do NOT blindly follow all dbo:/dbp: relations — that returns noise
4. NEVER use dbo:wikiPageWikiLink
5. Return ?entity and ?label columns
6. Include the seed entities themselves in the results
7. Use UNION to combine multiple relation paths
8. Add LIMIT 10
9. Generate ONLY the SPARQL query, no explanation

EXAMPLE for "Who directed Inception?":
SELECT DISTINCT ?entity ?label WHERE {{
  {{ VALUES ?entity {{ <http://dbpedia.org/resource/Inception> }}
     ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  UNION
  {{ <http://dbpedia.org/resource/Inception> dbo:director ?entity .
     ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
}} LIMIT 10"""

SPARQL_DECOMPOSE_PROMPT = """You are a SPARQL expert for DBpedia. Generate a query for ONE entity relevant to a comparison question.

{schema}

QUESTION: {question}
TARGET ENTITY: {entity_name} (<{entity_uri}>)

Generate a SPARQL query that extracts the relevant property value(s) for ONLY this entity.
Use OPTIONAL patterns so the query returns results even if some properties are missing.
Add LIMIT 10.
Generate ONLY the SPARQL query, no explanation."""

CONTEXT_SPARQL_RETRY_PROMPT = """Your previous context-retrieval SPARQL query returned {problem}.

Previous query:
{sparql}

Fix the query. We need to find related DBpedia entities whose Wikipedia pages help answer the question.

{schema}

QUESTION: {question}

Generate ONLY the corrected SPARQL query:"""


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
                r'-\s*dbo:(\w+)\s*->\s*(\w+):\s*(\w+(?:,\s*\w+)*)', stripped
            )
            if m:
                chains[m.group(1)] = (
                    m.group(2),
                    [p.strip() for p in m.group(3).split(",")],
                )
    return chains


def _rule_based_route(question: str, schema_ctx: str) -> tuple[str, str] | None:
    """Return ('SPARQL', reason) if rules match, None if inconclusive (let LLM decide)."""
    q_lower = question.lower()

    # Parse properties from both schema formats:
    #   Format 1: "dbo:prop -> value" (actual properties section)
    actual_props = set(re.findall(r'(?:dbo|dbp):\w+', schema_ctx))
    #   Format 2: "  dbo: prop1, prop2, ..." (cached ontology list, may be truncated)
    for m in re.finditer(r'^\s+(dbo|dbp): ([\w]+(?:,\s*[\w]+)*)', schema_ctx, re.MULTILINE):
        prefix = m.group(1)
        for name in m.group(2).split(','):
            name = name.strip()
            if name:
                actual_props.add(f"{prefix}:{name}")

    comparison_words = [
        "earlier", "later", "older", "younger", "first", "before", "after",
    ]
    date_props = {
        "dbo:birthDate", "dbp:birthDate", "dbo:deathDate", "dbp:deathDate",
        "dbo:activeYearsStartYear", "dbp:established", "dbo:foundingYear",
        "dbp:foundingDate", "dbo:releaseDate", "dbp:released",
    }

    # Comparison + date properties on the entity itself -> SPARQL
    if any(w in q_lower for w in comparison_words):
        if actual_props & date_props:
            return ("SPARQL", "rule:direct_date_comparison")

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
                return ("SPARQL", "rule:bridge_hop2")

    # Parent chain: father/mother + any property question -> SPARQL
    # Handles: "Where did X's father graduate?", "When was X's mother born?"
    parent_words = ["father", "mother", "parent", "paternal", "maternal"]
    parent_props = {"dbo:parent", "dbp:father", "dbp:mother"}
    if any(w in q_lower for w in parent_words) and actual_props & parent_props:
        return ("SPARQL", "rule:parent_chain")

    # Direct property lookup
    direct_map = {
        "direct": {"dbo:director", "dbp:director"},
        "wrote": {"dbo:author", "dbp:author", "dbo:writer", "dbp:writer"},
        "born": {"dbo:birthDate", "dbp:birthDate", "dbo:birthPlace", "dbp:birthPlace"},
        "died": {"dbo:deathDate", "dbp:deathDate", "dbo:deathPlace"},
        "capital": {"dbo:capital", "dbp:capital"},
        "population": {"dbo:populationTotal", "dbp:population"},
        "spouse": {"dbo:spouse", "dbp:spouse"},
        "founded": {"dbo:foundingYear", "dbp:foundingDate", "dbo:founder", "dbp:founder"},
        "nationality": {"dbo:nationality", "dbp:nationality", "dbo:citizenship"},
        "father": {"dbo:parent", "dbp:father"},
        "mother": {"dbo:parent", "dbp:mother"},
        "award": {"dbo:award", "dbp:award"},
        "occupation": {"dbo:occupation", "dbp:occupation"},
    }
    for keyword, props in direct_map.items():
        if keyword in q_lower and actual_props & props:
            return ("SPARQL", "rule:direct_property")

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


def _try_per_entity_sparql(question: str, entities: list[Entity],
                           schema_ctx: str, llm_client, model: str,
                           verbose: bool = False) -> list[dict]:
    """Decompose a comparison query into per-entity SPARQL queries.

    Returns combined results if BOTH entities return data, else empty list.
    Only processes the first 2 entities (2WikiMultiHopQA comparisons are always 2-entity).
    """
    sep = "=" * 70
    if verbose:
        print(f"\n{sep}")
        print("STEP 4c: Per-Entity SPARQL Decomposition (comparison recovery)")
        print(sep)
        print(f"  Entities: {', '.join(e.label for e in entities[:2])}")

    per_entity_results = []
    for i, entity in enumerate(entities[:2]):
        if verbose:
            print(f"\n  --- Entity {i + 1}: {entity.label} ---")

        prompt = SPARQL_DECOMPOSE_PROMPT.format(
            schema=schema_ctx, question=question,
            entity_name=entity.label, entity_uri=entity.uri,
        )
        resp = call_llm(llm_client, prompt, model)
        query = sanitize_sparql_uris(extract_sparql(resp))
        if verbose:
            print(f"  Generated SPARQL:\n    {query}")

        results, err = execute_sparql(query)
        if verbose:
            if err:
                print(f"  Results: {len(results)} rows, error: '{err[:80]}'")
            else:
                print(f"  Results: {len(results)} rows")
            if results:
                for row in results[:5]:
                    print(f"    {row}")

        if not results or err:
            if verbose:
                print(f"  No data for {entity.label} — decomposition failed, falling to TEXT")
            return []

        for row in results:
            row["entity_name"] = entity.label
        per_entity_results.extend(results)

    if verbose:
        print(f"\n  Combined: {len(per_entity_results)} rows from both entities")

    return per_entity_results


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
            all_entities.append(Entity(uri=uri, label=label, source="sparql_neighbor"))
            if verbose:
                prop = b["p"]["value"].split("/")[-1]
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
                # execute_sparql converts URIs to local names (last segment)
                # Reconstruct full URI
                if val.startswith("http://dbpedia.org/resource/"):
                    uri = val
                elif "/" not in val and val.strip():
                    uri = f"http://dbpedia.org/resource/{val.replace(' ', '_')}"
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

        source = "seed" if uri in seed_uris else "sparql_context"
        entities.append(Entity(uri=uri, label=label, source=source))

    return entities


# ---------------------------------------------------------------------------
# Text fallback: LLM-generated context SPARQL
# ---------------------------------------------------------------------------

def _find_context_entities(question: str, entities: list[Entity],
                           schema_ctx: str, llm_client, model: str,
                           kg_client: DBpediaSPARQLClient,
                           verbose: bool = False) -> list[Entity]:
    """Find context entities for Wikipedia retrieval.

    Two-tier approach:
      1. LLM-generated SPARQL (targeted, may fail)
      2. Simple SPARQL neighbor query (reliable fallback)
    Returns Entity objects whose Wikipedia pages should be fetched.
    """
    sep = "=" * 70
    seed_uris = {e.uri for e in entities if e.uri}

    context_entities = []
    used_llm_sparql = False

    # --- Tier 1: LLM-generated context SPARQL (only if schema available) ---
    if schema_ctx:
        if verbose:
            print(f"\n{sep}")
            print("STEP 4a: Context entities — LLM-generated SPARQL")
            print(sep)

        prompt = CONTEXT_SPARQL_PROMPT.format(schema=schema_ctx, question=question)
        resp = call_llm(llm_client, prompt, model)
        sparql_query = sanitize_sparql_uris(extract_sparql(resp))
        if verbose:
            print(f"  Context SPARQL:\n    {sparql_query[:300]}")

        results, err = execute_sparql(sparql_query)
        if verbose:
            status = f"{len(results)} rows"
            if err:
                status += f", error: '{err[:60]}'"
            print(f"  Results: {status}")

        # Retry if empty or error
        if not results or err:
            problem = f"an error: {err}" if err else "empty results (no rows)"
            retry_prompt = CONTEXT_SPARQL_RETRY_PROMPT.format(
                problem=problem, sparql=sparql_query,
                schema=schema_ctx, question=question,
            )
            retry_resp = call_llm(llm_client, retry_prompt, model)
            retry_query = sanitize_sparql_uris(extract_sparql(retry_resp))
            if verbose:
                print(f"  Retry SPARQL:\n    {retry_query[:300]}")
            results2, err2 = execute_sparql(retry_query)
            if verbose:
                print(f"  Retry results: {len(results2)} rows")
            if results2:
                results = results2

        # Convert to entities
        context_entities = _sparql_results_to_entities(results, seed_uris)

        # Relevance filter: keep entities whose labels share words with question/seeds
        q_words = {w.strip("?.,!:;\"'").lower() for w in question.split()
                   if len(w.strip("?.,!:;\"'")) > 2}
        seed_words = set()
        for e in entities:
            seed_words.update(w.lower() for w in e.label.split() if len(w) > 2)
        relevance_words = q_words | seed_words

        filtered = []
        for e in context_entities:
            if e.uri in seed_uris:
                filtered.append(e)
                continue
            label_words = {w.lower().strip("(),") for w in e.label.split()
                           if len(w.strip("(),")) > 2}
            if label_words & relevance_words:
                filtered.append(e)
        if verbose and len(filtered) < len(context_entities):
            dropped = len(context_entities) - len(filtered)
            print(f"  Filtered: dropped {dropped} irrelevant entities")
        context_entities = filtered
        used_llm_sparql = True

    # --- Tier 2: SPARQL neighbor fallback (if LLM SPARQL found no new entities) ---
    new_uris = {e.uri for e in context_entities if e.uri} - seed_uris
    if not new_uris:
        if verbose:
            if used_llm_sparql:
                print(f"\n{sep}")
                print("STEP 4b: LLM SPARQL found no new entities — neighbor fallback")
                print(sep)
            else:
                print(f"\n{sep}")
                print("STEP 4: Context entities — SPARQL neighbors")
                print(sep)
        context_entities = _sparql_neighbors(
            entities, kg_client, verbose=verbose,
        )
        if verbose:
            print(f"  Seed: {len(entities)}, Total: {len(context_entities)}")
        return context_entities

    # Always include seed entities (in case SPARQL missed them)
    seen_uris = {e.uri for e in context_entities}
    for e in entities:
        if e.uri and e.uri not in seen_uris:
            context_entities.insert(0, e)

    if verbose:
        print(f"  Context entities ({len(context_entities)}):")
        for e in context_entities:
            print(f"    - {e.label} [{e.source}]")

    return context_entities


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def hybrid_answer(question: str, api_key: str,
                  model: str = "llama-3.3-70b-versatile",
                  verbose: bool = False,
                  llm_client=None, linker=None,
                  kg_client=None, schema_disc=None) -> tuple[str, str]:
    """Schema-routed hybrid QA: SPARQL when schema supports it, text otherwise.

    Args:
        llm_client: Optional pre-created Groq client (for batch reuse)
        linker: Optional pre-created DBpediaEntityLinker (for batch reuse)
        kg_client: Optional pre-created DBpediaSPARQLClient (for batch reuse)
        schema_disc: Optional pre-created SchemaDiscovery (for batch reuse)

    Returns: HybridResult with answer, method, route_source, and sparql_fallback.
    """
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
    schema_ctx = get_schema_for_uris(schema_disc, schema_dicts) if schema_dicts else ""
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

    # === Step 3: Route Decision — can SPARQL answer this? ===
    sparql_answer = ""
    route = "TEXT"
    route_source = "no_schema_fallback"
    if schema_ctx:
        # Rule-based first (saves an LLM call when rules are conclusive)
        rule_result = _rule_based_route(question, schema_ctx)
        if rule_result:
            route, route_source = rule_result
            if verbose:
                print(f"\n{sep}")
                print(f"STEP 3: Route Decision → {route} ({route_source})")
                print(sep)
        else:
            route_prompt = ROUTE_PROMPT.format(schema=schema_ctx, question=question)
            route_resp = call_llm(llm_client, route_prompt, model).strip().upper()
            route = "SPARQL" if "SPARQL" in route_resp else "TEXT"
            route_source = "llm"
            if verbose:
                print(f"\n{sep}")
                print(f"STEP 3: Route Decision → {route} (llm)")
                print(sep)
                print(f"  LLM response: {route_resp[:80]}")

    # === Step 4: SPARQL Generation + Execution (only if routed) ===
    if route == "SPARQL" and schema_ctx:
        if verbose:
            print(f"\n{sep}")
            print("STEP 4: SPARQL Generation + Execution")
            print(sep)

        sparql_prompt = SPARQL_GEN_PROMPT.format(schema=schema_ctx, question=question)
        sparql_resp = call_llm(llm_client, sparql_prompt, model)
        sparql_query = sanitize_sparql_uris(extract_sparql(sparql_resp))
        if verbose:
            print(f"  Generated SPARQL:\n    {sparql_query[:200]}")

        results, err = execute_sparql(sparql_query)
        if verbose:
            print(f"  Results: {len(results)} rows, error: '{err[:80]}'" if err else f"  Results: {len(results)} rows")

        if not results or err:
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

        # Per-entity decomposition for comparison questions (JOIN failure recovery)
        if not results and _is_comparison_question(question, entities):
            results = _try_per_entity_sparql(
                question, entities, schema_ctx, llm_client, model, verbose,
            )

        if results:
            fmt_results = json.dumps(results[:20], indent=2)
            answer_prompt = ANSWER_PROMPT.format(
                question=question, results=fmt_results,
            )
            sparql_answer = call_llm(llm_client, answer_prompt, model).strip()
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
        return HybridResult(sparql_answer, "sparql", route_source)

    # === Text Fallback — find context entities + Wikipedia ===
    method_label = "text"
    sparql_fallback = route == "SPARQL"  # originally routed SPARQL but it failed
    all_entities = _find_context_entities(
        question, entities, schema_ctx, llm_client, model,
        kg_client, verbose,
    )

    # === Wikipedia retrieval (shared by both methods) ===
    if verbose:
        print(f"\n{sep}")
        print("STEP 5: Wikipedia retrieval")
        print(sep)
    raw_texts = retrieve_wikipedia(all_entities, question, verbose=verbose)
    if not raw_texts:
        if verbose:
            print("  No Wikipedia text retrieved.")
        return HybridResult("Could not find answer", method_label, route_source, sparql_fallback)

    # Section extraction and context assembly
    context, filtered_texts = assemble_context(raw_texts, question, verbose=verbose)

    # LLM reasoning
    if verbose:
        print(f"\n{sep}")
        print("STEP 6: LLM Reasoning over text")
        print(sep)
        print(f"  Context: {len(context)} chars from {len(filtered_texts)} pages")

    prompt = TEXT_REASON_PROMPT.format(question=question, context=context)
    response = call_llm(llm_client, prompt, model)

    answer = response or "Could not generate answer"
    if "FINAL ANSWER:" in answer:
        answer = answer.split("FINAL ANSWER:")[-1].strip()

    if verbose:
        fb_tag = " [sparql_fallback]" if sparql_fallback else ""
        print(f"\n{sep}")
        print(f"RESULT (method={method_label}, route={route_source}{fb_tag})")
        print(sep)
        print(f"  Question: {question}")
        print(f"  Answer:   {answer}")
        print(f"  Pages:    {len(filtered_texts)} ({', '.join(filtered_texts.keys())})")

    return HybridResult(answer, method_label, route_source, sparql_fallback)


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
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY env var")
        sys.exit(1)

    result = hybrid_answer(
        args.question, args.api_key, model=args.model,
        verbose=args.verbose,
    )
    if not args.verbose:
        fb = " [sparql_fallback]" if result.sparql_fallback else ""
        print(f"Answer ({result.method}, route={result.route_source}{fb}): {result.answer}")


if __name__ == "__main__":
    main()
