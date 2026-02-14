#!/usr/bin/env python3
"""Schema-Routed Hybrid QA: uses KG schema to route SPARQL vs text.

Pipeline:
  1. EntityLink(q)            — DBpedia Spotlight (spaCy fallback)
  2. SchemaDiscovery(E)       — entity classes, properties, sample values
  3. RouteDecision(q, S)      — LLM examines schema -> SPARQL | TEXT
  4. if SPARQL: GenerateSPARQL(q, S) -> Execute with retry -> answer
  5. if TEXT (or SPARQL fail): auto-chain context building:
     a) sparql_context  — LLM generates SPARQL to find context entities
     b) explore         — schema-guided KG exploration (fallback if a finds nothing)
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
from dbpedia_qa.models import Entity

from graphrag_wiki import (
    link_entities_to_dbpedia,
    explore_kg,
    retrieve_wikipedia,
    assemble_context,
)
from sparql_evaluation import (
    get_schema_for_uris,
    call_llm,
    extract_sparql,
    execute_sparql,
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

ROUTE_PROMPT = """You are a SPARQL expert for DBpedia. Given a question and the schema of linked entities, decide if a SPARQL query can answer this question.

{schema}

QUESTION: {question}

Look at the properties available for each entity. Can the answer be found by querying these properties?

Reply with EXACTLY one word:
- SPARQL — if the schema contains properties that can answer the question (e.g., dbo:director, dbp:birthDate, dbo:author)
- TEXT — if the schema lacks the needed properties and we should search Wikipedia text instead"""

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

CONTEXT_SPARQL_RETRY_PROMPT = """Your previous context-retrieval SPARQL query returned {problem}.

Previous query:
{sparql}

Fix the query. We need to find related DBpedia entities whose Wikipedia pages help answer the question.

{schema}

QUESTION: {question}

Generate ONLY the corrected SPARQL query:"""


# ---------------------------------------------------------------------------
# Helpers
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

def sparql_context_fallback(question: str, entities: list[Entity],
                            schema_ctx: str, llm_client, model: str,
                            verbose: bool = False) -> list[Entity]:
    """LLM generates SPARQL to find entities for context retrieval.

    Returns Entity objects whose Wikipedia pages should be fetched.
    """
    sep = "=" * 70
    seed_uris = {e.uri for e in entities if e.uri}

    if verbose:
        print(f"\n{sep}")
        print("STEP 4: Text Fallback — LLM-generated context SPARQL")
        print(sep)

    # Generate context SPARQL
    prompt = CONTEXT_SPARQL_PROMPT.format(schema=schema_ctx, question=question)
    resp = call_llm(llm_client, prompt, model)
    sparql_query = extract_sparql(resp)
    if verbose:
        print(f"  Context SPARQL:\n    {sparql_query[:300]}")

    # Execute
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
        retry_query = extract_sparql(retry_resp)
        if verbose:
            print(f"  Retry SPARQL:\n    {retry_query[:300]}")
        results2, err2 = execute_sparql(retry_query)
        if verbose:
            print(f"  Retry results: {len(results2)} rows")
        if results2:
            results = results2

    # Convert to entities
    context_entities = _sparql_results_to_entities(results, seed_uris)

    # Filter: keep entities whose labels share words with question or seeds
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
# Text fallback: programmatic explore_kg (original method)
# ---------------------------------------------------------------------------

def explore_kg_fallback(question: str, entities: list[Entity],
                        kg_client: DBpediaSPARQLClient,
                        schema_disc: SchemaDiscovery,
                        verbose: bool = False) -> list[Entity]:
    """Schema-guided KG exploration (programmatic, no LLM call)."""
    sep = "=" * 70
    if verbose:
        print(f"\n{sep}")
        print("STEP 4: Text Fallback — Schema-guided KG exploration")
        print(sep)

    all_entities = explore_kg(entities, kg_client, schema_disc, verbose=verbose)
    if verbose:
        seed = len(entities)
        print(f"  Seed: {seed}, Total: {len(all_entities)}")

    return all_entities


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def hybrid_answer(question: str, api_key: str,
                  model: str = "llama-3.3-70b-versatile",
                  fallback: str = "auto",
                  verbose: bool = False) -> tuple[str, str]:
    """Schema-routed hybrid QA: SPARQL when schema supports it, text otherwise.

    Args:
        fallback: "auto" (sparql_context first, explore if no new entities),
                  "sparql_context" (LLM-generated context SPARQL only), or
                  "explore" (programmatic KG exploration only)

    Returns: (answer, method) where method is "sparql", "text_explore",
             or "text_sparql_context".
    """
    sep = "=" * 70
    llm_client = Groq(api_key=api_key)
    linker = DBpediaEntityLinker(min_confidence=0.5)
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()

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
    if schema_ctx:
        route_prompt = ROUTE_PROMPT.format(schema=schema_ctx, question=question)
        route_resp = call_llm(llm_client, route_prompt, model).strip().upper()
        route = "SPARQL" if "SPARQL" in route_resp else "TEXT"
        if verbose:
            print(f"\n{sep}")
            print(f"STEP 3: Route Decision → {route}")
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
        sparql_query = extract_sparql(sparql_resp)
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
            retry_query = extract_sparql(retry_resp)
            if verbose:
                print(f"  Retry SPARQL:\n    {retry_query[:200]}")
            results2, err2 = execute_sparql(retry_query)
            if verbose:
                print(f"  Retry results: {len(results2)} rows")
            if results2:
                results = results2

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
            print("RESULT (method=sparql)")
            print(sep)
            print(f"  Question: {question}")
            print(f"  Answer:   {sparql_answer}")
        return sparql_answer, "sparql"

    # === Text Fallback — choose method ===
    # "auto" (default): try sparql_context first, fall back to explore
    # "sparql_context": only LLM-generated context SPARQL
    # "explore": only programmatic KG exploration
    all_entities = []
    method_label = ""
    seed_uris = {e.uri for e in entities if e.uri}

    if fallback in ("auto", "sparql_context") and schema_ctx:
        all_entities = sparql_context_fallback(
            question, entities, schema_ctx, llm_client, model, verbose,
        )
        method_label = "text_sparql_context"

    # Auto-chain: fall back to explore if sparql_context found no NEW entities
    new_entities = [e for e in all_entities if e.uri not in seed_uris]
    if not new_entities and fallback in ("auto", "explore"):
        if verbose and fallback == "auto" and schema_ctx:
            print(f"\n{'=' * 70}")
            print("SPARQL context found no new entities — falling back to KG exploration")
            print("=" * 70)
        all_entities = explore_kg_fallback(
            question, entities, kg_client, schema_disc, verbose,
        )
        method_label = "text_explore"

    # === Wikipedia retrieval (shared by both methods) ===
    if verbose:
        print(f"\n{sep}")
        print("STEP 5: Wikipedia retrieval")
        print(sep)
    raw_texts = retrieve_wikipedia(all_entities, question, verbose=verbose)
    if not raw_texts:
        if verbose:
            print("  No Wikipedia text retrieved.")
        return "Could not find answer", method_label

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
        print(f"\n{sep}")
        print(f"RESULT (method={method_label})")
        print(sep)
        print(f"  Question: {question}")
        print(f"  Answer:   {answer}")
        print(f"  Pages:    {len(filtered_texts)} ({', '.join(filtered_texts.keys())})")

    return answer, method_label


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
    parser.add_argument(
        "--fallback",
        choices=["auto", "explore", "sparql_context"],
        default="auto",
        help="Text fallback strategy: 'auto' (try sparql_context first, "
             "then explore), 'explore' (programmatic KG only), or "
             "'sparql_context' (LLM-generated SPARQL only)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show pipeline steps")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY env var")
        sys.exit(1)

    answer, method = hybrid_answer(
        args.question, args.api_key, model=args.model,
        fallback=args.fallback, verbose=args.verbose,
    )
    if not args.verbose:
        print(f"Answer ({method}): {answer}")


if __name__ == "__main__":
    main()
