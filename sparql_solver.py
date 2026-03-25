"""Unified SPARQL solver: generate, execute, retry, extract answer.

Single entry point `solve()` used by both the direct SPARQL path in hybrid_qa.py
and per-sub-question resolution in logicrag_decomposition.py.

Imports helper functions from sparql_evaluation.py (which stays standalone).
"""

import json
import logging
import re

from dbpedia_qa.models import Entity, extract_final_answer
from llm_provider import call_llm
from routing import _sparql_results_to_entities
from sparql_evaluation import (
    ANSWER_PROMPT,
    SPARQL_RETRY_PROMPT,
    execute_sparql,
    extract_sparql,
    sanitize_sparql_uris,
)

logger = logging.getLogger(__name__)

_last_solve_meta: dict = {}

# ---------------------------------------------------------------------------
# SPARQL read-only guard (C1: injection prevention)
# ---------------------------------------------------------------------------

_WRITE_OPS = re.compile(
    r'\b(INSERT|DELETE|DROP|CLEAR|CREATE|LOAD|COPY|MOVE|ADD|SERVICE)\b',
    re.IGNORECASE,
)


def _is_readonly_sparql(query: str) -> bool:
    """Return True iff query contains no SPARQL write operations."""
    return not _WRITE_OPS.search(query)


# Null-answer patterns returned by the LLM that should be treated as no answer
_NULL_ANSWERS = {
    "could not", "n/a", "unknown", "none", "not found", "no answer",
    "could not determine", "no information", "not available",
}

# DBpedia list-article patterns — these are pointers to disambiguation/list pages,
# not actual answers. Detected in both raw URIs and LLM-decoded label strings.
_LIST_ARTICLE_PATTERN = re.compile(
    r'(?:^|/)List[_ ]of[_ ]'          # "List of ..." or "List_of_..." (URI or label)
    r'|discography$'                    # "...discography"
    r'|filmography$'                    # "...filmography"
    r'|bibliography$'                   # "...bibliography"
    r'|Awards[_ ]and[_ ]nominations',   # "Awards and nominations received by ..."
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Unified SPARQL generation prompt (with optional known_facts for sub-Qs)
# ---------------------------------------------------------------------------

UNIFIED_SPARQL_PROMPT = """You are a SPARQL expert for DBpedia. Generate a query to answer the question.

{schema}
{known_facts_block}
QUESTION: {question}

RULES:
1. Use the EXACT entity URIs provided in the schema
2. For multi-hop queries, chain triple patterns using dbo: properties (they return URIs, not strings).
   Use the "2-Hop chains" section in the schema to identify valid chains.
   Example: if schema shows dbo:director -> Person: birthDate, write:
   ?film dbo:director ?person . ?person dbo:birthDate ?date .
3. Prefer dbo: properties for multi-hop joins; only use dbp: for single-hop leaf values
4. Always add LIMIT 100
5. For comparison questions (who/which was born first, etc.), return data for BOTH entities so results can be compared
6. If an entity URI contains special characters (apostrophes, commas, parentheses), use full IRI syntax <http://dbpedia.org/resource/...> instead of dbr: prefix
7. If known facts provide entity URIs, use those URIs directly in your query
8. Generate ONLY the SPARQL query, no explanation"""


def solve(
    question: str,
    schema_ctx: str,
    entities: list[Entity],
    llm_client,
    model: str,
    known_facts: str = "",
    verbose: bool = False,
    debug: bool = False,
    preserve_uris: bool = False,
) -> tuple[str | None, str, list[Entity]]:
    """Generate SPARQL, execute with retry, extract answer.

    Args:
        question: The question to answer via SPARQL.
        schema_ctx: Schema context string for the entities.
        entities: Linked entities (used for seed URI tracking).
        llm_client: OpenAI-compatible LLM client.
        model: LLM model name.
        known_facts: Optional accumulated facts from prior sub-questions.
            Empty string for direct queries, populated for LogicRAG sub-Qs.
        verbose: Print pipeline steps.
        debug: Print debug details (implies verbose).
        preserve_uris: Pass to execute_sparql to keep full URIs.

    Returns:
        (answer, sparql_query, discovered_entities)
        answer is None if SPARQL resolution fails.
    """
    if debug:
        verbose = True

    global _last_solve_meta
    _last_solve_meta = {}

    # Build prompt — unified for direct and sub-Q paths
    known_facts_block = ""
    if known_facts:
        known_facts_block = f"\nKnown facts (from previous steps):\n{known_facts}\n"

    sparql_prompt = UNIFIED_SPARQL_PROMPT.format(
        schema=schema_ctx,
        question=question,
        known_facts_block=known_facts_block,
    )
    sparql_resp = call_llm(llm_client, sparql_prompt, model)
    sparql_query = sanitize_sparql_uris(extract_sparql(sparql_resp))
    _last_solve_meta["sparql_gen_prompt"] = sparql_prompt
    _last_solve_meta["sparql_gen_response"] = sparql_resp
    _last_solve_meta["initial_query"] = sparql_query

    if verbose:
        print(f"  Generated SPARQL:\n    {sparql_query[:200]}")

    # Reject write operations before executing
    if not _is_readonly_sparql(sparql_query):
        logger.warning("Rejected non-read-only SPARQL: %s", sparql_query[:100])
        return None, sparql_query, []

    # Execute
    results, err = execute_sparql(sparql_query, preserve_uris=preserve_uris)
    _last_solve_meta["initial_results"] = results[:20] if results else []
    _last_solve_meta["initial_result_count"] = len(results) if results else 0
    _last_solve_meta["initial_error"] = err
    if verbose:
        if err:
            print(f"  Results: {len(results)} rows, error: '{err[:80]}'")
        else:
            print(f"  Results: {len(results)} rows")

    # Retry on failure
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
        _last_solve_meta["retry_prompt"] = retry_prompt
        _last_solve_meta["retry_response"] = retry_resp
        _last_solve_meta["retry_query"] = retry_query
        if verbose:
            print(f"  Retry SPARQL:\n    {retry_query[:200]}")
        if not _is_readonly_sparql(retry_query):
            logger.warning("Rejected non-read-only retry SPARQL: %s", retry_query[:100])
            results2, err2 = [], "write operation rejected"
        else:
            results2, err2 = execute_sparql(retry_query, preserve_uris=preserve_uris)
        _last_solve_meta["retry_results"] = results2[:20] if results2 else []
        _last_solve_meta["retry_result_count"] = len(results2) if results2 else 0
        _last_solve_meta["retry_error"] = err2
        _last_solve_meta["used_retry"] = True
        if verbose:
            print(f"  Retry results: {len(results2)} rows")
        if results2:
            results = results2
            sparql_query = retry_query
        elif debug:
            print(f"  [DEBUG] Retry also failed: {err2 or 'empty results'}")
            print(f"  [DEBUG] Retry query:\n    {retry_query}")

    if not results:
        return None, sparql_query, []

    # Extract answer from results
    # For sub-Q paths with preserve_uris, produce human-readable results for LLM
    if preserve_uris:
        from urllib.parse import unquote
        readable = []
        for row in results[:10]:
            rr = {}
            for k, v in row.items():
                if isinstance(v, str) and v.startswith("http://"):
                    rr[k] = unquote(v.split("/")[-1]).replace("_", " ")
                else:
                    rr[k] = v
            readable.append(rr)
        fmt_results = json.dumps(readable, indent=2)
    else:
        fmt_results = json.dumps(results[:20], indent=2)

    answer_prompt = ANSWER_PROMPT.format(
        question=question, results=fmt_results,
    )
    answer_raw_response = call_llm(llm_client, answer_prompt, model)
    _last_solve_meta["answer_prompt"] = answer_prompt
    _last_solve_meta["answer_raw_response"] = answer_raw_response
    answer = extract_final_answer(answer_raw_response)

    if answer:
        clean = answer.strip().lower().rstrip(".")
        if clean in _NULL_ANSWERS:
            answer = None

    if answer and _LIST_ARTICLE_PATTERN.search(answer.strip()):
        logger.debug("Rejected list-article answer: %s", answer)
        answer = None

    # Discover entities from SPARQL results
    seed_uris = {e.uri for e in entities if e.uri}
    discovered = _sparql_results_to_entities(results, seed_uris)

    if verbose and answer:
        print(f"  SPARQL answer: {answer}")

    return answer, sparql_query, discovered
