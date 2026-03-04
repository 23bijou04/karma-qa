"""LogicRAG-style sub-question decomposition with schema-aware SPARQL/TEXT routing.

Decomposes multi-hop questions into dependency-ordered sub-questions, routes each
to SPARQL or TEXT based on schema coverage, chains grounded answers downstream.

Key functions:
  _try_logicrag_decomposition()  — Main entry point (called from hybrid_qa.py)
  _decompose_subquestions()      — LLM decomposes question into sub-Qs
  _resolve_subquestion()         — Routes + resolves a single sub-Q
"""

import json
import logging
import re
import sys
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from urllib.parse import quote as url_quote, unquote

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from ablation_study_v3.wiki_utils import uri_to_wiki_title
from dbpedia_qa.models import Entity, extract_final_answer
from graphrag_wiki import assemble_context, retrieve_wikipedia
from sparql_evaluation import (
    ANSWER_PROMPT,
    SPARQL_RETRY_PROMPT,
    call_llm,
    execute_sparql,
    extract_sparql,
    get_schema_for_uris,
    sanitize_sparql_uris,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SubQuestion:
    """A decomposed sub-question with dependency tracking."""

    text: str
    depends_on: list[int] = dc_field(default_factory=list)
    original_idx: int = -1
    answer: str | None = None
    method: str | None = None
    entities: list[Entity] | None = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """Decompose this multi-hop question into independent sub-questions.

QUESTION: {question}
ENTITIES: {entity_names}
SCHEMA SUMMARY: {schema_summary}
{type_hint}
Rules:
- Each sub-question should ask for ONE fact
- If a sub-question needs the answer from another, mark the dependency
- For comparison questions: extract the same property for each entity (parallel, no dependencies)
- For bridge questions: first find the intermediate entity, then ask about its property
- If the question is simple (single fact lookup), return exactly 1 sub-question with no dependencies

Return JSON:
{{
  "sub_questions": ["sub-Q1 text", "sub-Q2 text", ...],
  "dependencies": [[dependent_idx, dependency_idx], ...]
}}

Example - Bridge: "When was the director of Inception born?"
{{
  "sub_questions": ["Who directed Inception?", "When was [director] born?"],
  "dependencies": [[1, 0]]
}}

Example - Comparison: "Who was born first, the director of X or Y?"
{{
  "sub_questions": ["Who directed X?", "Who directed Y?",
    "When was [director of X] born?", "When was [director of Y] born?"],
  "dependencies": [[2, 0], [3, 1]]
}}

Example - Simple: "Who directed Inception?"
{{
  "sub_questions": ["Who directed Inception?"],
  "dependencies": []
}}

JSON only, no explanation:"""

SUBQ_SPARQL_PROMPT = """Generate a SPARQL query to answer this sub-question about DBpedia entities.

{schema}

Known facts (from previous steps):
{known_facts}

SUB-QUESTION: {sub_question}

Rules:
- Use EXACT URIs from the schema
- Use known facts to resolve entity references
- Return a simple SELECT query with ?answer as the result variable
- If the question asks about a known entity's property, query that property directly

Generate ONLY the SPARQL query:"""

SYNTHESIS_PROMPT = """Answer the original question using ONLY the verified facts below.

QUESTION: {question}

VERIFIED FACTS:
{facts_block}

Instructions:
- Facts marked [SPARQL] are verified KG data -- trust them completely
- Facts marked [TEXT] are extracted from Wikipedia -- trust but verify consistency
- Combine the facts to answer the question step-by-step
- If facts are insufficient, say what is missing
- End with exactly: FINAL ANSWER: <answer>"""

TEXT_SUBQ_PROMPT = """Answer this sub-question using the sources below.

Sub-question: {question}

{kg_facts}Wikipedia text:
{context}

Instructions:
- Knowledge graph facts (if present) are verified structured data -- trust them
- Use Wikipedia text to fill gaps not covered by KG facts
- Give a brief, factual answer
- End with exactly: FINAL ANSWER: <answer>
"""


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def _topological_sort_indices(n: int, edges: list[list[int]]) -> list[int]:
    """Topological sort of n nodes given directed edges [[dependent, dependency], ...].

    Uses Kahn's algorithm. Returns indices in execution order.
    Falls back to 0..n-1 on cycles.
    """
    if n == 0:
        return []

    # Build adjacency: dependency -> [dependents]
    graph: dict[int, list[int]] = {i: [] for i in range(n)}
    in_degree: dict[int, int] = {i: 0 for i in range(n)}
    for dependent, dependency in edges:
        if 0 <= dependent < n and 0 <= dependency < n:
            graph[dependency].append(dependent)
            in_degree[dependent] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == n else list(range(n))


# ---------------------------------------------------------------------------
# Answer substitution
# ---------------------------------------------------------------------------

def _substitute_answers(text: str, known_facts: dict[int, str]) -> str:
    """Replace placeholder references in sub-question text with resolved answers.

    Uses belt-and-suspenders approach:
    1. Regex: replace bracketed placeholders like [director], [answer to Q0]
    2. Direct: replace generic references like "Q0", "sub-question 0"
    """
    result = text
    # Pass 1: Replace bracketed placeholders (ordered by index)
    for idx in sorted(known_facts.keys()):
        answer = known_facts[idx]
        result = re.sub(r'\[.*?\]', answer, result, count=1)
    # Pass 2: Direct string replacement for known answer values
    for idx, answer in known_facts.items():
        if answer and answer not in result:
            for pattern in [f"Q{idx}", f"sub-question {idx}", f"answer {idx}"]:
                if pattern.lower() in result.lower():
                    result = re.sub(
                        re.escape(pattern), answer, result, flags=re.IGNORECASE,
                    )
                    break
    return result


# ---------------------------------------------------------------------------
# Bridge entity resolution
# ---------------------------------------------------------------------------

def _split_compound_answer(answer: str) -> list[str]:
    """Split compound answers like 'A and B' into candidate names."""
    parts = re.split(r'\s+and\s+', answer, maxsplit=2)
    return [p.strip() for p in parts if p.strip()]


def _make_fallback_entity(name: str, verbose: bool = False) -> Entity:
    """Construct unverified Entity from name (fallback when SPARQL linking fails)."""
    local_name = url_quote(name.replace(' ', '_'), safe="_")
    uri = f"http://dbpedia.org/resource/{local_name}"
    if verbose:
        print(f"    Guessed bridge entity URI: {uri}")
    return Entity(uri=uri, label=name, source="logicrag_guessed")


def _resolve_answer_entities(
    answer_val: str, kg_client, verbose: bool = False,
) -> list[Entity]:
    """Resolve a sub-Q answer string to validated DBpedia entities.

    Strategy: try-unsplit-first, then split on ' and ', validate each.
    Falls back to blind URI construction if SPARQL linking fails.
    """
    if not answer_val or re.match(r'^[\d,.\-/\s]+$', answer_val.strip()):
        return []

    answer_val = answer_val.strip().rstrip('.')

    # Try full answer as single entity first ("Simon and Garfunkel" case)
    linked = kg_client.search_entity_by_name(answer_val)
    if linked:
        if verbose:
            print(f"    Linked bridge entity: '{answer_val}' -> {linked.uri}")
        return [linked]

    # Split on " and " and try each fragment
    parts = _split_compound_answer(answer_val)
    if len(parts) <= 1:
        return [_make_fallback_entity(answer_val, verbose)]

    entities = []
    for part in parts:
        linked = kg_client.search_entity_by_name(part)
        if linked:
            entities.append(linked)
            if verbose:
                print(f"    Linked bridge entity: '{part}' -> {linked.uri}")
        else:
            entities.append(_make_fallback_entity(part, verbose))
    return entities


# ---------------------------------------------------------------------------
# Decomposition (Phase 1)
# ---------------------------------------------------------------------------

def _decompose_subquestions(
    question: str, entities: list[Entity],
    schema_ctx: str, llm_client, model: str,
    verbose: bool = False, debug: bool = False,
    is_comparison: bool = False,
) -> list[SubQuestion] | None:
    """Decompose multi-hop question into dependency-ordered sub-questions.

    Returns None if question is single-hop (1 sub-question) or decomposition fails.
    """
    entity_names = ", ".join(e.label for e in entities) if entities else "none"
    # Compact schema summary: entity classes and key properties
    schema_lines = []
    for line in schema_ctx.split("\n")[:30]:
        stripped = line.strip()
        if stripped.startswith(("Entity:", "Class:", "dbo:", "dbp:", "2-Hop")):
            schema_lines.append(stripped)
    schema_summary = "\n".join(schema_lines) if schema_lines else "No schema available"

    type_hint = ""
    if is_comparison:
        type_hint = ("This is a COMPARISON question. Extract the same property for "
                     "each entity as independent parallel sub-questions (no dependencies "
                     "between entity queries).")

    prompt = DECOMPOSE_PROMPT.format(
        question=question, entity_names=entity_names,
        schema_summary=schema_summary, type_hint=type_hint,
    )
    resp = call_llm(llm_client, prompt, model)

    # Parse JSON response
    try:
        clean = resp.strip()
        clean = clean.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        if verbose:
            logger.warning(f"Decomposition JSON parse failed: {resp[:200]}")
        return None

    sub_qs = data.get("sub_questions", [])
    deps = data.get("dependencies", [])

    if not sub_qs or len(sub_qs) < 2:
        return None  # Single-hop, skip decomposition

    if len(sub_qs) > 6:
        sub_qs = sub_qs[:6]  # Safety cap

    # Build ordered SubQuestion list
    order = _topological_sort_indices(len(sub_qs), deps)
    dep_map: dict[int, list[int]] = {}
    for dependent, dependency in deps:
        if 0 <= dependent < len(sub_qs) and 0 <= dependency < len(sub_qs):
            dep_map.setdefault(dependent, []).append(dependency)

    ordered = []
    for idx in order:
        ordered.append(SubQuestion(
            text=sub_qs[idx],
            depends_on=dep_map.get(idx, []),
            original_idx=idx,
        ))

    if verbose:
        print(f"  Decomposed into {len(ordered)} sub-questions:")
        for i, sq in enumerate(ordered):
            deps_str = f" (depends on: {sq.depends_on})" if sq.depends_on else ""
            print(f"    {i}: {sq.text}{deps_str}")

    return ordered


# ---------------------------------------------------------------------------
# Per-sub-question resolution (Phase 2)
# ---------------------------------------------------------------------------

def _resolve_subquestion(
    sub_q: SubQuestion, idx: int,
    known_facts: dict[int, str],
    known_entities: dict[int, list[Entity]],
    parent_entities: list[Entity],
    parent_schema_ctx: str,
    llm_client, model: str,
    linker, kg_client, schema_disc,
    verbose: bool = False, debug: bool = False,
) -> tuple[str | None, str, list[Entity]]:
    """Resolve a single sub-question via SPARQL or TEXT.

    Returns (answer, method, discovered_entities).
    answer is None if resolution fails.
    """
    # Lazy import to avoid circular dependency
    from hybrid_qa import (
        _rule_based_route,
        _sparql_results_to_entities,
        entities_to_schema_dicts,
    )

    # Step 1: Substitute known answers into sub-question text
    resolved_text = _substitute_answers(sub_q.text, known_facts)
    if verbose:
        print(f"\n    Sub-Q {idx}: {resolved_text}")
    if debug and sub_q.depends_on:
        for dep_idx in sub_q.depends_on:
            if dep_idx in known_facts:
                print(f"    [DEBUG] Q{dep_idx} -> {known_facts[dep_idx]}")

    # Step 2: Gather entities (parent + discovered from dependencies)
    all_entities = list(parent_entities)
    for dep_idx in sub_q.depends_on:
        if dep_idx in known_entities:
            for e in known_entities[dep_idx]:
                if e not in all_entities:
                    all_entities.append(e)

    # Bridge entity linking: resolve answer text to validated DBpedia entities
    for dep_idx in sub_q.depends_on:
        if dep_idx in known_facts:
            resolved = _resolve_answer_entities(
                known_facts[dep_idx], kg_client, verbose=verbose,
            )
            for e in resolved:
                if not any(ex.uri == e.uri for ex in all_entities):
                    all_entities.append(e)

    # Step 3: Build schema context (reuse parent + extend for new entities)
    new_entities = [
        e for e in all_entities
        if e.uri and e.uri not in {pe.uri for pe in parent_entities if pe.uri}
    ]
    if new_entities:
        new_schema_dicts = entities_to_schema_dicts(new_entities)
        new_schema = get_schema_for_uris(schema_disc, new_schema_dicts, question=resolved_text) if new_schema_dicts else ""
        combined_schema = parent_schema_ctx + "\n" + new_schema if new_schema else parent_schema_ctx
    else:
        combined_schema = parent_schema_ctx

    # Step 4: Route decision (rule-based only, no LLM route prompt)
    route_result = _rule_based_route(resolved_text, combined_schema, debug=debug)
    attempt_sparql = route_result is not None or bool(combined_schema)
    if debug:
        tag = route_result[1] if route_result else "no rule match"
        print(f"    [DEBUG] Route: {tag}, attempt_sparql={attempt_sparql}")

    # Step 5a: Try SPARQL
    if attempt_sparql and combined_schema:
        # Build facts with entity URIs for accurate SPARQL generation
        facts_parts = []
        for k, v in known_facts.items():
            line = f"Q{k}: {v}"
            if k in known_entities and known_entities[k]:
                uris = [e.uri for e in known_entities[k] if e.uri]
                if uris:
                    line += f" (URI: <{uris[0]}>)"
            facts_parts.append(line)
        facts_text = "\n".join(facts_parts) or "None"
        sparql_prompt = SUBQ_SPARQL_PROMPT.format(
            schema=combined_schema, known_facts=facts_text,
            sub_question=resolved_text,
        )
        sparql_resp = call_llm(llm_client, sparql_prompt, model)
        sparql_query = sanitize_sparql_uris(extract_sparql(sparql_resp))

        if debug and sparql_query:
            print(f"    [DEBUG] SPARQL query:\n      {sparql_query}")
        if sparql_query:
            results, err = execute_sparql(sparql_query, preserve_uris=True)
            if debug:
                if results:
                    print(f"    [DEBUG] SPARQL -> {len(results)} rows")
                elif err:
                    print(f"    [DEBUG] SPARQL failed: {err}")
                else:
                    print("    [DEBUG] SPARQL -> empty results")

            # Retry on failure (matching hybrid_qa.py pattern)
            if not results and sparql_query:
                problem = f"an error: {err}" if err else "empty results (no rows)"
                retry_prompt = SPARQL_RETRY_PROMPT.format(
                    problem=problem, sparql=sparql_query,
                    schema=combined_schema, question=resolved_text,
                )
                retry_resp = call_llm(llm_client, retry_prompt, model)
                retry_query = sanitize_sparql_uris(extract_sparql(retry_resp))
                if debug:
                    print(f"    [DEBUG] Retry SPARQL:\n      {retry_query}")
                if retry_query:
                    results2, err2 = execute_sparql(retry_query, preserve_uris=True)
                    if results2:
                        results = results2
                        if debug:
                            print(f"    [DEBUG] Retry -> {len(results2)} rows")
                    elif debug:
                        print(f"    [DEBUG] Retry failed: {err2 or 'empty results'}")

            if results:
                # Human-readable results for LLM (strip URIs to local names)
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
                answer_prompt = ANSWER_PROMPT.format(
                    question=resolved_text, results=fmt_results,
                )
                answer = extract_final_answer(call_llm(llm_client, answer_prompt, model))
                if answer and answer.lower() not in ("", "could not", "n/a", "unknown"):
                    discovered = _sparql_results_to_entities(
                        results, {e.uri for e in all_entities if e.uri},
                    )
                    if verbose:
                        print(f"    -> SPARQL answer: {answer}")
                    return answer, "sparql", discovered

    # Step 5b: TEXT fallback
    if debug:
        print(f"    [DEBUG] TEXT fallback, entities: {[e.label for e in all_entities]}")
    raw_texts = retrieve_wikipedia(all_entities, resolved_text, verbose=False)
    if not raw_texts:
        if verbose:
            print("    -> No Wikipedia text, resolution failed")
        return None, "text", []

    kg_facts_lines = "\n".join(f"- Q{k}: {v}" for k, v in known_facts.items())
    kg_facts_block = f"Known facts:\n{kg_facts_lines}\n\n" if kg_facts_lines else ""
    seed_titles = {
        uri_to_wiki_title(e.uri) for e in all_entities
        if e.uri and "/resource/" in e.uri
    }
    context, _ = assemble_context(
        raw_texts, resolved_text, seed_titles=seed_titles,
        verbose=False,
    )
    prompt = TEXT_SUBQ_PROMPT.format(
        question=resolved_text, context=context, kg_facts=kg_facts_block,
    )
    response = call_llm(llm_client, prompt, model)
    answer = extract_final_answer(response) if response else None

    if verbose:
        print(f"    -> TEXT answer: {answer}")

    return answer, "text", []


# ---------------------------------------------------------------------------
# Orchestration (Phase 3)
# ---------------------------------------------------------------------------

def _try_logicrag_decomposition(
    question: str, entities: list[Entity],
    schema_ctx: str, llm_client, model: str,
    linker, kg_client, schema_disc,
    verbose: bool = False, debug: bool = False,
) -> str | None:
    """LogicRAG-style sub-question decomposition with per-sub-Q SPARQL/TEXT routing.

    Decomposes multi-hop question into dependency-ordered sub-questions.
    Each sub-Q is routed independently: SPARQL when schema supports it, TEXT otherwise.
    SPARQL answers become grounded facts injected into downstream sub-Qs.

    Returns answer string if successful, None otherwise (caller falls through
    to existing routing logic).
    """
    if debug:
        verbose = True
    sep = "=" * 70

    # --- Step 1: Decompose ---
    if verbose:
        print(f"\n{sep}")
        print("STEP 3b: LogicRAG Sub-Question Decomposition")
        print(sep)

    # Detect comparison pattern for type hint
    from hybrid_qa import _is_comparison_question
    is_comparison = _is_comparison_question(question, entities)

    sub_questions = _decompose_subquestions(
        question, entities, schema_ctx, llm_client, model, verbose, debug=debug,
        is_comparison=is_comparison,
    )
    if not sub_questions:
        if verbose:
            print("  Decomposition returned None (single-hop or parse failure)")
        return None

    # --- Step 2: Resolve each sub-question in order ---
    known_facts: dict[int, str] = {}
    known_entities: dict[int, list[Entity]] = {}
    methods_used: list[str] = []

    for loop_idx, sub_q in enumerate(sub_questions):
        if verbose:
            print(f"\n  --- Resolving sub-Q {loop_idx}/{len(sub_questions) - 1} ---")

        answer, method, discovered = _resolve_subquestion(
            sub_q, loop_idx,
            known_facts, known_entities,
            entities, schema_ctx,
            llm_client, model,
            linker, kg_client, schema_disc,
            verbose, debug=debug,
        )

        oidx = sub_q.original_idx
        if answer:
            known_facts[oidx] = answer
            sub_q.answer = answer
            sub_q.method = method
        if discovered:
            known_entities[oidx] = discovered
            sub_q.entities = discovered

        methods_used.append(method)

    # --- Step 3: Synthesize final answer from accumulated facts ---
    if not known_facts:
        if verbose:
            print("  No sub-questions resolved, decomposition failed")
        return None

    # If only one fact, return directly
    if len(known_facts) == 1:
        sole_answer = list(known_facts.values())[0]
        if verbose:
            print(f"  Single resolved fact: {sole_answer}")
        return sole_answer

    # Build facts block with provenance labels
    facts_lines = []
    for sq in sub_questions:
        if sq.original_idx in known_facts:
            method_tag = "SPARQL" if sq.method == "sparql" else "TEXT"
            facts_lines.append(
                f"- [{method_tag}] {sq.text} -> {known_facts[sq.original_idx]}"
            )
    facts_block = "\n".join(facts_lines)

    if verbose:
        print("\n  --- Final Synthesis ---")
        print(f"  Facts ({len(known_facts)}):")
        for line in facts_lines:
            print(f"    {line}")

    prompt = SYNTHESIS_PROMPT.format(
        question=question, facts_block=facts_block,
    )
    response = call_llm(llm_client, prompt, model)
    answer = extract_final_answer(response) if response else None

    if verbose:
        print(f"  Synthesized answer: {answer}")

    return answer
