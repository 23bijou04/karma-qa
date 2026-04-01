#!/usr/bin/env python3
"""Schema-Routed Hybrid QA: uses KG schema to route SPARQL vs decomposition.

Orchestrator-only: delegates SPARQL solving to sparql_solver.py,
text solving to text_solver.py, routing to routing.py,
and multi-hop decomposition to logicrag_decomposition.py.

Two-route architecture:
  1. EntityLink(q)            — DBpedia Spotlight (spaCy fallback)
  2. SchemaDiscovery(E)       — entity classes, properties, sample values
  3. RouteDecision(q, S)      — rule-based + SPARQL ASK verification + LLM fallback
  4a. SPARQL Direct           — sparql_solver.solve() when schema covers the question
  4b. LogicRAG Decomposition  — sub-question decomposition, each sub-Q uses SPARQL or text

Text retrieval (text_solver.solve) is a fallback mechanism within each sub-question
and a last resort if both routes fail — not a separate top-level route.

Usage:
  python3 hybrid_qa.py "Who directed Inception?" --verbose
  python3 hybrid_qa.py "Who destroys the last Horcrux?" --verbose
  python3 hybrid_qa.py "When was Albert Einstein born?" --api-key $GROQ_API_KEY
"""

import argparse
import logging
import sys
import time
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import logicrag_decomposition  # noqa: E402
import routing as routing_mod  # noqa: E402
import sparql_solver  # noqa: E402
import text_solver  # noqa: E402
from ablation_study_v3.wiki_utils import uri_to_wiki_title  # noqa: E402
from dbpedia_qa.dbpedia import (  # noqa: E402
    DBpediaEntityLinker,
    DBpediaSPARQLClient,
    SchemaDiscovery,
    WikipediaResolver,
)
from dbpedia_qa.models import HybridResult  # noqa: E402
from entity_profile import build_entity_profile  # noqa: E402
from graphrag_wiki import link_entities_to_dbpedia  # noqa: E402
from llm_provider import create_llm_client  # noqa: E402
from logicrag_decomposition import _try_logicrag_decomposition  # noqa: E402
from question_analysis import analyze_question  # noqa: E402
from routing import route_question  # noqa: E402
from schema_projection import project  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def hybrid_answer(question: str, api_key: str = None,
                  model: str = "llama-3.3-70b-versatile",
                  verbose: bool = False, debug: bool = False,
                  llm_client=None, linker=None,
                  kg_client=None, schema_disc=None,
                  provider: str = "groq",
                  benchmark_corpus: list[dict] | None = None) -> HybridResult:
    """Schema-routed hybrid QA with two-route architecture.

    Routes:
      1. SPARQL Direct — single-hop or multi-hop when schema coverage is complete
      2. LogicRAG Decomposition — multi-hop with incomplete schema, or SPARQL failed;
         each sub-question independently uses SPARQL or text retrieval

    Text retrieval is the fallback within sub-questions and a last resort, not a peer route.

    Returns: HybridResult with answer, method, route_source, and sparql_fallback.
    """
    if debug:
        verbose = True
    sep = "=" * 70
    llm_client = llm_client or create_llm_client(provider, api_key=api_key)
    linker = linker or DBpediaEntityLinker(
        min_confidence=0.5, llm_client=llm_client, llm_model=model,
        wiki_resolver=WikipediaResolver(),
    )
    kg_client = kg_client or DBpediaSPARQLClient()
    schema_disc = schema_disc or SchemaDiscovery()
    t_start = time.time()

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
    t_entity_link = time.time()

    # === Step 2: Build EntityProfiles ===
    profiles = [build_entity_profile(e.uri, e.label, schema_disc) for e in entities if e.uri]
    analysis = analyze_question(question, profiles)
    projections = [project(p, analysis) for p in profiles]
    # Render schema for SPARQL gen and LogicRAG (prefixes once at top)
    schema_ctx = (
        "\n\n---\n\n".join(
            p.render_for_sparql(include_prefixes=(i == 0))
            for i, p in enumerate(projections)
        )
        if projections else ""
    )
    if verbose:
        print(f"\n{sep}")
        print("STEP 2: Schema Discovery (EntityProfile + Projection)")
        print(sep)
        for p in projections:
            print(f"  {p.entity_profile.label}: class={p.primary_class}, "
                  f"completeness={p.completeness}")
        if not projections:
            print("  (no URIs to query schema for)")
    t_schema = time.time()

    # === Step 3: Route Decision — rules + ASK verification + LLM fallback ===
    route, route_source = route_question(
        question, entities, kg_client, llm_client, model,
        projections=projections, schema_ctx=schema_ctx, debug=debug,
    )
    t_route = time.time()
    if verbose:
        print(f"\n{sep}")
        print(f"STEP 3: Route Decision → {route} ({route_source})")
        print(sep)

    # Build debug_info — wrapped in try/except so metadata failure doesn't crash the question
    try:
        debug_info = {
            "entities_linked": [
                {"label": e.label, "uri": e.uri, "confidence": e.confidence, "source": e.source}
                for e in entities
            ],
            "route": route,
            "route_meta": routing_mod._last_route_meta.copy(),
            "analysis": {
                "hop_structure": analysis.hop_structure,
                "is_comparison": analysis.is_comparison,
                "target_properties": sorted(analysis.target_properties),
                "bridge_keyword": analysis.bridge_keyword,
                "bridge_property": analysis.bridge_property,
                "hop2_keyword": analysis.hop2_keyword,
                "hop2_properties": sorted(analysis.hop2_properties)
                if analysis.hop2_properties else [],
            },
            "projections": [
                {
                    "entity_label": p.entity_profile.label,
                    "primary_class": p.primary_class,
                    "completeness": p.completeness,
                    "missing_hops": p.missing_hops,
                    "relevant_properties": p.relevant_properties,
                }
                for p in projections
            ],
            "stage_timings": {
                "entity_linking_s": round(t_entity_link - t_start, 3),
                "schema_projection_s": round(t_schema - t_entity_link, 3),
                "routing_s": round(t_route - t_schema, 3),
            },
        }
    except Exception as e:
        debug_info = {"assembly_error": str(e)}

    # === Step 4: Direct SPARQL (single-hop or multi-hop with full schema) ===
    sparql_answer = None
    executed_sparql_query = ""
    if route == "SPARQL" and schema_ctx:
        if verbose:
            print(f"\n{sep}")
            print("STEP 4: SPARQL Generation + Execution")
            print(sep)

        sparql_answer, executed_sparql_query, _discovered = sparql_solver.solve(
            question, schema_ctx, entities, llm_client, model,
            verbose=verbose, debug=debug,
        )

    if sparql_answer:
        debug_info["sparql_solver_meta"] = sparql_solver._last_solve_meta.copy()
        debug_info["stage_timings"]["sparql_solve_s"] = round(time.time() - t_route, 3)
        if verbose:
            print(f"\n{sep}")
            print(f"RESULT (method=sparql, route={route_source})")
            print(sep)
            print(f"  Question: {question}")
            print(f"  Answer:   {sparql_answer}")
        return HybridResult(
            sparql_answer, "sparql", route_source,
            sparql_query=executed_sparql_query,
            debug_info=debug_info,
        )

    # === Step 5: LogicRAG Decomposition (incomplete schema, or SPARQL failed) ===
    sparql_fallback = route == "SPARQL"  # originally routed SPARQL but it failed
    logicrag_answer, sub_q_log = _try_logicrag_decomposition(
        question, entities, schema_ctx, llm_client, model,
        linker, kg_client, schema_disc, verbose, debug=debug,
        benchmark_corpus=benchmark_corpus,
    )
    if logicrag_answer:
        debug_info["decomp_meta"] = logicrag_decomposition._last_decomp_meta.copy()
        debug_info["stage_timings"]["logicrag_s"] = round(time.time() - t_route, 3)
        return HybridResult(
            logicrag_answer, "logicrag_decomposed", route_source,
            sparql_fallback=sparql_fallback,
            sparql_query="[logicrag_decomposed]",
            sub_question_log=sub_q_log,
            debug_info=debug_info,
        )

    # === Step 5b: SPARQL pre-check for direct properties before TEXT ===
    # Safety net: if an entity has a direct property answering the question,
    # use SPARQL instead of TEXT. Catches routing gaps from entity-linking
    # failures where properties weren't detected but the URI is correct.
    if entities and analysis.target_properties and schema_ctx:
        from dbpedia_qa.dbpedia.sparql_utils import escape_sparql_uri

        for entity in entities:
            if not entity.uri:
                continue
            safe_uri = escape_sparql_uri(entity.uri)
            for prop in analysis.target_properties:
                ask = f"ASK {{ <{safe_uri}> {prop} ?x }}"
                try:
                    result = kg_client.query(ask)
                    if result.get("boolean"):
                        if verbose:
                            print(f"\n{sep}")
                            print(f"STEP 5b: SPARQL pre-check hit: "
                                  f"{entity.label} has {prop}")
                            print(sep)
                        precheck_answer, precheck_query, _ = sparql_solver.solve(
                            question, schema_ctx, entities, llm_client, model,
                            verbose=verbose, debug=debug,
                        )
                        if precheck_answer:
                            debug_info["sparql_solver_meta"] = sparql_solver._last_solve_meta.copy()
                            debug_info["stage_timings"]["precheck_s"] = round(
                                time.time() - t_route, 3)
                            return HybridResult(
                                precheck_answer, "sparql",
                                "precheck_" + route_source,
                                sparql_fallback=True,
                                sparql_query=precheck_query,
                                debug_info=debug_info,
                            )
                        break  # Only need one hit to attempt SPARQL
                except Exception:
                    continue  # ASK failed, try next property
            else:
                continue
            break  # Break outer loop if inner broke

    # === Step 6: Text Fallback — type-guided discovery + Wikipedia + LLM ===
    from entity_discovery import discover_context_entities

    if verbose:
        print(f"\n{sep}")
        print("STEP 6a: Context entities — type-guided discovery")
        print(sep)
    all_entities = discover_context_entities(
        entities, projections, kg_client, question=question,
    )
    if verbose:
        print(f"  Seed: {len(entities)}, Discovered: {len(all_entities) - len(entities)}")
        for e in all_entities:
            if e.source == "sparql_neighbor":
                print(f"    + {e.label} ({e.discovery_relation})")

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

    # Build seed titles for budget allocation
    bridge_patterns = ("bridge_comparison", "bridge_hop2", "parent_chain")
    if any(p in route_source for p in bridge_patterns):
        spotlight_uris = {e.uri for e in entities if e.uri}
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in all_entities
            if e.uri and "/resource/" in e.uri and e.uri not in spotlight_uris
        }
    else:
        seed_titles = {
            uri_to_wiki_title(e.uri) for e in entities if e.uri and "/resource/" in e.uri
        }

    if verbose:
        print(f"\n{sep}")
        print("STEP 6b: Wikipedia retrieval + LLM Reasoning")
        print(sep)

    answer = text_solver.solve(
        question, all_entities,
        known_facts=kg_facts,
        relation_hints=relation_hints,
        llm_client=llm_client, model=model,
        verbose=verbose,
        seed_titles=seed_titles,
        benchmark_corpus=benchmark_corpus,
    )

    if not answer:
        answer = "Could not find answer"

    if verbose:
        fb_tag = " [sparql_fallback]" if sparql_fallback else ""
        print(f"\n{sep}")
        print(f"RESULT (method=text, route={route_source}{fb_tag})")
        print(sep)
        print(f"  Question: {question}")
        print(f"  Answer:   {answer}")

    debug_info["text_path"] = {
        "seed_entity_count": len(entities),
        "all_entity_count": len(all_entities),
        "kg_facts_chars": len(kg_facts),
    }
    debug_info["text_solver_meta"] = text_solver._last_solve_meta.copy()
    debug_info["stage_timings"]["text_solve_s"] = round(time.time() - t_route, 3)

    return HybridResult(
        answer, "text", route_source, sparql_fallback,
        kg_context=kg_facts,
        debug_info=debug_info,
    )


def main():
    from sparql_evaluation import force_ipv4
    force_ipv4()

    parser = argparse.ArgumentParser(
        description="Schema-Routed Hybrid QA: SPARQL-first + text fallback"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument(
        "--api-key", default=None,
        help="API key (default: provider-specific env var)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="LLM model (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--provider", default="groq", choices=["groq", "github", "ollama"],
        help="LLM provider (default: groq)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show pipeline steps")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show SPARQL gap analysis and decomposition traces (implies --verbose)",
    )
    parser.add_argument(
        "--corpus", default=None,
        help="Path to benchmark corpus JSON for TEXT route RAG (same as LogicRAG)",
    )
    args = parser.parse_args()

    benchmark_corpus = None
    if args.corpus:
        import json
        with open(args.corpus) as f:
            benchmark_corpus = json.load(f)
        print(f"Loaded benchmark corpus: {len(benchmark_corpus)} entries")

    try:
        result = hybrid_answer(
            args.question, api_key=args.api_key, model=args.model,
            verbose=args.verbose, debug=args.debug, provider=args.provider,
            benchmark_corpus=benchmark_corpus,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if not args.verbose:
        fb = " [sparql_fallback]" if result.sparql_fallback else ""
        print(f"Answer ({result.method}, route={result.route_source}{fb}): {result.answer}")


if __name__ == "__main__":
    main()
