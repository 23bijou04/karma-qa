#!/usr/bin/env python3
"""Schema-Routed Hybrid QA: uses KG schema to route SPARQL vs text.

Orchestrator-only: delegates SPARQL solving to sparql_solver.py,
text solving to text_solver.py, routing to routing.py,
and multi-hop decomposition to logicrag_decomposition.py.

Three-path architecture:
  1. EntityLink(q)            — DBpedia Spotlight (spaCy fallback)
  2. SchemaDiscovery(E)       — entity classes, properties, sample values
  3. RouteDecision(q, S)      — rule-based + SPARQL ASK verification + LLM fallback
  4. Direct SPARQL            — sparql_solver.solve() (single-hop or multi-hop with full schema)
  5. LogicRAG decomposition   — logicrag_decomposition (incomplete schema or SPARQL failed)
  6. Text fallback            — text_solver.solve()

Usage:
  python3 hybrid_qa.py "Who directed Inception?" --verbose
  python3 hybrid_qa.py "Who destroys the last Horcrux?" --verbose
  python3 hybrid_qa.py "When was Albert Einstein born?" --api-key $GROQ_API_KEY
"""

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import unquote

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import sparql_solver  # noqa: E402
import text_solver  # noqa: E402
from ablation_study_v3.wiki_utils import uri_to_wiki_title  # noqa: E402
from dbpedia_qa.dbpedia import (  # noqa: E402
    DBpediaEntityLinker,
    DBpediaSPARQLClient,
    SchemaDiscovery,
)
from dbpedia_qa.models import Entity, HybridResult  # noqa: E402
from entity_profile import build_entity_profile  # noqa: E402
from graphrag_wiki import link_entities_to_dbpedia  # noqa: E402
from llm_provider import create_llm_client  # noqa: E402
from logicrag_decomposition import _try_logicrag_decomposition  # noqa: E402
from question_analysis import analyze_question  # noqa: E402
from routing import route_question  # noqa: E402
from schema_projection import project  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def hybrid_answer(question: str, api_key: str = None,
                  model: str = "llama-3.3-70b-versatile",
                  verbose: bool = False, debug: bool = False,
                  llm_client=None, linker=None,
                  kg_client=None, schema_disc=None,
                  provider: str = "groq") -> HybridResult:
    """Schema-routed hybrid QA with 3-path architecture.

    Paths:
      1. Direct SPARQL — single-hop or multi-hop when schema coverage is complete
      2. LogicRAG decomposition — multi-hop with incomplete schema, or direct SPARQL failed
      3. Simple text fallback — neighbors + Wikipedia + LLM reasoning

    Returns: HybridResult with answer, method, route_source, and sparql_fallback.
    """
    if debug:
        verbose = True
    sep = "=" * 70
    llm_client = llm_client or create_llm_client(provider, api_key=api_key)
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

    # === Step 3: Route Decision — rules + ASK verification + LLM fallback ===
    route, route_source = route_question(
        question, entities, kg_client, llm_client, model,
        projections=projections, schema_ctx=schema_ctx, debug=debug,
    )
    if verbose:
        print(f"\n{sep}")
        print(f"STEP 3: Route Decision → {route} ({route_source})")
        print(sep)

    # === Step 4: Direct SPARQL (single-hop or multi-hop with full schema) ===
    sparql_answer = None
    executed_sparql_query = ""
    if route == "SPARQL" and schema_ctx:
        if verbose:
            print(f"\n{sep}")
            print("STEP 4: SPARQL Generation + Execution")
            print(sep)

        sparql_answer, executed_sparql_query, _ = sparql_solver.solve(
            question, schema_ctx, entities, llm_client, model,
            verbose=verbose, debug=debug,
        )

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

    # === Step 5: LogicRAG Decomposition (incomplete schema, or SPARQL failed) ===
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

    return HybridResult(
        answer, "text", route_source, sparql_fallback,
        kg_context=kg_facts,
    )


def main():
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
        "--provider", default="groq", choices=["groq", "github"],
        help="LLM provider (default: groq)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show pipeline steps")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show SPARQL gap analysis and decomposition traces (implies --verbose)",
    )
    args = parser.parse_args()

    try:
        result = hybrid_answer(
            args.question, api_key=args.api_key, model=args.model,
            verbose=args.verbose, debug=args.debug, provider=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if not args.verbose:
        fb = " [sparql_fallback]" if result.sparql_fallback else ""
        print(f"Answer ({result.method}, route={result.route_source}{fb}): {result.answer}")


if __name__ == "__main__":
    main()
