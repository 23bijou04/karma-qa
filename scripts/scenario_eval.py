#!/usr/bin/env python3
"""Scenario-stratified evaluation: Hybrid vs LogicRAG-forced.

Runs hybrid_answer() or a LogicRAG-forced wrapper on ~N questions per scenario
(A-F) from the scenario classification dataset. JSONL append-only with resume.

Usage:
    python scripts/scenario_eval.py --mode hybrid --per-scenario 10 --delay 2
    python scripts/scenario_eval.py --mode logicrag --per-scenario 10 --delay 2
    python scripts/scenario_eval.py --mode hybrid --per-scenario 1   # pilot
"""

import argparse
import importlib.util
import json
import os
import random
import signal
import sys
import time
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from dbpedia_qa.dbpedia import DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery
from dbpedia_qa.models import extract_final_answer

from llm_provider import create_llm_client

# Load metrics directly (bypass evaluation/__init__.py heavy imports)
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment

import sparql_solver  # noqa: E402
import text_solver  # noqa: E402
from ablation_study_v3.wiki_utils import uri_to_wiki_title  # noqa: E402

# LogicRAG + text fallback imports
from entity_discovery import discover_context_entities  # noqa: E402
from entity_profile import build_entity_profile  # noqa: E402
from graphrag_wiki import link_entities_to_dbpedia  # noqa: E402
from hybrid_qa import hybrid_answer  # noqa: E402
from logicrag_decomposition import _try_logicrag_decomposition  # noqa: E402
from question_analysis import analyze_question  # noqa: E402
from schema_projection import project  # noqa: E402

DEFAULT_SCENARIO_FILE = "data/eval_1000q/ground-truth/scenario_classification_1000.json"
LOGICRAG_CORPUS = os.path.join(
    PROJECT_ROOT, "baselines", "LogicRAG", "dataset", "2wikimultihopqa_corpus.json"
)


def load_benchmark_corpus(corpus_file: str) -> list[dict]:
    """Load benchmark corpus for fair TEXT-route comparison."""
    with open(corpus_file) as f:
        corpus = json.load(f)
    print(f"Loaded benchmark corpus: {len(corpus)} entries from {corpus_file}")
    return corpus


# ---------------------------------------------------------------------------
# Question selection
# ---------------------------------------------------------------------------

def select_questions(
    data: dict, per_scenario: int, seed: int = 42, scenarios: list[str] | None = None,
) -> list[dict]:
    """Stratified sample: N questions per scenario (A-F), sorted by scenario.

    If *scenarios* is given, only include those scenarios (e.g. ["D"]).
    """
    questions = data["questions"]
    by_scenario = defaultdict(list)
    for q in questions:
        by_scenario[q["scenario"]].append(q)

    if scenarios:
        unknown = set(scenarios) - set(by_scenario.keys())
        if unknown:
            valid = sorted(by_scenario.keys())
            raise ValueError(f"Unknown scenarios: {unknown}. Valid: {valid}")

    rng = random.Random(seed)
    selected = []
    allowed = set(scenarios) if scenarios else set(by_scenario.keys())
    for scenario in sorted(by_scenario.keys()):
        pool = by_scenario[scenario]
        n = min(per_scenario, len(pool))
        sampled = rng.sample(pool, n)  # always consume RNG for seed stability
        if scenario in allowed:
            selected.extend(sampled)

    return selected


# ---------------------------------------------------------------------------
# Answer cleaning (shared with hybrid_batch_eval.py)
# ---------------------------------------------------------------------------

def clean_answer(raw: str) -> str:
    """Normalize answer from both SPARQL and text routes."""
    if not raw:
        return ""
    answer = extract_final_answer(raw)
    for prefix in ["The answer is ", "Answer: ", "Based on the text, "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    return answer.strip().rstrip(".")


# ---------------------------------------------------------------------------
# LogicRAG-forced wrapper
# ---------------------------------------------------------------------------

def logicrag_forced_answer(
    question: str,
    entities,
    llm_client,
    model: str,
    linker,
    kg_client,
    schema_disc,
    verbose: bool = False,
    benchmark_corpus: list[dict] | None = None,
) -> tuple[str, str]:
    """Force LogicRAG decomposition path, with SPARQL + text fallback.

    Returns (answer_string, method_label).
    """
    # Build schema context (same as hybrid_qa.py steps 2-3)
    profiles = [
        build_entity_profile(e.uri, e.label, schema_disc) for e in entities if e.uri
    ]
    analysis = analyze_question(question, profiles)
    projections = [project(p, analysis) for p in profiles]
    schema_ctx = (
        "\n\n---\n\n".join(
            p.render_for_sparql(include_prefixes=(i == 0))
            for i, p in enumerate(projections)
        )
        if projections else ""
    )

    # Try LogicRAG decomposition
    logicrag_answer, _sub_q_log = _try_logicrag_decomposition(
        question=question,
        entities=entities,
        schema_ctx=schema_ctx,
        llm_client=llm_client,
        model=model,
        linker=linker,
        kg_client=kg_client,
        schema_disc=schema_disc,
        verbose=verbose,
    )
    if logicrag_answer:
        return logicrag_answer, "logicrag_decomposed"

    # LogicRAG returned None (single-hop) — try SPARQL first
    if schema_ctx:
        sparql_answer, _, _ = sparql_solver.solve(
            question, schema_ctx, entities, llm_client, model,
            verbose=verbose,
        )
        if sparql_answer:
            return sparql_answer, "logicrag_sparql_fallback"

    # Text fallback with full entity discovery context
    all_entities = entities
    if projections:
        all_entities = discover_context_entities(
            entities, projections, kg_client, question=question,
        )

    # Build KG facts from neighbor relations
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
            relation_hints[uri_to_wiki_title(e.uri)] = e.discovery_relation

    # Build seed titles
    seed_titles = {
        uri_to_wiki_title(e.uri) for e in entities if e.uri and "/resource/" in e.uri
    }

    answer = text_solver.solve(
        question, all_entities,
        known_facts=kg_facts,
        relation_hints=relation_hints,
        llm_client=llm_client, model=model,
        verbose=verbose,
        seed_titles=seed_titles,
        benchmark_corpus=benchmark_corpus,
    )

    if answer:
        return answer, "logicrag_text_fallback"
    return "Could not find answer", "logicrag_text_fallback"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ids(results_file: str) -> set[str]:
    """Load question IDs already processed (for resume)."""
    done = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("question_id"):
                            done.add(rec["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------

_stats = {"done": 0, "total": 0, "em_sum": 0.0, "f1_sum": 0.0, "start": 0.0}


def _signal_handler(sig, frame):
    elapsed = time.time() - _stats["start"] if _stats["start"] else 0
    d = _stats["done"]
    em = _stats["em_sum"] / d * 100 if d else 0
    f1 = _stats["f1_sum"] / d * 100 if d else 0
    print(f"\n--- Interrupted after {d}/{_stats['total']} questions ---")
    print(f"Running EM={em:.1f}%, F1={f1:.1f}%, Elapsed={elapsed / 60:.1f}min")
    print("Results saved. Resume by running the same command.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    from sparql_evaluation import force_ipv4
    force_ipv4()

    parser = argparse.ArgumentParser(description="Scenario evaluation: Hybrid vs LogicRAG-forced")
    parser.add_argument("--mode", required=True, choices=["hybrid", "logicrag"],
                        help="Pipeline mode")
    parser.add_argument("--per-scenario", type=int, default=10,
                        help="Questions per scenario (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between questions (default: 2)")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--provider", default="groq", choices=["groq", "github"])
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: data/eval_1000q/pilot-100q/scenario_eval_{mode}.jsonl)")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenarios to include (e.g. D or A,B,D; default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    output_file = args.output or f"data/eval_1000q/pilot-100q/scenario_eval_{args.mode}.jsonl"

    # Load and select questions
    with open(args.scenario_file) as f:
        data = json.load(f)
    scenario_filter = [s.strip().upper() for s in args.scenarios.split(",")] if args.scenarios else None
    selected = select_questions(data, args.per_scenario, args.seed, scenarios=scenario_filter)
    print(f"Selected {len(selected)} questions ({args.per_scenario}/scenario) "
          f"from {len(data['questions'])} total")

    # Resume
    done_ids = load_completed_ids(output_file)
    remaining = [q for q in selected if q["question_id"] not in done_ids]
    if done_ids:
        print(f"RESUME: {len(done_ids)} done, {len(remaining)} remaining")
    if not remaining:
        print("All questions already processed.")
        return

    # Create shared clients
    print("Creating shared clients...")
    try:
        llm_client = create_llm_client(args.provider, api_key=args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    linker = DBpediaEntityLinker(min_confidence=0.5)
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()
    benchmark_corpus = load_benchmark_corpus(LOGICRAG_CORPUS)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    _stats["total"] = len(remaining)
    _stats["start"] = time.time()
    em_sum, f1_sum = 0.0, 0.0
    by_scenario = defaultdict(lambda: {"count": 0, "em": 0.0, "f1": 0.0})

    print(f"\n--- {args.mode.upper()} mode: {len(remaining)} questions (delay={args.delay}s) ---\n")

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]
        scenario = q["scenario"]

        t0 = time.time()
        record = {
            "question_id": qid,
            "question": question,
            "question_type": q["question_type"],
            "scenario": scenario,
            "scenario_label": q.get("scenario_label", ""),
            "gold_answer": gold,
            "mode": args.mode,
        }

        try:
            if args.mode == "hybrid":
                result = hybrid_answer(
                    question, args.api_key, model=args.model,
                    llm_client=llm_client, linker=linker,
                    kg_client=kg_client, schema_disc=schema_disc,
                    verbose=args.verbose,
                    benchmark_corpus=benchmark_corpus,
                )
                record["raw_answer"] = result.answer
                record["predicted_answer"] = clean_answer(result.answer)
                record["method"] = result.method
                record["route_source"] = result.route_source
                record["sparql_fallback"] = result.sparql_fallback
                if result.sub_question_log:
                    record["sub_question_log"] = result.sub_question_log
            else:
                # LogicRAG-forced
                entities = link_entities_to_dbpedia(question, linker)
                answer, method = logicrag_forced_answer(
                    question, entities, llm_client, args.model,
                    linker, kg_client, schema_disc, verbose=args.verbose,
                    benchmark_corpus=benchmark_corpus,
                )
                record["raw_answer"] = answer
                record["predicted_answer"] = clean_answer(answer)
                record["method"] = method
            record["error"] = None
        except Exception as e:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "error"
            record["error"] = str(e)

        # Metrics
        pred = record["predicted_answer"]
        raw = record["raw_answer"]
        record["em"] = compute_em(pred, gold) if pred else 0.0
        record["f1"] = compute_f1(pred, gold) if pred else 0.0
        record["containment"] = compute_containment(raw, gold) if raw else 0.0
        record["elapsed_sec"] = round(time.time() - t0, 2)

        # Append JSONL
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Running stats
        em_sum += record["em"]
        f1_sum += record["f1"]
        done = i + 1
        _stats["done"] = done
        _stats["em_sum"] = em_sum
        _stats["f1_sum"] = f1_sum

        by_scenario[scenario]["count"] += 1
        by_scenario[scenario]["em"] += record["em"]
        by_scenario[scenario]["f1"] += record["f1"]

        # Progress
        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        print(
            f"[{done}/{len(remaining)}] Scenario {scenario} | "
            f"{question[:55]}..."
        )
        print(
            f"  -> gold='{gold}' pred='{pred[:55]}' [{em_tag}] "
            f"method={record['method']} {record['elapsed_sec']}s"
        )

        if done % 10 == 0:
            elapsed = time.time() - _stats["start"]
            eta = (elapsed / done) * (len(remaining) - done)
            print(
                f"  --- Running: EM={em_sum / done * 100:.1f}% F1={f1_sum / done * 100:.1f}% "
                f"| Elapsed={elapsed / 60:.1f}min ETA={eta / 60:.1f}min ---"
            )

        if i < len(remaining) - 1:
            time.sleep(args.delay)

    # Final summary
    elapsed = time.time() - _stats["start"]
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(remaining)} questions in {elapsed / 60:.1f}min")
    print(f"Overall EM={em_sum / len(remaining) * 100:.1f}%, "
          f"F1={f1_sum / len(remaining) * 100:.1f}%")
    print("\nBy scenario:")
    for sc in sorted(by_scenario.keys()):
        s = by_scenario[sc]
        c = s["count"]
        print(f"  {sc}: n={c}, EM={s['em'] / c * 100:.1f}%, F1={s['f1'] / c * 100:.1f}%")
    print(f"\nResults: {output_file}")


if __name__ == "__main__":
    main()
