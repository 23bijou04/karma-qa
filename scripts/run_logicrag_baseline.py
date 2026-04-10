#!/usr/bin/env python3
"""Run original LogicRAG baseline on the same 60 scenario-eval questions.

Selects the same 10/scenario stratified sample (seed=42) used by scenario_eval.py,
runs the original LogicRAG code (baselines/LogicRAG/) with Groq backend,
and outputs JSONL in our standard format with EM/F1/containment metrics.

Usage:
    python scripts/run_logicrag_baseline.py --api-key $GROQ_API_KEY
    python scripts/run_logicrag_baseline.py --api-key $GROQ_API_KEY --per-scenario 5  # quick test
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
LOGICRAG_DIR = os.path.join(PROJECT_ROOT, "baselines", "LogicRAG")

# Load our metrics directly (bypass evaluation/__init__.py heavy imports)
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment

DEFAULT_SCENARIO_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "scenario_classification_1000.json"
)
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "scenario_eval_logicrag_baseline.jsonl"
)
LOGICRAG_DATASET = os.path.join(LOGICRAG_DIR, "dataset", "2wikimultihopqa.json")
LOGICRAG_CORPUS = os.path.join(LOGICRAG_DIR, "dataset", "2wikimultihopqa_corpus.json")


# ---------------------------------------------------------------------------
# Question selection (mirrors scenario_eval.py exactly)
# ---------------------------------------------------------------------------

def select_questions(data: dict, per_scenario: int, seed: int = 42) -> list[dict]:
    """Stratified sample: N questions per scenario (A-F), sorted by scenario."""
    questions = data["questions"]
    by_scenario = defaultdict(list)
    for q in questions:
        by_scenario[q["scenario"]].append(q)

    rng = random.Random(seed)
    selected = []
    for scenario in sorted(by_scenario.keys()):
        pool = by_scenario[scenario]
        n = min(per_scenario, len(pool))
        selected.extend(rng.sample(pool, n))
    return selected


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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run original LogicRAG baseline on scenario eval questions"
    )
    parser.add_argument("--api-key", required=True, help="Groq API key")
    parser.add_argument("--per-scenario", type=int, default=10,
                        help="Questions per scenario (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between questions (default: 2)")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="LogicRAG max reasoning rounds (default: 5)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="LogicRAG retrieval top-k (default: 5)")
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Set Groq API key for LogicRAG config
    os.environ["GROQ_API_KEY"] = args.api_key

    # Load scenario questions and build ID→question map
    with open(args.scenario_file) as f:
        scenario_data = json.load(f)
    selected = select_questions(scenario_data, args.per_scenario, args.seed)
    selected_ids = {q["question_id"] for q in selected}
    print(f"Selected {len(selected)} questions ({args.per_scenario}/scenario)")

    # Load LogicRAG dataset and build _id→entry map
    with open(LOGICRAG_DATASET) as f:
        lr_data = json.load(f)
    lr_by_id = {entry["_id"]: entry for entry in lr_data}

    # Verify overlap
    missing = selected_ids - set(lr_by_id.keys())
    if missing:
        print(f"WARNING: {len(missing)} selected questions not in LogicRAG dataset")

    # Resume support
    done_ids = load_completed_ids(args.output)
    remaining = [q for q in selected if q["question_id"] not in done_ids]
    if done_ids:
        print(f"RESUME: {len(done_ids)} done, {len(remaining)} remaining")
    if not remaining:
        print("All questions already processed.")
        return

    # Initialize LogicRAG (must be done after setting env var)
    print("Initializing LogicRAG model (loading corpus + embeddings)...")
    sys.path.insert(0, LOGICRAG_DIR)
    from src.models.logic_rag import LogicRAG  # noqa: E402

    rag = LogicRAG(LOGICRAG_CORPUS)
    rag.set_max_rounds(args.max_rounds)
    rag.set_top_k(args.top_k)
    print("LogicRAG ready.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    _stats["total"] = len(remaining)
    _stats["start"] = time.time()
    em_sum, f1_sum = 0.0, 0.0
    by_scenario = defaultdict(lambda: {"count": 0, "em": 0.0, "f1": 0.0})

    print(f"\n--- LogicRAG baseline: {len(remaining)} questions "
          f"(max_rounds={args.max_rounds}, top_k={args.top_k}, delay={args.delay}s) ---\n")

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
            "mode": "logicrag_baseline",
        }

        try:
            answer, contexts, rounds = rag.answer_question(question)
            record["raw_answer"] = answer or ""
            record["predicted_answer"] = (answer or "").strip()
            record["method"] = "logicrag_original"
            record["rounds"] = rounds
            record["error"] = None
        except Exception as e:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "error"
            record["rounds"] = 0
            record["error"] = str(e)

        # Metrics
        pred = record["predicted_answer"]
        raw = record["raw_answer"]
        record["em"] = compute_em(pred, gold) if pred else 0.0
        record["f1"] = compute_f1(pred, gold) if pred else 0.0
        record["containment"] = compute_containment(raw, gold) if raw else 0.0
        record["elapsed_sec"] = round(time.time() - t0, 2)

        # Append JSONL
        with open(args.output, "a", encoding="utf-8") as f:
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

        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        print(
            f"[{done}/{len(remaining)}] Scenario {scenario} | "
            f"{question[:55]}..."
        )
        print(
            f"  -> gold='{gold}' pred='{pred[:55]}' [{em_tag}] "
            f"rounds={record['rounds']} {record['elapsed_sec']}s"
        )

        if done % 10 == 0:
            elapsed = time.time() - _stats["start"]
            eta = (elapsed / done) * (len(remaining) - done)
            print(
                f"  --- Running: EM={em_sum / done * 100:.1f}% "
                f"F1={f1_sum / done * 100:.1f}% "
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
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()
