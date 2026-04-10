#!/usr/bin/env python3
"""Batch LogicRAG evaluation on HotpotQA and MuSiQue benchmarks.

Adapts run-logicrag-1000q.py for cross-benchmark comparison.
Loads questions from LogicRAG dataset format, runs LogicRAGInstrumented,
outputs JSONL with per-question metrics.

Usage:
    source baselines/LogicRAG/venv_logicrag/bin/activate
    python scripts/run-logicrag-cross-benchmark.py --api-key $GROQ_API_KEY --benchmark hotpotqa --limit 5
    python scripts/run-logicrag-cross-benchmark.py --api-key $GROQ_API_KEY --benchmark musique
"""

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGICRAG_DIR = os.path.join(PROJECT_ROOT, "baselines", "LogicRAG")

# ---------------------------------------------------------------------------
# Load metrics (bypass evaluation/__init__.py heavy imports)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment


# ---------------------------------------------------------------------------
# Benchmark data loaders
# ---------------------------------------------------------------------------
def load_benchmark_questions(benchmark: str) -> list[dict]:
    """Load questions from LogicRAG dataset files, normalized to unified format."""
    path = os.path.join(LOGICRAG_DIR, "dataset", f"{benchmark}.json")
    with open(path) as f:
        raw = json.load(f)

    questions = []
    for entry in raw:
        if benchmark == "hotpotqa":
            questions.append({
                "question_id": entry["_id"],
                "question": entry["question"],
                "gold_answer": entry["answer"],
                "gold_aliases": [],
                "question_type": entry.get("type", "unknown"),
            })
        elif benchmark == "musique":
            # Skip unanswerable (all 1000 are answerable, but defensive)
            if not entry.get("answerable", True):
                continue
            # Infer hop count from ID prefix: "2hop__" -> "2hop"
            qid = entry["id"]
            qtype = qid.split("__")[0] if "__" in qid else "unknown"
            questions.append({
                "question_id": qid,
                "question": entry["question"],
                "gold_answer": entry["answer"],
                "gold_aliases": entry.get("answer_aliases", []),
                "question_type": qtype,
            })
    return questions


def compute_best_metrics(pred: str, gold: str, aliases: list[str]) -> dict:
    """Compute EM/F1/containment, taking max over gold answer + aliases."""
    candidates = [gold] + (aliases or [])
    best_em, best_f1, best_cont = 0.0, 0.0, 0.0
    for c in candidates:
        if not c:
            continue
        best_em = max(best_em, compute_em(pred, c))
        best_f1 = max(best_f1, compute_f1(pred, c))
        best_cont = max(best_cont, compute_containment(pred, c))
    return {"em": best_em, "f1": best_f1, "containment": best_cont}


def load_completed_ids(output_path: str) -> set[str]:
    """Read question IDs already written to JSONL (for resume)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
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
# Timeout + signal handlers
# ---------------------------------------------------------------------------
def _timeout_handler(signum, frame):
    raise TimeoutError("question timeout")


_stats = {"done": 0, "total": 0, "em_sum": 0.0, "f1_sum": 0.0, "start": 0.0}


def _signal_handler(sig, frame):
    signal.alarm(0)
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
        description="Run LogicRAG on HotpotQA/MuSiQue cross-benchmark evaluation"
    )
    parser.add_argument("--api-key", required=True, help="Groq API key")
    parser.add_argument(
        "--benchmark", required=True, choices=["hotpotqa", "musique"],
        help="Benchmark to evaluate on",
    )
    parser.add_argument("--output", default=None, help="Output JSONL path (auto-set if omitted)")
    parser.add_argument("--limit", type=int, default=None, help="Cap questions (for testing)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-question timeout (seconds)")
    parser.add_argument("--max-rounds", type=int, default=3, help="LogicRAG max reasoning rounds")
    parser.add_argument("--top-k", type=int, default=3, help="LogicRAG retrieval top-k")
    args = parser.parse_args()

    # Default output path
    if not args.output:
        args.output = os.path.join(
            PROJECT_ROOT, "data", "cross_benchmark", args.benchmark,
            f"logicrag_{args.benchmark}_1000q.jsonl",
        )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    os.environ["GROQ_API_KEY"] = args.api_key

    # Config summary
    corpus_path = os.path.join(LOGICRAG_DIR, "dataset", f"{args.benchmark}_corpus.json")
    print(f"=== LogicRAG Cross-Benchmark: {args.benchmark} ===")
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d')}")
    print("LLM:        llama-3.3-70b-versatile (Groq)")
    print(f"top_k:      {args.top_k}")
    print(f"max_rounds: {args.max_rounds}")
    print(f"timeout:    {args.timeout}s")
    print(f"corpus:     {corpus_path}")
    print(f"output:     {args.output}")
    print("=" * 50)

    # Load questions
    questions = load_benchmark_questions(args.benchmark)
    print(f"Loaded {len(questions)} questions")

    # Resume support
    done_ids = load_completed_ids(args.output)
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if done_ids:
        print(f"RESUME: {len(done_ids)} done, {len(remaining)} remaining")
    if not remaining:
        print("All questions already processed.")
        return
    if args.limit:
        remaining = remaining[:args.limit]
        print(f"Limited to {len(remaining)} questions")

    # Initialize LogicRAGInstrumented
    print("Initializing LogicRAGInstrumented (loading corpus + embeddings)...")
    sys.path.insert(0, LOGICRAG_DIR)
    spec = importlib.util.spec_from_file_location(
        "logicrag_instrumented",
        os.path.join(PROJECT_ROOT, "scripts", "logicrag-instrumented.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    LogicRAGInstrumented = mod.LogicRAGInstrumented

    # Use per-benchmark cache dir to avoid embedding collisions
    cache_dir = os.path.join(PROJECT_ROOT, "cache", f"logicrag_{args.benchmark}")
    rag = LogicRAGInstrumented(corpus_path, cache_dir=cache_dir)
    rag.set_max_rounds(args.max_rounds)
    rag.set_top_k(args.top_k)
    print("LogicRAGInstrumented ready.\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Evaluation loop
    _stats["total"] = len(remaining)
    _stats["start"] = time.time()
    em_sum, f1_sum = 0.0, 0.0

    print(f"--- {len(remaining)} questions (max_rounds={args.max_rounds}, "
          f"top_k={args.top_k}, timeout={args.timeout}s) ---\n")

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]

        # Run with timeout
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.timeout)
        t0 = time.time()
        try:
            answer, contexts, rounds = rag.answer_question(question)
            signal.alarm(0)
            method = "logicrag"
            trace = rag.last_trace
            token_delta = rag.last_token_delta
        except TimeoutError:
            signal.alarm(0)
            answer, method, trace, token_delta = "", "timeout", None, None
        except Exception as e:
            signal.alarm(0)
            answer, method, trace, token_delta = "", f"error: {e}", None, None
        elapsed = time.time() - t0

        # Compute metrics (alias-aware for MuSiQue)
        pred = (answer or "").strip()
        metrics = compute_best_metrics(pred, gold, q["gold_aliases"]) if pred else {
            "em": 0.0, "f1": 0.0, "containment": 0.0,
        }

        record = {
            "question_id": qid,
            "question": question,
            "question_type": q["question_type"],
            "gold_answer": gold,
            "gold_aliases": q["gold_aliases"],
            "benchmark": args.benchmark,
            "predicted_answer": pred,
            "method": method,
            "rounds": len(trace.get("rounds", [])) if trace else 0,
            "token_usage": token_delta or {"prompt": 0, "completion": 0},
            "em": metrics["em"],
            "f1": metrics["f1"],
            "containment": metrics["containment"],
            "elapsed_sec": round(elapsed, 2),
            "error": None if method == "logicrag" else method,
        }

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        em_sum += record["em"]
        f1_sum += record["f1"]
        done = i + 1
        _stats["done"] = done
        _stats["em_sum"] = em_sum
        _stats["f1_sum"] = f1_sum

        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        print(f"[{done}/{len(remaining)}] {question[:55]}...")
        print(f"  -> gold='{gold}' pred='{pred[:55]}' [{em_tag}] "
              f"rounds={record['rounds']} {record['elapsed_sec']}s")

        if done % 10 == 0:
            elapsed_total = time.time() - _stats["start"]
            eta = (elapsed_total / done) * (len(remaining) - done)
            print(f"  --- Running: EM={em_sum / done * 100:.1f}% "
                  f"F1={f1_sum / done * 100:.1f}% "
                  f"| Elapsed={elapsed_total / 60:.1f}min ETA={eta / 60:.1f}min ---")

    # Final summary
    elapsed_total = time.time() - _stats["start"]
    total = len(remaining)
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {total} questions in {elapsed_total / 60:.1f}min")
    if total > 0:
        print(f"Overall EM={em_sum / total * 100:.1f}%, F1={f1_sum / total * 100:.1f}%")
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()
