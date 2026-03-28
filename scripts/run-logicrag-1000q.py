#!/usr/bin/env python3
"""Batch evaluation runner for LogicRAG baseline on 1000 eval questions.

Runs LogicRAGInstrumented on every question from coverage_analysis_1000_corrected.json,
joining with hop_verification_results.json for answerability ground truth.
Outputs one JSONL record per question with full trace, metrics, and token usage.

Features:
    - SIGALRM per-question timeout (Unix only)
    - Auto-resume from existing JSONL output
    - SIGINT/SIGTERM graceful shutdown with running stats
    - Progress + running EM/F1 every 10 questions

Usage:
    source baselines/LogicRAG/venv_logicrag/bin/activate
    python scripts/run-logicrag-1000q.py --api-key $GROQ_API_KEY --limit 5
    python scripts/run-logicrag-1000q.py --api-key $GROQ_API_KEY  # full run
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

COVERAGE_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "coverage_analysis_1000_corrected.json"
)
HOP_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "hop_verification_results.json"
)
LOGICRAG_DATASET = os.path.join(LOGICRAG_DIR, "dataset", "2wikimultihopqa.json")
LOGICRAG_CORPUS = os.path.join(LOGICRAG_DIR, "dataset", "2wikimultihopqa_corpus.json")
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "logicrag_1000q_traced.jsonl"
)

# ---------------------------------------------------------------------------
# TeeWriter: duplicate stdout/stderr to both terminal and log file
# ---------------------------------------------------------------------------
class TeeWriter:
    """Write to multiple streams simultaneously (terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        # Delegate to first stream (e.g., real stdout) for signal compatibility
        return self.streams[0].fileno()


# ---------------------------------------------------------------------------
# Load our metrics (bypass evaluation/__init__.py heavy imports)
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
# Data loading helpers
# ---------------------------------------------------------------------------
def load_questions(path: str) -> list[dict]:
    """Load questions from coverage_analysis JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


def build_hop_map(path: str) -> dict[str, bool]:
    """Build {question_id: chain_connected} from hop verification results."""
    with open(path) as f:
        data = json.load(f)
    return {r["question_id"]: r["chain_connected"] for r in data["results"]}


def load_completed_ids(output_path: str) -> set[str]:
    """Read question IDs already written to the output JSONL (for resume)."""
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
# Timeout handler (SIGALRM — Unix only)
# ---------------------------------------------------------------------------
def _timeout_handler(signum, frame):
    raise TimeoutError("question timeout")


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------
def run_question(rag, question: str, timeout_sec: int) -> tuple[str, dict | None, str]:
    """Run a single question with SIGALRM timeout.

    Returns (answer, trace_or_None, method_string).
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        answer, contexts, rounds = rag.answer_question(question)
        signal.alarm(0)  # cancel alarm
        return answer, rag.last_trace, "logicrag_original"
    except TimeoutError:
        signal.alarm(0)
        return "", None, "timeout"
    except Exception as e:
        signal.alarm(0)
        return "", None, f"error: {e}"


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------
def build_record(q: dict, hop_map: dict, answer: str, trace: dict | None,
                 method: str, token_delta: dict | None, elapsed: float) -> dict:
    """Assemble one JSONL record from question metadata + run results."""
    pred = (answer or "").strip()
    gold = q.get("gold_answer", "")

    # Metrics — 0 for empty predictions
    em = compute_em(pred, gold) if pred else 0.0
    f1 = compute_f1(pred, gold) if pred else 0.0
    containment = compute_containment(pred, gold) if pred else 0.0

    # Round count from trace
    if trace and trace.get("early_exit_warmup"):
        rounds = 0
    elif trace:
        rounds = len(trace.get("rounds", []))
    else:
        rounds = 0

    return {
        "question_id": q["question_id"],
        "question": q["question"],
        "question_type": q["question_type"],
        "gold_answer": gold,
        "is_answerable": q["is_answerable"],
        "hop_verified_answerable": hop_map.get(q["question_id"], False),
        "mode": "logicrag_baseline",
        "predicted_answer": pred,
        "method": method,
        "rounds": rounds,
        "early_exit_warmup": trace["early_exit_warmup"] if trace else None,
        "trace": trace,
        "token_usage": token_delta or {"prompt": 0, "completion": 0},
        "em": em,
        "f1": f1,
        "containment": containment,
        "elapsed_sec": round(elapsed, 2),
        "error": None if method == "logicrag_original" else method,
    }


# ---------------------------------------------------------------------------
# Running stats (shared state for signal handler)
# ---------------------------------------------------------------------------
_stats = {"done": 0, "total": 0, "em_sum": 0.0, "f1_sum": 0.0, "start": 0.0}


def _signal_handler(sig, frame):
    """Print running stats on SIGINT/SIGTERM and exit cleanly."""
    signal.alarm(0)  # cancel any pending SIGALRM to avoid race during shutdown
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
        description="Run LogicRAG baseline on 1000 eval questions with full tracing"
    )
    parser.add_argument("--api-key", required=True, help="Groq API key")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output JSONL path (default: data/eval_1000q/logicrag_1000q_traced.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap remaining questions (for testing)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-question SIGALRM timeout in seconds (default: 300)")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="LogicRAG max reasoning rounds (default: 3, matches paper)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="LogicRAG retrieval top-k (default: 3, paper-optimal)")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for log files (default: logs/)")
    args = parser.parse_args()

    # Register graceful shutdown handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Set API key before importing LogicRAG (config.py reads it at import)
    os.environ["GROQ_API_KEY"] = args.api_key

    # ---------------------------------------------------------------
    # Set up file logging (TeeWriter duplicates stdout+stderr to file)
    # ---------------------------------------------------------------
    today = datetime.now().strftime("%y%m%d")
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(PROJECT_ROOT, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"logicrag_1000q_{today}_topk{args.top_k}_rounds{args.max_rounds}.log"
    log_path = os.path.join(log_dir, log_filename)
    _log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    sys.stdout = TeeWriter(sys.__stdout__, _log_file)
    sys.stderr = TeeWriter(sys.__stderr__, _log_file)

    # ---------------------------------------------------------------
    # Config summary
    # ---------------------------------------------------------------
    corpus_name = os.path.basename(LOGICRAG_CORPUS)
    output_name = args.output
    print("=== LogicRAG 1000q Experiment Config ===")
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d')}")
    print("LLM:        llama-3.3-70b-versatile (Groq)")
    print("Embedding:  all-MiniLM-L6-v2")
    print(f"top_k:      {args.top_k}")
    print(f"max_rounds: {args.max_rounds}")
    print(f"timeout:    {args.timeout}s")
    print("max_tokens: 250")
    print(f"corpus:     {corpus_name}")
    print(f"output:     {output_name}")
    print(f"log:        {log_path}")
    print("=" * 40)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    print("Loading questions...")
    questions = load_questions(COVERAGE_FILE)
    hop_map = build_hop_map(HOP_FILE)
    print(f"  {len(questions)} questions, {sum(hop_map.values())} hop-verified answerable")

    # Load LogicRAG dataset for ID validation
    with open(LOGICRAG_DATASET) as f:
        lr_data = json.load(f)
    lr_by_id = {entry["_id"]: entry for entry in lr_data}

    missing_ids = {q["question_id"] for q in questions} - set(lr_by_id.keys())
    if missing_ids:
        print(f"  WARNING: {len(missing_ids)} questions not in LogicRAG dataset (will skip)")

    # Resume support
    done_ids = load_completed_ids(args.output)
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if done_ids:
        print(f"  RESUME: {len(done_ids)} done, {len(remaining)} remaining")
    if not remaining:
        print("All questions already processed.")
        return

    if args.limit:
        remaining = remaining[:args.limit]
        print(f"  Limited to {len(remaining)} questions")

    # ---------------------------------------------------------------
    # Initialize LogicRAG (must happen after setting GROQ_API_KEY)
    # ---------------------------------------------------------------
    print("Initializing LogicRAGInstrumented (loading corpus + embeddings)...")
    sys.path.insert(0, LOGICRAG_DIR)
    # Import our instrumented subclass (scripts/ is already on path via PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    # Python resolves hyphenated filenames via importlib
    spec = importlib.util.spec_from_file_location(
        "logicrag_instrumented",
        os.path.join(PROJECT_ROOT, "scripts", "logicrag-instrumented.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    LogicRAGInstrumented = mod.LogicRAGInstrumented

    rag = LogicRAGInstrumented(LOGICRAG_CORPUS)
    rag.set_max_rounds(args.max_rounds)
    rag.set_top_k(args.top_k)
    print("LogicRAGInstrumented ready.\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------
    _stats["total"] = len(remaining)
    _stats["start"] = time.time()
    em_sum, f1_sum = 0.0, 0.0

    print(f"--- LogicRAG 1000q eval: {len(remaining)} questions "
          f"(max_rounds={args.max_rounds}, top_k={args.top_k}, "
          f"timeout={args.timeout}s) ---\n")

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]

        # Skip questions not in LogicRAG dataset
        if qid not in lr_by_id:
            record = build_record(
                q, hop_map, "", None,
                "skipped_not_in_logicrag_dataset", None, 0.0,
            )
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done = i + 1
            _stats["done"] = done
            print(f"[{done}/{len(remaining)}] SKIP (not in LR dataset): {question[:60]}")
            continue

        # Run the question
        t0 = time.time()
        answer, trace, method = run_question(rag, question, args.timeout)
        elapsed = time.time() - t0

        # Token delta from the instrumented class
        token_delta = rag.last_token_delta if method == "logicrag_original" else None

        record = build_record(q, hop_map, answer, trace, method, token_delta, elapsed)

        # Append to JSONL (flush ensures record is persisted before next question)
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        # Update running stats
        em_sum += record["em"]
        f1_sum += record["f1"]
        done = i + 1
        _stats["done"] = done
        _stats["em_sum"] = em_sum
        _stats["f1_sum"] = f1_sum

        # Per-question output
        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        print(
            f"[{done}/{len(remaining)}] {question[:55]}..."
        )
        print(
            f"  -> gold='{gold}' pred='{record['predicted_answer'][:55]}' "
            f"[{em_tag}] rounds={record['rounds']} {record['elapsed_sec']}s"
        )

        # Running summary every 10 questions
        if done % 10 == 0:
            elapsed_total = time.time() - _stats["start"]
            eta = (elapsed_total / done) * (len(remaining) - done)
            print(
                f"  --- Running: EM={em_sum / done * 100:.1f}% "
                f"F1={f1_sum / done * 100:.1f}% "
                f"| Elapsed={elapsed_total / 60:.1f}min ETA={eta / 60:.1f}min ---"
            )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    elapsed_total = time.time() - _stats["start"]
    total = len(remaining)
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {total} questions in {elapsed_total / 60:.1f}min")
    if total > 0:
        print(f"Overall EM={em_sum / total * 100:.1f}%, F1={f1_sum / total * 100:.1f}%")
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()
