#!/usr/bin/env python3
"""Backbone sensitivity ablation: run Hybrid, LogicRAG, or LinearRAG on 100q with gemma3:12b.

Usage:
    # Run each system separately (can run in parallel in different terminals):
    python scripts/run-backbone-ablation.py hybrid
    python scripts/run-backbone-ablation.py logicrag
    python scripts/run-backbone-ablation.py linearrag

    # Judge all results afterward:
    python scripts/run-backbone-ablation.py judge

    # Quick test with 2 questions:
    python scripts/run-backbone-ablation.py hybrid --limit 2

All systems use gemma3:12b via Ollama (http://localhost:11434/v1).
Output: data/eval_1000q/ablation/ablation_gemma3_{system}_100q.jsonl
"""

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Paths
SAMPLE_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "ablation", "gemma3_100q_sample.json",
)
COVERAGE_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "ablation", "gemma3_100q_coverage.json",
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "eval_1000q", "ablation")
LOGICRAG_DIR = os.path.join(PROJECT_ROOT, "baselines", "LogicRAG")
LOGICRAG_CORPUS = os.path.join(LOGICRAG_DIR, "dataset", "2wikimultihopqa_corpus.json")
LOGICRAG_DATASET = os.path.join(LOGICRAG_DIR, "dataset", "2wikimultihopqa.json")

# Default model/provider for ablation
MODEL = "gemma3:12b"
PROVIDER = "ollama"
QUESTION_TIMEOUT = 600  # 10 min per question (local model is slower)

# Load metrics
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class _QuestionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _QuestionTimeout("question timeout")


def load_sample() -> list[dict]:
    """Load the 100q ablation sample."""
    with open(SAMPLE_FILE) as f:
        return json.load(f)


def load_completed_ids(output_path: str) -> set[str]:
    """Read question IDs already written to output JSONL (for resume)."""
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


def append_record(output_path: str, record: dict):
    """Append one JSONL record to output file."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# Graceful shutdown
_stats = {"done": 0, "total": 0, "em_sum": 0.0, "start": 0.0}


def _signal_handler(sig, frame):
    signal.alarm(0)
    d = _stats["done"]
    em = _stats["em_sum"] / d * 100 if d else 0
    elapsed = time.time() - _stats["start"] if _stats["start"] else 0
    print(f"\n--- Interrupted after {d}/{_stats['total']} questions ---")
    print(f"Running EM={em:.1f}%, Elapsed={elapsed / 60:.1f}min")
    print("Results saved. Resume by running the same command.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Hybrid runner
# ---------------------------------------------------------------------------

def run_hybrid(questions: list[dict], output_path: str, limit: int):
    """Run Hybrid pipeline on questions with gemma3:12b via Ollama."""
    from sparql_evaluation import force_ipv4
    force_ipv4()

    from dbpedia_qa.dbpedia import (
        DBpediaEntityLinker,
        DBpediaSPARQLClient,
        SchemaDiscovery,
        WikipediaResolver,
    )
    from dbpedia_qa.models import extract_final_answer

    from hybrid_qa import hybrid_answer
    from llm_provider import create_llm_client

    # Create shared clients
    llm_client = create_llm_client(PROVIDER)
    linker = DBpediaEntityLinker(
        min_confidence=0.5, llm_client=llm_client, llm_model=MODEL,
        wiki_resolver=WikipediaResolver(),
    )
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()

    # Load benchmark corpus for TEXT route RAG
    with open(LOGICRAG_CORPUS) as f:
        benchmark_corpus = json.load(f)

    done_ids = load_completed_ids(output_path)
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if limit > 0:
        remaining = remaining[:limit]

    if not remaining:
        print("All questions already processed.")
        return

    print(f"Running Hybrid on {len(remaining)} questions (model={MODEL})...\n")
    _stats["total"] = len(remaining)
    _stats["start"] = time.time()

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]

        t0 = time.time()
        record = {
            "question_id": qid,
            "question_type": q["question_type"],
            "question": question,
            "gold_answer": gold,
            "mode": "hybrid_gemma3",
        }

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(QUESTION_TIMEOUT)
        try:
            result = hybrid_answer(
                question, None, model=MODEL,
                llm_client=llm_client, linker=linker,
                kg_client=kg_client, schema_disc=schema_disc,
                benchmark_corpus=benchmark_corpus,
            )
            raw = result.answer or ""
            record["raw_answer"] = raw
            record["predicted_answer"] = extract_final_answer(raw).strip()
            record["method"] = result.method
            record["route_source"] = result.route_source
            record["error"] = ""
        except _QuestionTimeout:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "timeout"
            record["route_source"] = ""
            record["error"] = f"Timeout after {QUESTION_TIMEOUT}s"
        except Exception:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "error"
            record["route_source"] = ""
            record["error"] = traceback.format_exc()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        pred = record["predicted_answer"]
        record["em"] = compute_em(pred, gold) if pred else 0.0
        record["f1"] = compute_f1(pred, gold) if pred else 0.0
        record["containment"] = compute_containment(pred, gold) if pred else 0.0
        record["latency_seconds"] = round(time.time() - t0, 2)

        append_record(output_path, record)
        _stats["done"] = i + 1
        _stats["em_sum"] += record["em"]

        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        print(f"[{i+1}/{len(remaining)}] {question[:55]}...")
        print(f"  -> [{em_tag}] method={record['method']} {record['latency_seconds']}s")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - _stats["start"]
            d = i + 1
            print(f"  --- Running EM={_stats['em_sum']/d*100:.1f}% | "
                  f"Elapsed={elapsed/60:.1f}min ETA={(elapsed/d)*(len(remaining)-d)/60:.1f}min ---")


# ---------------------------------------------------------------------------
# LogicRAG runner
# ---------------------------------------------------------------------------

def run_logicrag(questions: list[dict], output_path: str, limit: int):
    """Run LogicRAG baseline on questions with gemma3:12b via Ollama."""
    # Set env vars BEFORE importing LogicRAG (config reads at import time)
    os.environ["OPENAI_API_KEY"] = "ollama"
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["LLM_MODEL"] = MODEL

    # Import LogicRAGInstrumented
    sys.path.insert(0, LOGICRAG_DIR)
    spec = importlib.util.spec_from_file_location(
        "logicrag_instrumented",
        os.path.join(PROJECT_ROOT, "scripts", "logicrag-instrumented.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    LogicRAGInstrumented = mod.LogicRAGInstrumented

    # Load LogicRAG dataset to validate question IDs
    with open(LOGICRAG_DATASET) as f:
        lr_data = json.load(f)
    lr_by_id = {entry["_id"]: entry for entry in lr_data}

    print("Initializing LogicRAGInstrumented (loading corpus + embeddings)...")
    rag = LogicRAGInstrumented(LOGICRAG_CORPUS)
    rag.set_max_rounds(3)
    rag.set_top_k(3)
    print("Ready.\n")

    done_ids = load_completed_ids(output_path)
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if limit > 0:
        remaining = remaining[:limit]

    if not remaining:
        print("All questions already processed.")
        return

    print(f"Running LogicRAG on {len(remaining)} questions (model={MODEL})...\n")
    _stats["total"] = len(remaining)
    _stats["start"] = time.time()

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]

        record = {
            "question_id": qid,
            "question_type": q["question_type"],
            "question": question,
            "gold_answer": gold,
            "mode": "logicrag_gemma3",
        }

        if qid not in lr_by_id:
            record["predicted_answer"] = ""
            record["method"] = "skipped_not_in_dataset"
            record["em"] = 0.0
            record["f1"] = 0.0
            record["containment"] = 0.0
            record["latency_seconds"] = 0.0
            record["error"] = "not in LogicRAG dataset"
            append_record(output_path, record)
            print(f"[{i+1}/{len(remaining)}] SKIP: {question[:55]}")
            continue

        t0 = time.time()
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(QUESTION_TIMEOUT)
        try:
            answer, _contexts, _rounds = rag.answer_question(question)
            record["raw_answer"] = answer or ""
            record["predicted_answer"] = (answer or "").strip()
            record["method"] = "logicrag_original"
            record["error"] = ""
        except _QuestionTimeout:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "timeout"
            record["error"] = f"Timeout after {QUESTION_TIMEOUT}s"
        except Exception:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "error"
            record["error"] = traceback.format_exc()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        pred = record["predicted_answer"]
        record["em"] = compute_em(pred, gold) if pred else 0.0
        record["f1"] = compute_f1(pred, gold) if pred else 0.0
        record["containment"] = compute_containment(pred, gold) if pred else 0.0
        record["latency_seconds"] = round(time.time() - t0, 2)

        append_record(output_path, record)
        _stats["done"] = i + 1
        _stats["em_sum"] += record["em"]

        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        print(f"[{i+1}/{len(remaining)}] {question[:55]}...")
        print(f"  -> [{em_tag}] {record['latency_seconds']}s")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - _stats["start"]
            d = i + 1
            print(f"  --- Running EM={_stats['em_sum']/d*100:.1f}% | "
                  f"Elapsed={elapsed/60:.1f}min ETA={(elapsed/d)*(len(remaining)-d)/60:.1f}min ---")


# ---------------------------------------------------------------------------
# LinearRAG runner
# ---------------------------------------------------------------------------

def run_linearrag(questions: list[dict], output_path: str, limit: int):
    """Run LinearRAG on 100q with gemma3:12b via Ollama.

    LinearRAG has its own run.py with different dataset format.
    We shell out to its runner and then convert results to our JSONL format.
    """
    # Set env vars for Ollama
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "ollama"
    env["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

    import subprocess

    linearrag_dir = os.path.join(PROJECT_ROOT, "baselines", "LinearRAG")
    dataset_name = "ablation_100q"

    # Check if dataset exists
    dataset_dir = os.path.join(linearrag_dir, "dataset", dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"ERROR: LinearRAG dataset not found at {dataset_dir}")
        print("Run: python scripts/extract-ablation-100q.py first")
        sys.exit(1)

    print(f"Running LinearRAG on {dataset_name} (model={MODEL})...")
    print(f"  Dataset: {dataset_dir}")
    print("  This runs LinearRAG's own pipeline — output goes to baselines/LinearRAG/results/")
    print()

    # Use the system Python (LinearRAG might need its own venv)
    # Try LinearRAG's venv first, fall back to current Python
    lr_python = os.path.join(linearrag_dir, "venv_linearrag", "bin", "python")
    if not os.path.exists(lr_python):
        lr_python = sys.executable
        print(f"  WARNING: Using current Python ({lr_python}), not LinearRAG venv")

    cmd = [
        lr_python, "run.py",
        "--dataset_name", dataset_name,
        "--llm_model", MODEL,
        "--max_workers", "1",  # sequential for local model
        "--spacy_model", "en_core_web_sm",  # lighter model
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  CWD: {linearrag_dir}")
    print()

    result = subprocess.run(
        cmd, cwd=linearrag_dir, env=env,
        capture_output=False,  # stream output
    )

    if result.returncode != 0:
        print(f"\nLinearRAG exited with code {result.returncode}")
        return

    # Find the most recent results directory
    results_dir = os.path.join(linearrag_dir, "results", dataset_name)
    if not os.path.exists(results_dir):
        print(f"No results found at {results_dir}")
        return

    # Get most recent timestamped dir
    subdirs = sorted(os.listdir(results_dir))
    if not subdirs:
        print("No result subdirectories found")
        return

    latest = subdirs[-1]
    predictions_path = os.path.join(results_dir, latest, "predictions.json")
    if not os.path.exists(predictions_path):
        print(f"No predictions.json found at {predictions_path}")
        return

    # Convert LinearRAG predictions to our JSONL format
    print(f"\nConverting {predictions_path} to {output_path}...")
    with open(predictions_path) as f:
        predictions = json.load(f)

    # Build gold answer lookup from our sample
    gold_by_id = {q["question_id"]: q for q in questions}

    # Skip already-converted IDs (prevents duplicate append on re-run)
    done_ids = load_completed_ids(output_path)

    count = 0
    for pred in predictions:
        qid = pred.get("id", "")
        if qid in done_ids:
            continue
        gold_q = gold_by_id.get(qid)
        if not gold_q:
            continue

        gold = gold_q["gold_answer"]
        predicted = (pred.get("prediction", "") or "").strip()

        record = {
            "question_id": qid,
            "question_type": gold_q["question_type"],
            "question": pred.get("question", gold_q["question"]),
            "gold_answer": gold,
            "mode": "linearrag_gemma3",
            "predicted_answer": predicted,
            "method": "linearrag",
            "em": compute_em(predicted, gold) if predicted else 0.0,
            "f1": compute_f1(predicted, gold) if predicted else 0.0,
            "containment": compute_containment(predicted, gold) if predicted else 0.0,
            "error": "",
        }
        append_record(output_path, record)
        count += 1

    print(f"Converted {count} predictions to JSONL")


# ---------------------------------------------------------------------------
# Judge runner
# ---------------------------------------------------------------------------

def run_judge(system: str = "all"):
    """Judge ablation results with gemma3:12b."""
    systems = ["hybrid", "logicrag", "linearrag"] if system == "all" else [system]

    for s in systems:
        input_path = os.path.join(OUTPUT_DIR, f"ablation_gemma3_{s}_100q.jsonl")
        if not os.path.exists(input_path):
            print(f"Skipping {s}: {input_path} not found")
            continue

        print(f"\nJudging {s} results: {input_path}")
        # Use the existing judge script
        import subprocess
        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, "scripts", "judge_hybrid_eval.py"),
            input_path,
            "--provider", "ollama",
            "--model", MODEL,
        ]
        subprocess.run(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _update_model(model: str):
    """Update the module-level MODEL variable."""
    global MODEL
    MODEL = model


def main():
    parser = argparse.ArgumentParser(
        description="Backbone sensitivity ablation: gemma3:12b on 100q",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "system", choices=["hybrid", "logicrag", "linearrag", "judge"],
        help="Which system to run (or 'judge' to judge all results)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Model name (default: {MODEL})",
    )
    args = parser.parse_args()

    # Update module-level MODEL if user overrides
    _update_model(args.model)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.system == "judge":
        run_judge()
        return

    questions = load_sample()
    output_path = os.path.join(OUTPUT_DIR, f"ablation_gemma3_{args.system}_100q.jsonl")

    print(f"=== Backbone Ablation: {args.system} ===")
    print(f"Model: {MODEL}")
    print(f"Questions: {len(questions)}")
    print(f"Output: {output_path}")
    print()

    if args.system == "hybrid":
        run_hybrid(questions, output_path, args.limit)
    elif args.system == "logicrag":
        run_logicrag(questions, output_path, args.limit)
    elif args.system == "linearrag":
        run_linearrag(questions, output_path, args.limit)

    # Final summary
    if os.path.exists(output_path):
        done_ids = load_completed_ids(output_path)
        print(f"\n=== Done: {len(done_ids)} questions in {output_path} ===")


if __name__ == "__main__":
    main()
