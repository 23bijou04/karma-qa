#!/usr/bin/env python3
"""Batch evaluation of Hybrid QA on all 1,000 2WikiMultiHopQA questions.

Runs hybrid_qa.hybrid_answer() on every question, recording route taken
(sparql / text), timing, and metrics.
JSONL append-only output with resume capability.

Usage:
    python scripts/hybrid_batch_eval.py --api-key $GROQ_API_KEY --limit 5   # pilot
    python scripts/hybrid_batch_eval.py --api-key $GROQ_API_KEY             # full run
"""

import argparse
import importlib.util
import json
import os
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
from groq import Groq

# Load metrics directly to avoid evaluation/__init__.py heavy imports
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment

# Import hybrid_answer (must come after sys.path setup)
from hybrid_qa import hybrid_answer  # noqa: E402

DEFAULT_COVERAGE = "data/eval_1000q/coverage_analysis_1000.json"
DEFAULT_OUTPUT = "data/eval_1000q/hybrid_eval_1000.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_questions(coverage_file: str) -> list[dict]:
    """Load ALL questions from coverage analysis (not just answerable)."""
    with open(coverage_file) as f:
        data = json.load(f)
    questions = data["questions"]
    answerable = sum(1 for q in questions if q["is_answerable"])
    print(f"Loaded {len(questions)} questions ({answerable} answerable) from {coverage_file}")
    return questions


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
# Answer cleaning
# ---------------------------------------------------------------------------

def clean_answer(raw: str) -> str:
    """Normalize answer from both SPARQL and text routes."""
    if not raw:
        return ""
    answer = extract_final_answer(raw)
    # Strip common LLM preambles
    for prefix in ["The answer is ", "Answer: ", "Based on the text, "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    return answer.strip().rstrip(".")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Global state for signal handler
_stats = {"done": 0, "total": 0, "em_sum": 0.0, "f1_sum": 0.0, "start": 0.0}


def _signal_handler(sig, frame):
    elapsed = time.time() - _stats["start"] if _stats["start"] else 0
    d = _stats["done"]
    em = _stats["em_sum"] / d * 100 if d else 0
    f1 = _stats["f1_sum"] / d * 100 if d else 0
    print(f"\n--- Interrupted after {d}/{_stats['total']} questions ---")
    print(f"Running EM={em:.1f}%, F1={f1:.1f}%, Elapsed={elapsed/60:.1f}min")
    print("Results saved. Resume by running the same command.")
    sys.exit(0)


def main():
    from sparql_evaluation import force_ipv4
    force_ipv4()
    parser = argparse.ArgumentParser(description="Hybrid QA batch evaluation")
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"),
                        help="Groq API key (or set GROQ_API_KEY)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--coverage-file", default=DEFAULT_COVERAGE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between questions (default: 2)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY")
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Load questions
    all_questions = load_all_questions(args.coverage_file)
    if args.limit > 0:
        all_questions = all_questions[:args.limit]

    # Resume
    done_ids = load_completed_ids(args.output_file)
    remaining = [q for q in all_questions if q["question_id"] not in done_ids]
    if done_ids:
        print(f"RESUME: {len(done_ids)} done, {len(remaining)} remaining")

    if not remaining:
        print("All questions already processed.")
        return

    # Create shared clients once
    print("Creating shared clients...")
    llm_client = Groq(api_key=args.api_key)
    linker = DBpediaEntityLinker(min_confidence=0.5)
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Track stats
    _stats["total"] = len(remaining)
    _stats["start"] = time.time()
    em_sum, f1_sum = 0.0, 0.0
    by_method = defaultdict(lambda: {"count": 0, "em": 0.0, "f1": 0.0})

    print(f"\n--- Starting {len(remaining)} questions (delay={args.delay}s) ---\n")

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]
        qtype = q["question_type"]
        is_answerable = q["is_answerable"]

        t0 = time.time()
        record = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold,
            "is_answerable": is_answerable,
        }

        try:
            result = hybrid_answer(
                question, args.api_key, model=args.model,
                llm_client=llm_client, linker=linker,
                kg_client=kg_client, schema_disc=schema_disc,
            )
            record["raw_answer"] = result.answer
            record["predicted_answer"] = clean_answer(result.answer)
            record["method"] = result.method
            record["route_source"] = result.route_source
            record["sparql_fallback"] = result.sparql_fallback
            record["sparql_query"] = getattr(result, "sparql_query", None) or ""
            record["kg_context"] = (getattr(result, "kg_context", None) or "")[:200]
            record["error"] = ""
        except Exception as e:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "error"
            record["route_source"] = ""
            record["sparql_fallback"] = False
            record["sparql_query"] = ""
            record["kg_context"] = ""
            record["error"] = str(e)

        # Compute metrics
        pred = record["predicted_answer"]
        record["em"] = compute_em(pred, gold) if pred else 0.0
        record["f1"] = compute_f1(pred, gold) if pred else 0.0
        record["containment"] = compute_containment(
            record["raw_answer"], gold
        ) if record["raw_answer"] else 0.0
        record["latency_seconds"] = round(time.time() - t0, 2)

        # Append to JSONL
        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update running stats
        em_sum += record["em"]
        f1_sum += record["f1"]
        done = i + 1
        _stats["done"] = done + len(done_ids)
        _stats["em_sum"] = em_sum
        _stats["f1_sum"] = f1_sum

        method = record["method"]
        by_method[method]["count"] += 1
        by_method[method]["em"] += record["em"]
        by_method[method]["f1"] += record["f1"]

        # Progress
        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        if record["containment"] > 0 and record["em"] == 0:
            em_tag += " cont=1"
        elapsed = time.time() - _stats["start"]
        eta = (elapsed / done) * (len(remaining) - done) if done > 0 else 0
        print(
            f"[{len(done_ids)+done}/{len(done_ids)+len(remaining)}] "
            f"{question[:60]}..."
        )
        print(
            f"  -> gold='{gold}' pred='{pred[:60]}' [{em_tag}] "
            f"method={record['method']} {record['latency_seconds']}s"
        )
        if done % 25 == 0:
            print(
                f"  --- Running: EM={em_sum/done*100:.1f}% F1={f1_sum/done*100:.1f}% "
                f"| Elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min ---"
            )

        # Rate limit delay
        if i < len(remaining) - 1:
            time.sleep(args.delay)

    # Final summary
    total_done = len(done_ids) + len(remaining)
    total_em = em_sum / len(remaining) if remaining else 0
    total_f1 = f1_sum / len(remaining) if remaining else 0
    elapsed = time.time() - _stats["start"]

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_done} questions")
    print(f"This run: {len(remaining)} questions in {elapsed/60:.1f}min")
    print(f"  EM:  {total_em*100:.1f}%")
    print(f"  F1:  {total_f1*100:.1f}%")
    print("\nBy method:")
    for method, stats in sorted(by_method.items()):
        c = stats["count"]
        me = stats["em"] / c * 100 if c else 0
        mf = stats["f1"] / c * 100 if c else 0
        print(f"  {method}: {c} questions, EM={me:.1f}%, F1={mf:.1f}%")
    print(f"\nResults: {args.output_file}")


if __name__ == "__main__":
    main()
