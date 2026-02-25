#!/usr/bin/env python3
"""Evaluate hybrid_qa pipeline on the 19 questions that defeat both SPARQL and GRASP.

Runs each question through hybrid_qa.py via subprocess:
  EntityLink → Schema → SPARQL-first → text fallback (KG explore + Wikipedia) → LLM

Saves all verbose intermediate output per question for analysis.

Usage:
    python3 graphrag_eval_failures.py --limit 3          # quick test
    python3 graphrag_eval_failures.py                    # all 19
    python3 graphrag_eval_failures.py --no-resume        # start fresh
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evaluation.metrics import compute_containment, compute_em, compute_f1

# ---------------------------------------------------------------------------
# Extract answer from hybrid_qa verbose output
# ---------------------------------------------------------------------------

def extract_answer_from_output(output: str) -> tuple[str, str]:
    """Parse hybrid_qa.py output to extract answer and method.

    Looks for patterns:
      RESULT (method=sparql)
        Answer: <answer>
      OR
      Answer (text_explore): <answer>
      OR
      Answer (<method>): <answer>
    """
    answer = ""
    method = ""

    # Pattern 1: verbose RESULT block
    m = re.search(r'RESULT\s+\(method=(\w+)\)', output)
    if m:
        method = m.group(1)

    # Pattern 2: "Answer:" line in RESULT block
    m = re.search(r'Answer:\s*(.+)', output)
    if m:
        answer = m.group(1).strip()

    # Pattern 3: non-verbose "Answer (method): text"
    if not answer:
        m = re.search(r'Answer\s+\((\w+)\):\s*(.+)', output)
        if m:
            method = m.group(1)
            answer = m.group(2).strip()

    # Pattern 4: FINAL ANSWER in LLM output (case-insensitive)
    if not answer:
        m = re.search(r'FINAL ANSWER:\s*(.+)', output, re.IGNORECASE)
        if m:
            answer = m.group(1).strip()

    return answer, method


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

FAILURES_PATH = "data/both_systems_failures.json"
OUTPUT_JSONL = "data/graphrag_eval_failures.jsonl"
TRACES_DIR = "data/graphrag_eval_traces"
SCRIPT_DIR = Path(__file__).resolve().parent


def load_questions(limit: int | None = None) -> list[dict]:
    with open(FAILURES_PATH) as f:
        data = json.load(f)
    questions = data["questions"]
    if limit:
        questions = questions[:limit]
    return questions


def load_completed_ids() -> set[str]:
    completed = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed.add(rec["question_id"])
    return completed


def run_hybrid_qa(question: str, api_key: str, model: str) -> tuple[str, float]:
    """Run hybrid_qa.py as subprocess, return (full_output, latency)."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "hybrid_qa.py"),
        question,
        "--api-key", api_key,
        "--model", model,
        "--verbose",
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
        cwd=str(SCRIPT_DIR),
    )
    latency = time.time() - t0
    output = result.stdout + "\n" + result.stderr
    return output, latency


def print_summary(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    em_sum = sum(r["em"] for r in results)
    f1_sum = sum(r["f1"] for r in results)
    cont_sum = sum(r["containment"] for r in results)
    answered = sum(1 for r in results if r["hybrid_answer"])

    print(f"\n{'=' * 70}")
    print(f"  HYBRID QA EVALUATION ON BOTH-SYSTEM FAILURES (n={n})")
    print(f"{'=' * 70}")
    print(f"  EM:          {em_sum/n:.4f}  ({em_sum:.0f}/{n})")
    print(f"  F1:          {f1_sum/n:.4f}")
    print(f"  Containment: {cont_sum/n:.4f}  ({cont_sum:.0f}/{n})")
    print(f"  Answered:    {answered}/{n}")

    # Methods breakdown
    methods = defaultdict(int)
    for r in results:
        methods[r.get("method", "unknown")] += 1
    print(f"  Methods:     {dict(methods)}")

    # By type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    print(f"\n  {'Type':<22} {'n':>4} {'EM':>8} {'F1':>8} {'Cont':>8}")
    print(f"  {'-'*22} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")
    for qtype in sorted(by_type.keys()):
        items = by_type[qtype]
        tn = len(items)
        tem = sum(r["em"] for r in items) / tn
        tf1 = sum(r["f1"] for r in items) / tn
        tcont = sum(r["containment"] for r in items) / tn
        print(f"  {qtype:<22} {tn:>4} {tem:>8.4f} {tf1:>8.4f} {tcont:>8.4f}")

    # Per-question detail
    print(f"\n  {'#':>2} {'Type':<18} {'EM':>3} {'Method':<16} {'Gold':<22} {'Hybrid Answer':<40}")
    print(f"  {'-'*2} {'-'*18} {'-'*3} {'-'*16} {'-'*22} {'-'*40}")
    for i, r in enumerate(results, 1):
        em_mark = "Y" if r["em"] else "N"
        gold = r["gold_answer"][:21]
        pred = r["hybrid_answer"][:39] if r["hybrid_answer"] else "(empty)"
        meth = r.get("method", "?")[:15]
        print(f"  {i:>2} {r['question_type']:<18} {em_mark:>3} {meth:<16} {gold:<22} {pred:<40}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid_qa on questions that defeat both SPARQL and GRASP"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-key",
                        default=os.environ.get("GROQ_API_KEY"))
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set GROQ_API_KEY env var")
        sys.exit(1)

    os.makedirs(TRACES_DIR, exist_ok=True)

    questions = load_questions(args.limit)
    print(f"Loaded {len(questions)} both-system failure questions")
    print(f"Model: {args.model}")

    completed_ids = set() if args.no_resume else load_completed_ids()
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already completed")

    # Load existing results for final summary
    existing_results = []
    if not args.no_resume and os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as f:
            for line in f:
                if line.strip():
                    existing_results.append(json.loads(line))
    results = list(existing_results)

    for i, q in enumerate(questions, 1):
        if q["question_id"] in completed_ids:
            print(f"[{i}/{len(questions)}] SKIP (done): {q['question'][:60]}")
            continue

        qid = q["question_id"]
        print(f"\n[{i}/{len(questions)}] {q['question_type']}: {q['question']}")
        print(f"  Gold: {q['gold_answer']}")

        try:
            output, latency = run_hybrid_qa(q["question"], args.api_key, args.model)
        except subprocess.TimeoutExpired:
            output = "TIMEOUT after 300s"
            latency = 300.0
        except Exception as e:
            output = f"ERROR: {e}"
            latency = 0.0

        # Save full verbose trace
        trace_path = os.path.join(TRACES_DIR, f"{qid}.txt")
        with open(trace_path, "w") as f:
            f.write(output)

        # Extract answer
        answer, method = extract_answer_from_output(output)

        em = compute_em(answer, q["gold_answer"]) if answer else 0.0
        f1 = compute_f1(answer, q["gold_answer"]) if answer else 0.0
        containment = compute_containment(answer, q["gold_answer"]) if answer else 0.0

        result = {
            "question_id": qid,
            "question_type": q["question_type"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "sparql_answer": q.get("sparql_answer", ""),
            "grasp_answer": q.get("grasp_answer", ""),
            "hybrid_answer": answer,
            "method": method,
            "em": em,
            "f1": round(f1, 4),
            "containment": containment,
            "latency_seconds": round(latency, 1),
            "trace_file": trace_path,
        }
        results.append(result)

        # Append to JSONL immediately
        with open(OUTPUT_JSONL, "a") as f:
            f.write(json.dumps(result) + "\n")

        em_mark = "CORRECT" if em else "WRONG"
        print(f"  Hybrid ({method}): {answer[:80]}")
        print(f"  EM={em_mark}  F1={f1:.2f}  Cont={containment:.0f}  ({latency:.1f}s)")

        time.sleep(1.0)

    print_summary(results)

    # Save summary
    if results:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "total": len(results),
            "em": round(sum(r["em"] for r in results) / len(results), 4),
            "f1": round(sum(r["f1"] for r in results) / len(results), 4),
            "containment": round(sum(r["containment"] for r in results) / len(results), 4),
        }
        summary_path = "data/graphrag_eval_failures_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results: {OUTPUT_JSONL}")
        print(f"Traces:  {TRACES_DIR}/")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
