#!/usr/bin/env python3
"""Evaluate GRASP text2SPARQL on WikiMultiHop answerable questions.

Sends the 31% DBpedia-answerable questions from 2WikiMultiHopQA
to GRASP (grasp.cs.uni-freiburg.de) via WebSocket and compares
GRASP's answers against gold answers using EM and F1.

Usage:
    # Test with 5 questions
    python3 grasp_evaluation.py --limit 5

    # Run on all 276 answerable questions (from 800-sample)
    python3 grasp_evaluation.py

    # Use wikidata instead of dbpedia
    python3 grasp_evaluation.py --kg wikidata --limit 10

    # Resume from previous run
    python3 grasp_evaluation.py --resume data/grasp_eval_dbpedia.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime

import string
from collections import Counter, defaultdict

import websockets


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (2WikiMultiHopQA standard)."""
    answer = answer.lower()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    answer = ''.join(ch for ch in answer if ch not in string.punctuation)
    return ' '.join(answer.split()).strip()


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


GRASP_WS_URL = "wss://grasp.cs.uni-freiburg.de/api/live"
COVERAGE_FILE = "data/coverage_analysis_800.json"
WS_TIMEOUT = 180  # seconds per message
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds between retries


def load_answerable_questions(coverage_file: str) -> list[dict]:
    """Load questions marked as answerable from coverage analysis."""
    with open(coverage_file) as f:
        data = json.load(f)

    answerable = [q for q in data["questions"] if q["is_answerable"]]
    print(f"Loaded {len(answerable)} answerable questions from {coverage_file}")
    return answerable


def load_full_questions(dev_file: str = "data/dev_full.json") -> dict:
    """Load full question data keyed by question_id."""
    with open(dev_file) as f:
        data = json.load(f)
    return {q["_id"]: q for q in data}


def load_completed_ids(results_file: str) -> set[str]:
    """Load already-completed question IDs from a JSONL results file."""
    completed = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        completed.add(rec["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return completed


def clean_verbose_answer(raw: str) -> str:
    """Extract concise answer from GRASP's verbose explanation.

    GRASP returns answers like:
      "The Heart of Ezra Greer. Its director died on ..."
      "Charlotte Spurkeland was born later. DBpedia shows ..."
      "Thedi Vandha Mappillai — its director B. R. Panthulu ..."
      "yes"  /  "no"

    We extract just the core answer entity/value.
    """
    raw = raw.strip()
    if not raw:
        return ""

    # Short answers (yes/no, numbers, dates) are fine as-is
    if len(raw.split()) <= 3:
        return raw

    # Split on common explanation delimiters
    # Order matters: try most specific first
    candidate = raw
    for sep in [' — ', '. Its ', '. The ', '. DBpedia ', '. According ',
                '. This ', '. He ', '. She ', '. It ', '; ']:
        if sep in candidate:
            candidate = candidate.split(sep)[0].strip().rstrip('.')
            break
    else:
        # No separator found: take first sentence
        candidate = re.split(r'(?<=[.!])\s+', raw)[0].rstrip('.')

    # Remove trailing parenthetical explanations: "Film Title (the director..."
    candidate = re.sub(r'\s*\((?:the |its |whose |directed ).*$', '', candidate, flags=re.IGNORECASE)

    # Remove trailing filler phrases (comparison/bridge questions)
    filler_patterns = [
        ' was born later', ' was born first', ' was born earlier',
        ' died earlier', ' died later', ' died first',
        ' was released first', ' was released earlier',
        ' was released later', ' came out first',
        ' has the director who', ' is the answer',
        ' is older', ' is younger', ' is the older',
        ' was founded first', ' was founded earlier',
    ]
    for filler in filler_patterns:
        if filler in candidate.lower():
            idx = candidate.lower().index(filler)
            candidate = candidate[:idx].strip()
            break

    return candidate.strip().rstrip('.')


def extract_answer_from_output(output: dict) -> tuple[str, str]:
    """Extract the final answer string from GRASP output.

    Returns (raw_answer, cleaned_answer).
    """
    if not output or not isinstance(output, dict):
        return "", ""

    grasp_output = output.get("output", {})
    raw = ""
    if isinstance(grasp_output, dict):
        if "answer" in grasp_output:
            raw = str(grasp_output["answer"]).strip()
        elif "output" in grasp_output:
            raw = str(grasp_output["output"]).strip()

    cleaned = clean_verbose_answer(raw)
    return raw, cleaned


def extract_sparql_from_output(output: dict) -> str:
    """Extract the SPARQL query from GRASP output."""
    if not output or not isinstance(output, dict):
        return ""
    grasp_output = output.get("output", {})
    if isinstance(grasp_output, dict) and "sparql" in grasp_output:
        return grasp_output["sparql"]
    return ""


async def query_grasp(
    question: str,
    knowledge_graphs: list[str],
    task: str = "sparql-qa",
    timeout: int = WS_TIMEOUT,
) -> dict:
    """Send a question to GRASP via WebSocket and return the full response."""
    async with websockets.connect(GRASP_WS_URL) as ws:
        msg = {
            "task": task,
            "input": question,
            "knowledge_graphs": knowledge_graphs,
            "past": None,
        }
        await ws.send(json.dumps(msg))

        tool_calls = []
        reasoning_steps = []

        while True:
            response = await asyncio.wait_for(ws.recv(), timeout=timeout)
            data = json.loads(response)

            # Acknowledge
            await ws.send(json.dumps({"received": True}))

            msg_type = data.get("type", "")

            if msg_type == "tool":
                tool_calls.append({
                    "name": data.get("name", ""),
                    "args": data.get("args", {}),
                })

            elif msg_type == "model":
                if data.get("reasoning"):
                    reasoning_steps.append(data["reasoning"][:200])

            elif msg_type == "output":
                data["_tool_calls"] = tool_calls
                data["_reasoning_steps"] = reasoning_steps
                return data

            elif "error" in data and "type" not in data:
                return {
                    "error": data["error"],
                    "_tool_calls": tool_calls,
                    "_reasoning_steps": reasoning_steps,
                }

    return {"error": "Connection closed unexpectedly"}


async def evaluate_question(
    q: dict,
    kg: str,
    task: str,
) -> dict:
    """Evaluate a single question against GRASP and return result record."""
    question_id = q["question_id"]
    question_text = q["question"]
    gold_answer = q["gold_answer"]
    question_type = q.get("question_type", "unknown")

    start = time.time()
    error = ""
    raw_answer = ""
    predicted_answer = ""
    sparql = ""
    grasp_raw = {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            grasp_raw = await query_grasp(
                question_text,
                knowledge_graphs=[kg],
                task=task,
            )

            if "error" in grasp_raw and "type" not in grasp_raw:
                error = grasp_raw["error"]
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
            else:
                raw_answer, predicted_answer = extract_answer_from_output(grasp_raw)
                sparql = extract_sparql_from_output(grasp_raw)
                error = ""
                break

        except asyncio.TimeoutError:
            error = f"Timeout after {WS_TIMEOUT}s (attempt {attempt})"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
        except websockets.exceptions.ConnectionClosed as e:
            error = f"Connection closed: {e} (attempt {attempt})"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            break

    latency = time.time() - start

    em = compute_em(predicted_answer, gold_answer) if predicted_answer else 0.0
    f1 = compute_f1(predicted_answer, gold_answer) if predicted_answer else 0.0

    # Also compute metrics on raw answer for comparison
    raw_em = compute_em(raw_answer, gold_answer) if raw_answer else 0.0
    raw_f1 = compute_f1(raw_answer, gold_answer) if raw_answer else 0.0

    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question_text,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "raw_answer": raw_answer,
        "sparql": sparql,
        "em": em,
        "f1": f1,
        "raw_em": raw_em,
        "raw_f1": raw_f1,
        "latency_seconds": round(latency, 2),
        "tool_calls": len(grasp_raw.get("_tool_calls", [])),
        "error": error,
    }


def print_progress(results: list[dict], total: int):
    """Print running metrics summary."""
    n = len(results)
    if n == 0:
        return

    answered = [r for r in results if r["predicted_answer"]]
    errors = [r for r in results if r["error"]]
    avg_em = sum(r["em"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    avg_latency = sum(r["latency_seconds"] for r in results) / n

    print(f"\n  [{n}/{total}] EM={avg_em:.1%} F1={avg_f1:.1%} "
          f"answered={len(answered)}/{n} errors={len(errors)} "
          f"avg_latency={avg_latency:.1f}s")


def save_summary(results: list[dict], kg: str, output_dir: str):
    """Save aggregated summary JSON."""
    n = len(results)
    if n == 0:
        return

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    type_stats = {}
    for qtype, recs in sorted(by_type.items()):
        k = len(recs)
        type_stats[qtype] = {
            "n": k,
            "em": round(sum(r["em"] for r in recs) / k, 4),
            "f1": round(sum(r["f1"] for r in recs) / k, 4),
            "answered": sum(1 for r in recs if r["predicted_answer"]),
            "errors": sum(1 for r in recs if r["error"]),
        }

    summary = {
        "kg": kg,
        "timestamp": datetime.now().isoformat(),
        "total": n,
        "overall": {
            "em": round(sum(r["em"] for r in results) / n, 4),
            "f1": round(sum(r["f1"] for r in results) / n, 4),
            "answered": sum(1 for r in results if r["predicted_answer"]),
            "errors": sum(1 for r in results if r["error"]),
            "avg_latency": round(sum(r["latency_seconds"] for r in results) / n, 2),
        },
        "by_type": type_stats,
    }

    summary_file = os.path.join(output_dir, f"grasp_eval_{kg}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GRASP text2SPARQL on WikiMultiHop answerable questions"
    )
    parser.add_argument(
        "--kg", default="dbpedia",
        help="Knowledge graph to use (default: dbpedia)",
    )
    parser.add_argument(
        "--task", default="sparql-qa",
        choices=["sparql-qa", "general-qa"],
        help="GRASP task type (default: sparql-qa)",
    )
    parser.add_argument(
        "--coverage-file", default=COVERAGE_FILE,
        help="Coverage analysis file with per-question answerability",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max questions to evaluate (0 = all answerable)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from existing JSONL results file (skip completed)",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory for output files",
    )
    args = parser.parse_args()

    # Load answerable questions
    questions = load_answerable_questions(args.coverage_file)

    # Apply limit
    if args.limit > 0:
        questions = questions[:args.limit]

    # Output file
    results_file = os.path.join(
        args.output_dir, f"grasp_eval_{args.kg}.jsonl"
    )

    # Resume support
    completed_ids = set()
    all_results = []
    if args.resume:
        results_file = args.resume
        completed_ids = load_completed_ids(results_file)
        # Load existing results for summary
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming: {len(completed_ids)} questions already completed")

    remaining = [q for q in questions if q["question_id"] not in completed_ids]
    total = len(questions)

    print(f"\nGRASP Evaluation")
    print(f"  KG: {args.kg}")
    print(f"  Task: {args.task}")
    print(f"  Total answerable: {total}")
    print(f"  Already done: {len(completed_ids)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Output: {results_file}")
    print(f"  Estimated time: ~{len(remaining) * 1.5:.0f} min")
    print()

    # Process questions sequentially (GRASP is a shared service)
    with open(results_file, "a") as out_f:
        for i, q in enumerate(remaining):
            idx = len(completed_ids) + i + 1
            print(f"[{idx}/{total}] {q['question'][:80]}...")

            result = await evaluate_question(q, args.kg, args.task)
            all_results.append(result)

            # Status indicator
            status = "OK" if result["em"] == 1.0 else "F1={:.2f}".format(result["f1"])
            if result["error"]:
                status = f"ERR: {result['error'][:50]}"
            elif not result["predicted_answer"]:
                status = "NO ANSWER"
            print(f"  -> gold='{result['gold_answer']}' "
                  f"pred='{result['predicted_answer'][:60]}' "
                  f"[{status}] {result['latency_seconds']:.1f}s")

            # Write result immediately (crash-safe)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            # Progress every 10
            if idx % 10 == 0:
                print_progress(all_results, total)

    # Final summary
    print("\n" + "=" * 70)
    print("GRASP EVALUATION COMPLETE")
    print("=" * 70)
    print_progress(all_results, total)
    save_summary(all_results, args.kg, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
