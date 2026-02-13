#!/usr/bin/env python3
"""Evaluate GRASP text2SPARQL on WikiMultiHop answerable questions.

Sends DBpedia-answerable questions from 2WikiMultiHopQA to GRASP
(grasp.cs.uni-freiburg.de) via WebSocket and compares answers
against gold using EM, F1, and containment metrics.

Full GRASP traces (tool calls, reasoning, SPARQL) are saved to
per-question JSON files for later analysis.

Sequential execution — GRASP is a shared academic service.

Usage:
    python3 grasp_evaluation.py --limit 5          # quick test
    python3 grasp_evaluation.py                    # all 276 answerable
    python3 grasp_evaluation.py --kg wikidata      # use wikidata
"""

import argparse
import asyncio
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import websockets

# ---------------------------------------------------------------------------
# Metrics (inlined from evaluation/metrics.py to avoid import issues)
# ---------------------------------------------------------------------------

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


def compute_containment(prediction: str, ground_truth: str) -> float:
    """Check if gold answer appears anywhere in the prediction (upper bound)."""
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRASP_WS_URL = "wss://grasp.cs.uni-freiburg.de/api/live"
COVERAGE_FILE = "data/coverage_analysis_800.json"
WS_TIMEOUT = 180        # seconds per WebSocket message
QUESTION_TIMEOUT = 600   # 10 min overall per question
MAX_RETRIES = 3
MIN_REQUEST_DELAY = 2    # seconds between requests (politeness)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_answerable_questions(coverage_file: str) -> list[dict]:
    """Load questions marked as answerable from coverage analysis."""
    with open(coverage_file) as f:
        data = json.load(f)

    answerable = [q for q in data["questions"] if q["is_answerable"]]
    print(f"Loaded {len(answerable)} answerable questions from {coverage_file}")
    return answerable


def load_completed_ids(results_file: str) -> tuple[set[str], list[dict]]:
    """Load completed question IDs and results from a JSONL results file.

    Deduplicates by question_id, keeping last occurrence.
    """
    records_by_id = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        records_by_id[rec["question_id"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass  # skip corrupt/partial lines
    return set(records_by_id.keys()), list(records_by_id.values())

# ---------------------------------------------------------------------------
# Answer extraction & cleaning
# ---------------------------------------------------------------------------

def clean_verbose_answer(raw: str) -> str:
    """Extract concise answer from GRASP's verbose explanation.

    GRASP returns answers like:
      Success: "The Heart of Ezra Greer. Its director died on ..."
      Success: "Charlotte Spurkeland was born later. DBpedia shows ..."
      Cancel:  "I cannot determine ... Charlotte Spurkeland (1987) was born later ..."
      Cancel:  "I identified ... shows Charlotte Spurkeland (1987) was born later ..."
      Short:   "yes"  /  "no"

    We extract just the core answer entity/value.
    """
    raw = raw.strip()
    if not raw:
        return ""

    # Short answers (yes/no, numbers, dates) are fine as-is
    if len(raw.split()) <= 3:
        return raw

    candidate = raw

    # For long cancel-type explanations (multi-paragraph, starts with "I ..."),
    # try to extract a direct answer statement first
    if len(raw) > 500 and raw.startswith("I "):
        # Look for direct answer patterns in the explanation text
        direct_patterns = [
            # "Charlotte Spurkeland (1987) was born later than ..."
            r'(\b[A-Z][A-Za-z\s\'-]+?)(?:\s*\(\d{4}\))?\s+was born (?:later|earlier|first)',
            r'(\b[A-Z][A-Za-z\s\'-]+?)(?:\s*\(\d{4}\))?\s+died (?:earlier|later|first)',
            r'(\b[A-Z][A-Za-z\s\'-]+?)(?:\s*\(\d{4}\))?\s+was released (?:earlier|later|first)',
            r'(\b[A-Z][A-Za-z\s\'-]+?)(?:\s*\(\d{4}\))?\s+was founded (?:earlier|later|first)',
            # "the answer is Charlotte Spurkeland"
            r'(?:the answer is|answer:|the result is)\s+([A-Z][A-Za-z\s\'-]+)',
            # "which shows Charlotte Spurkeland was born later"
            r'which shows\s+(\b[A-Z][A-Za-z\s\'-]+?)(?:\s*\(\d{4}\))?\s+was\b',
        ]
        for pat in direct_patterns:
            m = re.search(pat, raw)
            if m:
                candidate = m.group(1).strip().rstrip('.')
                return candidate

        # If no direct pattern found, the cancel explanation likely has no answer
        # Return empty — the containment metric will still catch partial matches
        return ""

    # Split on common explanation delimiters
    for sep in [' — ', '. Its ', '. The ', '. DBpedia ', '. According ',
                '. This ', '. He ', '. She ', '. It ', '; ']:
        if sep in candidate:
            candidate = candidate.split(sep)[0].strip().rstrip('.')
            break
    else:
        # No separator found: take first sentence
        candidate = re.split(r'(?<=[.!])\s+', raw)[0].rstrip('.')

    # Remove trailing parenthetical explanations
    candidate = re.sub(
        r'\s*\((?:the |its |whose |directed |born |died |with ).*$',
        '', candidate, flags=re.IGNORECASE,
    )

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
    """Extract answer from GRASP output. Returns (raw_answer, cleaned_answer).

    GRASP output.type can be:
      - "answer": success → answer in "answer" field
      - "cancel": failure/partial → explanation in "explanation" field
    """
    if not output or not isinstance(output, dict):
        return "", ""

    grasp_output = output.get("output", {})
    raw = ""
    if isinstance(grasp_output, dict):
        # Success case: answer field present
        if "answer" in grasp_output:
            raw = str(grasp_output["answer"]).strip()
        # Cancel case: GRASP couldn't produce a working SPARQL query
        # but may still contain the answer in its explanation
        elif "explanation" in grasp_output:
            raw = str(grasp_output["explanation"]).strip()
        elif "formatted" in grasp_output:
            raw = str(grasp_output["formatted"]).strip()
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

# ---------------------------------------------------------------------------
# GRASP WebSocket client
# ---------------------------------------------------------------------------

async def query_grasp(
    question: str,
    knowledge_graphs: list[str],
    task: str = "sparql-qa",
    msg_timeout: int = WS_TIMEOUT,
) -> tuple[dict, list[dict]]:
    """Send a question to GRASP and return (final_output, all_messages).

    Returns:
        Tuple of (final_response_dict, list_of_all_ws_messages).
        The final_response_dict has type="output" or contains "error".
        all_messages is the full trace of every WS message received.
    """
    all_messages = []

    async with websockets.connect(GRASP_WS_URL) as ws:
        msg = {
            "task": task,
            "input": question,
            "knowledge_graphs": knowledge_graphs,
            "past": None,
        }
        await ws.send(json.dumps(msg))

        while True:
            raw_response = await asyncio.wait_for(ws.recv(), timeout=msg_timeout)

            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError:
                # Malformed message from server — log and skip
                all_messages.append({"_raw": raw_response[:500], "_parse_error": True})
                await ws.send(json.dumps({"received": True}))
                continue

            all_messages.append(data)
            await ws.send(json.dumps({"received": True}))

            msg_type = data.get("type", "")

            if msg_type == "output":
                return data, all_messages

            if "error" in data and "type" not in data:
                return data, all_messages

    # Unreachable (while True only exits via return), but satisfies type checker
    return {"error": "Connection closed unexpectedly"}, all_messages

# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def retry_delay(attempt: int) -> float:
    """Exponential backoff: 5s, 15s, 45s."""
    return 5 * (3 ** (attempt - 1))


async def evaluate_question(
    q: dict,
    kg: str,
    task: str,
    trace_dir: str,
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
    all_messages = []

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            grasp_result, all_messages = await asyncio.wait_for(
                query_grasp(question_text, knowledge_graphs=[kg], task=task),
                timeout=QUESTION_TIMEOUT,
            )

            if "error" in grasp_result and "type" not in grasp_result:
                error = grasp_result["error"]
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(retry_delay(attempt))
                    continue
            else:
                raw_answer, predicted_answer = extract_answer_from_output(grasp_result)
                sparql = extract_sparql_from_output(grasp_result)
                error = ""
                break

        except asyncio.TimeoutError:
            error = f"Timeout (attempt {attempt})"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(retry_delay(attempt))
        except websockets.exceptions.ConnectionClosed as e:
            error = f"Connection closed: {e} (attempt {attempt})"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(retry_delay(attempt))
        except json.JSONDecodeError as e:
            error = f"JSONDecodeError: {e} (attempt {attempt})"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(retry_delay(attempt))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            break

    latency = time.time() - start

    # Metrics
    em = compute_em(predicted_answer, gold_answer) if predicted_answer else 0.0
    f1 = compute_f1(predicted_answer, gold_answer) if predicted_answer else 0.0
    raw_em = compute_em(raw_answer, gold_answer) if raw_answer else 0.0
    raw_f1 = compute_f1(raw_answer, gold_answer) if raw_answer else 0.0
    containment = compute_containment(raw_answer, gold_answer) if raw_answer else 0.0

    # Save full trace to separate file
    trace_file = ""
    if trace_dir and all_messages:
        os.makedirs(trace_dir, exist_ok=True)
        trace_file = os.path.join(trace_dir, f"{question_id}.json")
        trace_data = {
            "question_id": question_id,
            "question": question_text,
            "kg": kg,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "messages": all_messages,
        }
        with open(trace_file, "w") as tf:
            json.dump(trace_data, tf, indent=2, ensure_ascii=False)

    # Count tool calls from trace
    tool_call_count = sum(1 for m in all_messages if m.get("type") == "tool")

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
        "containment": containment,
        "latency_seconds": round(latency, 2),
        "tool_calls": tool_call_count,
        "error": error,
        "trace_file": trace_file,
    }

# ---------------------------------------------------------------------------
# Progress & summary
# ---------------------------------------------------------------------------

def print_progress(results: list[dict], total: int):
    """Print running metrics summary."""
    n = len(results)
    if n == 0:
        return

    answered = sum(1 for r in results if r["predicted_answer"])
    errors = sum(1 for r in results if r["error"])
    avg_em = sum(r["em"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    avg_cont = sum(r.get("containment", 0) for r in results) / n
    avg_latency = sum(r["latency_seconds"] for r in results) / n

    print(f"\n  [{n}/{total}] EM={avg_em:.1%} F1={avg_f1:.1%} "
          f"contain={avg_cont:.1%} "
          f"answered={answered}/{n} errors={errors} "
          f"avg_latency={avg_latency:.1f}s")


def save_summary(results: list[dict], kg: str, output_dir: str):
    """Save aggregated summary JSON."""
    n = len(results)
    if n == 0:
        return

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    def agg(recs):
        k = len(recs)
        return {
            "n": k,
            "em": round(sum(r["em"] for r in recs) / k, 4),
            "f1": round(sum(r["f1"] for r in recs) / k, 4),
            "containment": round(sum(r.get("containment", 0) for r in recs) / k, 4),
            "answered": sum(1 for r in recs if r["predicted_answer"]),
            "errors": sum(1 for r in recs if r["error"]),
        }

    summary = {
        "kg": kg,
        "timestamp": datetime.now().isoformat(),
        "total": n,
        "overall": {
            **agg(results),
            "avg_latency": round(sum(r["latency_seconds"] for r in results) / n, 2),
        },
        "by_type": {qtype: agg(recs) for qtype, recs in sorted(by_type.items())},
    }

    summary_file = os.path.join(output_dir, f"grasp_eval_{kg}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GRASP text2SPARQL on WikiMultiHop answerable questions"
    )
    parser.add_argument("--kg", default="dbpedia",
                        help="Knowledge graph (default: dbpedia)")
    parser.add_argument("--task", default="sparql-qa",
                        choices=["sparql-qa", "general-qa"],
                        help="GRASP task type (default: sparql-qa)")
    parser.add_argument("--coverage-file", default=COVERAGE_FILE,
                        help="Coverage analysis file with per-question answerability")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions to evaluate (0 = all answerable)")
    parser.add_argument("--output-dir", default="data",
                        help="Directory for output files")
    args = parser.parse_args()

    # Load answerable questions
    questions = load_answerable_questions(args.coverage_file)

    if args.limit > 0:
        questions = questions[:args.limit]

    # Output paths
    results_file = os.path.join(args.output_dir, f"grasp_eval_{args.kg}.jsonl")
    trace_dir = os.path.join(args.output_dir, f"grasp_eval_{args.kg}_traces")

    # Auto-resume: always check for existing results
    completed_ids, all_results = load_completed_ids(results_file)
    if completed_ids:
        print(f"Auto-resume: found {len(completed_ids)} completed in {results_file}")

    remaining = [q for q in questions if q["question_id"] not in completed_ids]
    total = len(questions)

    print(f"\nGRASP Evaluation")
    print(f"  KG: {args.kg}")
    print(f"  Task: {args.task}")
    print(f"  Total answerable: {total}")
    print(f"  Already done: {len(completed_ids)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Output: {results_file}")
    print(f"  Traces: {trace_dir}/")
    est_min = len(remaining) * 1.5
    print(f"  Estimated time: ~{est_min:.0f} min ({est_min/60:.1f} hr)")
    print()

    if not remaining:
        print("All questions already completed.")
        print_progress(all_results, total)
        save_summary(all_results, args.kg, args.output_dir)
        return

    # Process questions sequentially (GRASP is a shared public service)
    with open(results_file, "a") as out_f:
        for i, q in enumerate(remaining):
            idx = len(completed_ids) + i + 1
            print(f"[{idx}/{total}] {q['question'][:80]}...")

            result = await evaluate_question(q, args.kg, args.task, trace_dir)
            all_results.append(result)

            # Status indicator
            if result["error"]:
                status = f"ERR: {result['error'][:50]}"
            elif not result["predicted_answer"]:
                status = "NO ANSWER"
            elif result["em"] == 1.0:
                status = "EM=1"
            else:
                status = f"F1={result['f1']:.2f} cont={result['containment']:.0f}"
            print(f"  -> gold='{result['gold_answer']}' "
                  f"pred='{result['predicted_answer'][:60]}' "
                  f"[{status}] {result['latency_seconds']:.1f}s")

            # Write result immediately (crash-safe)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            # Progress every 10
            if idx % 10 == 0:
                print_progress(all_results, total)

            # Politeness delay between requests
            if i < len(remaining) - 1:
                await asyncio.sleep(MIN_REQUEST_DELAY)

    # Final summary
    print("\n" + "=" * 70)
    print("GRASP EVALUATION COMPLETE")
    print("=" * 70)
    print_progress(all_results, total)
    save_summary(all_results, args.kg, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
