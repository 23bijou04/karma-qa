#!/usr/bin/env python3
"""Offline evaluation of GRASP results.

Re-scores JSONL results from grasp_evaluation.py without re-querying GRASP.
Supports heuristic re-cleaning, containment check, LLM judge, and error analysis.

Usage:
    # Re-score with latest cleaning rules
    python3 grasp_evaluate_results.py data/grasp_eval_dbpedia.jsonl

    # Add LLM judge (requires Ollama running)
    python3 grasp_evaluate_results.py data/grasp_eval_dbpedia.jsonl --judge

    # Show error analysis
    python3 grasp_evaluate_results.py data/grasp_eval_dbpedia.jsonl --errors
"""

import argparse
import json
import sys
from collections import defaultdict

from evaluation.metrics import compute_containment, compute_em, compute_f1
from grasp_evaluation import clean_verbose_answer

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    """Load JSONL results, deduplicate by question_id (keep last)."""
    records_by_id = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records_by_id[rec["question_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    results = list(records_by_id.values())
    print(f"Loaded {len(results)} results from {path}")
    return results

# ---------------------------------------------------------------------------
# Re-scoring
# ---------------------------------------------------------------------------

def rescore(results: list[dict]) -> list[dict]:
    """Re-apply answer cleaning and recompute all metrics."""
    for r in results:
        raw = r.get("raw_answer", "")
        gold = r.get("gold_answer", "")

        # Re-clean with latest rules
        cleaned = clean_verbose_answer(raw) if raw else ""
        r["predicted_answer"] = cleaned

        # Recompute metrics
        r["em"] = compute_em(cleaned, gold) if cleaned else 0.0
        r["f1"] = compute_f1(cleaned, gold) if cleaned else 0.0
        r["raw_em"] = compute_em(raw, gold) if raw else 0.0
        r["raw_f1"] = compute_f1(raw, gold) if raw else 0.0
        r["containment"] = compute_containment(raw, gold) if raw else 0.0
    return results

# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def run_judge(results: list[dict], model: str = "qwen2.5:7b") -> list[dict]:
    """Run LLM judge on non-EM-matching answers."""
    try:
        from ablation_study_v3.llm_judge import judge_one
    except ImportError:
        print("Warning: Could not import judge_one from ablation_study_v3.llm_judge")
        print("Falling back to inline implementation")
        judge_one = _judge_one_fallback

    # Verify Ollama is running
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("Error: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    total = len(results)
    strict_em = sum(1 for r in results if r["em"] > 0)
    judge_correct = 0
    upgraded = 0

    for i, r in enumerate(results, 1):
        if r["em"] > 0:
            r["judge_correct"] = True
            judge_correct += 1
            continue

        pred = r.get("predicted_answer", "")
        if not pred:
            r["judge_correct"] = False
            continue

        is_correct = judge_one(
            r.get("question", ""),
            r.get("gold_answer", ""),
            pred,
            model,
        )
        r["judge_correct"] = is_correct
        if is_correct:
            judge_correct += 1
            upgraded += 1

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] strict_EM={strict_em} judge_EM={judge_correct}")

    print(f"\n  Judge complete: strict_EM={strict_em}/{total} "
          f"({strict_em/total*100:.1f}%) -> "
          f"judge_EM={judge_correct}/{total} "
          f"({judge_correct/total*100:.1f}%) "
          f"[+{upgraded} upgraded]")

    return results


def _judge_one_fallback(question: str, gold: str, pred: str, model: str) -> bool:
    """Fallback judge if import fails."""
    import requests

    prompt = f"""You are judging whether a predicted answer is semantically equivalent to the gold (correct) answer for a question-answering task.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Rules:
- "yes" if the predicted answer refers to the same entity, place, date, or concept as the gold answer
- "yes" if the predicted answer is more specific but correct (e.g., "Newark, New Jersey" vs "Newark")
- "yes" if it's the same entity with minor spelling/formatting differences
- "no" if the predicted answer is a different entity, wrong fact, or irrelevant
- "no" if the predicted answer is empty or says "UNKNOWN"

Answer ONLY "yes" or "no":"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 10}},
            timeout=30,
        )
        answer = resp.json().get("response", "").strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], title: str = "GRASP Evaluation Results"):
    """Print formatted summary table."""
    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)

    def agg(recs):
        k = len(recs)
        d = {
            "n": k,
            "em": sum(r["em"] for r in recs) / k,
            "f1": sum(r["f1"] for r in recs) / k,
            "containment": sum(r.get("containment", 0) for r in recs) / k,
            "answered": sum(1 for r in recs if r.get("predicted_answer")),
            "errors": sum(1 for r in recs if r.get("error")),
        }
        if any("judge_correct" in r for r in recs):
            judged = [r for r in recs if "judge_correct" in r]
            d["judge_em"] = sum(1 for r in judged if r["judge_correct"]) / k
        return d

    overall = agg(results)

    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")
    print(f"  N={n}  answered={overall['answered']}  errors={overall['errors']}")
    print()
    print(f"  {'Metric':<20} {'Overall':>10}", end="")
    for qtype in sorted(by_type.keys()):
        print(f"  {qtype[:12]:>12}", end="")
    print()
    print(f"  {'-'*20} {'-'*10}", end="")
    for _ in by_type:
        print(f"  {'-'*12}", end="")
    print()

    type_aggs = {qt: agg(recs) for qt, recs in sorted(by_type.items())}

    for metric_name, metric_key in [
        ("EM (cleaned)", "em"),
        ("F1 (cleaned)", "f1"),
        ("Containment", "containment"),
        ("Judge EM", "judge_em"),
    ]:
        if metric_key not in overall:
            continue
        print(f"  {metric_name:<20} {overall[metric_key]:>9.1%}", end="")
        for qtype in sorted(by_type.keys()):
            val = type_aggs[qtype].get(metric_key, 0)
            print(f"  {val:>11.1%}", end="")
        print()

    # N per type
    print(f"  {'N':<20} {n:>10}", end="")
    for qtype in sorted(by_type.keys()):
        print(f"  {type_aggs[qtype]['n']:>12}", end="")
    print()

    print(f"{'=' * 75}")


def print_errors(results: list[dict]):
    """Print error analysis breakdown."""
    n = len(results)
    if n == 0:
        return

    print(f"\n{'=' * 75}")
    print("  ERROR ANALYSIS")
    print(f"{'=' * 75}")

    # Error type breakdown
    error_types = defaultdict(int)
    for r in results:
        err = r.get("error", "")
        if not err:
            continue
        if "Timeout" in err:
            error_types["Timeout"] += 1
        elif "Connection closed" in err:
            error_types["Connection closed"] += 1
        elif "JSONDecodeError" in err:
            error_types["JSON parse error"] += 1
        else:
            error_types["Other"] += 1

    total_errors = sum(error_types.values())
    print(f"\n  Error breakdown ({total_errors} total):")
    for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"    {etype:<25} {count:>5} ({count/n*100:.1f}%)")

    # Answer status breakdown
    no_answer = sum(1 for r in results if not r.get("error") and not r.get("predicted_answer"))
    wrong = sum(1 for r in results if r.get("predicted_answer") and r["em"] == 0)
    correct = sum(1 for r in results if r["em"] > 0)
    print("\n  Answer status:")
    print(f"    Correct (EM=1):      {correct:>5} ({correct/n*100:.1f}%)")
    print(f"    Wrong answer:        {wrong:>5} ({wrong/n*100:.1f}%)")
    print(f"    No answer:           {no_answer:>5} ({no_answer/n*100:.1f}%)")
    print(f"    Errors:              {total_errors:>5} ({total_errors/n*100:.1f}%)")

    # Cleaning impact
    cleaning_helped = sum(1 for r in results if r["em"] > 0 and r.get("raw_em", 0) == 0)
    cleaning_hurt = sum(1 for r in results if r["em"] == 0 and r.get("raw_em", 0) > 0)
    print("\n  Cleaning impact:")
    print(f"    Cleaning helped (raw_em=0 -> em=1): {cleaning_helped}")
    print(f"    Cleaning hurt  (raw_em=1 -> em=0):  {cleaning_hurt}")

    # Containment gap (could be correct with better extraction)
    contained_but_wrong = sum(
        1 for r in results
        if r.get("containment", 0) > 0 and r["em"] == 0 and not r.get("error")
    )
    print("\n  Extraction gap:")
    print(f"    Gold in raw but EM=0: {contained_but_wrong} "
          f"(potential for better extraction)")

    # By question type
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)

    print("\n  By question type:")
    print(f"    {'Type':<20} {'N':>5} {'EM':>8} {'F1':>8} {'Cont':>8} {'Err':>5}")
    print(f"    {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for qtype, recs in sorted(by_type.items()):
        k = len(recs)
        em = sum(r["em"] for r in recs) / k
        f1 = sum(r["f1"] for r in recs) / k
        cont = sum(r.get("containment", 0) for r in recs) / k
        errs = sum(1 for r in recs if r.get("error"))
        print(f"    {qtype:<20} {k:>5} {em:>7.1%} {f1:>7.1%} {cont:>7.1%} {errs:>5}")

    print(f"{'=' * 75}")


def save_summary_json(results: list[dict], output_path: str):
    """Save summary as JSON file."""
    n = len(results)
    if n == 0:
        return

    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)

    def agg(recs):
        k = len(recs)
        d = {
            "n": k,
            "em": round(sum(r["em"] for r in recs) / k, 4),
            "f1": round(sum(r["f1"] for r in recs) / k, 4),
            "containment": round(sum(r.get("containment", 0) for r in recs) / k, 4),
            "answered": sum(1 for r in recs if r.get("predicted_answer")),
            "errors": sum(1 for r in recs if r.get("error")),
        }
        if any("judge_correct" in r for r in recs):
            judged = [r for r in recs if "judge_correct" in r]
            d["judge_em"] = round(sum(1 for r in judged if r["judge_correct"]) / k, 4)
        return d

    summary = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "total": n,
        "overall": agg(results),
        "by_type": {qt: agg(recs) for qt, recs in sorted(by_type.items())},
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluation of GRASP results"
    )
    parser.add_argument("results_file", help="Path to GRASP results JSONL")
    parser.add_argument("--judge", action="store_true",
                        help="Run LLM judge for semantic equivalence (requires Ollama)")
    parser.add_argument("--judge-model", default="qwen2.5:7b",
                        help="Ollama model for judging (default: qwen2.5:7b)")
    parser.add_argument("--errors", action="store_true",
                        help="Show detailed error analysis")
    parser.add_argument("--output", default=None,
                        help="Output summary JSON path (default: auto from input)")
    args = parser.parse_args()

    # Load and re-score
    results = load_results(args.results_file)
    results = rescore(results)

    # LLM judge
    if args.judge:
        print(f"\nRunning LLM judge (model={args.judge_model})...")
        results = run_judge(results, model=args.judge_model)

        # Save judged JSONL
        judged_path = args.results_file.replace(".jsonl", "_judged.jsonl")
        with open(judged_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Judged results saved to {judged_path}")

    # Print summary
    print_summary(results)

    # Error analysis
    if args.errors:
        print_errors(results)

    # Save summary JSON
    output_path = args.output
    if not output_path:
        output_path = args.results_file.replace(".jsonl", "_summary.json")
    save_summary_json(results, output_path)


if __name__ == "__main__":
    main()
