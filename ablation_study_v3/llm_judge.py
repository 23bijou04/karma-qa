"""LLM-as-judge re-evaluation of ablation study results.

Uses a local Ollama model to judge semantic equivalence between
predicted and gold answers, bypassing strict EM limitations.

Usage:
    python3 -m ablation_study_v3.llm_judge --input-dir data/ablation_v3 --levels 0 1
    python3 -m ablation_study_v3.llm_judge --input-dir data/ablation_v3 --levels 0 1 --model qwen2.5:3b
"""

import argparse
import json
import os
import time

import requests

JUDGE_PROMPT = """You are judging whether a predicted answer is semantically equivalent to the gold (correct) answer for a question-answering task.

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


def judge_one(question: str, gold: str, pred: str, model: str) -> bool:
    """Ask the local LLM to judge semantic equivalence."""
    if not pred or pred.strip() == "":
        return False

    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 10}},
            timeout=30,
        )
        answer = resp.json().get("response", "").strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"  Judge error: {e}")
        return False


def load_questions(input_dir: str) -> dict[str, dict]:
    """Load sample questions keyed by _id for question text lookup."""
    questions = {}
    for f in os.listdir(input_dir):
        if f.startswith("sample_") and f.endswith(".json"):
            with open(os.path.join(input_dir, f)) as fh:
                for q in json.load(fh):
                    questions[q["_id"]] = q
    return questions


def judge_level(input_dir: str, level: int, model: str,
                questions: dict[str, dict]) -> dict:
    """Re-judge all results for one level."""
    path = os.path.join(input_dir, f"level_{level}_results.jsonl")
    if not os.path.exists(path):
        print(f"  No results file for level {level}")
        return {}

    results = []
    with open(path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(results)
    strict_em = sum(1 for r in results if r["em"] > 0)
    judge_correct = 0
    upgraded = []

    for i, r in enumerate(results, 1):
        qid = r["question_id"]
        gold = r["gold_answer"]
        pred = r["predicted_answer"]

        # Already correct by strict EM — no need to judge
        if r["em"] > 0:
            judge_correct += 1
            r["judge_correct"] = True
            continue

        # Skip empty predictions
        if not pred:
            r["judge_correct"] = False
            continue

        # Get question text
        q_text = questions.get(qid, {}).get("question", "")

        is_correct = judge_one(q_text, gold, pred, model)
        r["judge_correct"] = is_correct

        if is_correct:
            judge_correct += 1
            upgraded.append((pred[:40], gold[:40], q_text[:60]))

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] strict_EM={strict_em} judge_EM={judge_correct}")

    # Save judged results
    out_path = os.path.join(input_dir, f"level_{level}_judged.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "level": level,
        "total": total,
        "strict_em": strict_em,
        "strict_em_pct": strict_em / total * 100 if total else 0,
        "judge_em": judge_correct,
        "judge_em_pct": judge_correct / total * 100 if total else 0,
        "upgraded": judge_correct - strict_em,
    }

    print(f"\n  Level {level}: strict_EM={strict_em}/{total} ({summary['strict_em_pct']:.1f}%)"
          f" → judge_EM={judge_correct}/{total} ({summary['judge_em_pct']:.1f}%)"
          f" [+{summary['upgraded']} upgraded]")

    if upgraded:
        print(f"  Upgraded examples:")
        for pred, gold, q in upgraded[:10]:
            print(f"    pred='{pred}' gold='{gold}'")
            print(f"      Q: {q}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge re-evaluation")
    parser.add_argument("--input-dir", default="data/ablation_v3",
                        help="Directory with level_*_results.jsonl files")
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Which levels to re-judge")
    parser.add_argument("--model", default="qwen2.5:7b",
                        help="Ollama model for judging")
    args = parser.parse_args()

    # Verify Ollama is running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("Error: Ollama is not running. Start it with: ollama serve")
        return

    print(f"LLM Judge: model={args.model}, input={args.input_dir}")
    print(f"Levels: {args.levels}\n")

    questions = load_questions(args.input_dir)
    print(f"Loaded {len(questions)} sample questions\n")

    all_summaries = {}
    for level in args.levels:
        print(f"{'=' * 60}")
        print(f"  Judging Level {level}")
        print(f"{'=' * 60}")
        summary = judge_level(args.input_dir, level, args.model, questions)
        if summary:
            all_summaries[f"level_{level}"] = summary
        print()

    # Save combined summary (merge with existing)
    if all_summaries:
        summary_path = os.path.join(args.input_dir, "judge_summary.json")
        existing = {}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                existing = json.load(f)
        existing.update(all_summaries)
        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Summary saved to: {summary_path}")

        # Print comparison table (all levels, including previously judged)
        print(f"\n{'=' * 60}")
        print("  Comparison: Strict EM vs LLM Judge")
        print(f"{'=' * 60}")
        print(f"{'Level':>8} | {'N':>4} | {'Strict EM':>10} | {'Judge EM':>10} | {'Gained':>6}")
        print(f"{'-'*8}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
        for key, s in sorted(existing.items()):
            print(f"  L{s['level']:>5} | {s['total']:>4} | "
                  f"{s['strict_em_pct']:>8.1f}% | "
                  f"{s['judge_em_pct']:>8.1f}% | "
                  f"+{s['upgraded']:>4}")


if __name__ == "__main__":
    main()
