#!/usr/bin/env python3
"""Extract 100q stratified sample from 1000q eval set for backbone ablation.

Outputs:
  - data/eval_1000q/ablation/gemma3_100q_sample.json  (question list)
  - data/eval_1000q/ablation/gemma3_100q_coverage.json (hybrid_batch_eval compatible)
  - baselines/LinearRAG/dataset/ablation_100q/         (LinearRAG compatible)

Sampling: 25 per question type from the 1000q eval set, seed=42.

NOTE: We sample from the 1000q eval set (not ablation_study_v3) because all
three systems (Hybrid, LogicRAG, LinearRAG) share this dataset. The ablation_v3
sample has only 17/200 overlap with the 1000q set.
"""

import json
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COVERAGE_FILE = os.path.join(
    PROJECT_ROOT, "data", "eval_1000q", "ground-truth",
    "coverage_analysis_1000_corrected.json",
)
DEV_FULL = os.path.join(PROJECT_ROOT, "data", "dev_full.json")
LINEARRAG_FULL = os.path.join(
    PROJECT_ROOT, "baselines", "LinearRAG", "dataset", "2wikimultihop",
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "eval_1000q", "ablation")
LINEARRAG_OUT = os.path.join(
    PROJECT_ROOT, "baselines", "LinearRAG", "dataset", "ablation_100q",
)

SEED = 42
N_PER_TYPE = 25


def main():
    # Load 1000q eval set
    with open(COVERAGE_FILE) as f:
        coverage = json.load(f)
    questions = coverage["questions"]
    print(f"Loaded {len(questions)} questions from eval set")

    # Stratified sample: 25 per type
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)

    rng = random.Random(SEED)
    sample = []
    for qtype in sorted(by_type):
        pool = by_type[qtype]
        n = min(N_PER_TYPE, len(pool))
        selected = rng.sample(pool, n)
        sample.extend(selected)
        print(f"  {qtype}: {n}/{len(pool)}")

    sample_ids = {q["question_id"] for q in sample}
    print(f"Total sample: {len(sample)} questions")

    # Save sample
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_path = os.path.join(OUTPUT_DIR, "gemma3_100q_sample.json")
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Saved sample: {sample_path}")

    # Create coverage-compatible JSON for hybrid_batch_eval.py
    coverage_subset = {
        "metadata": {
            "source": "backbone_ablation_100q",
            "seed": SEED,
            "n_per_type": N_PER_TYPE,
            "total": len(sample),
        },
        "questions": sample,
    }
    coverage_path = os.path.join(OUTPUT_DIR, "gemma3_100q_coverage.json")
    with open(coverage_path, "w") as f:
        json.dump(coverage_subset, f, indent=2, ensure_ascii=False)
    print(f"Saved coverage file: {coverage_path}")

    # Create LinearRAG-compatible dataset
    # Load full LinearRAG dataset and filter to sample IDs
    lr_q_path = os.path.join(LINEARRAG_FULL, "questions.json")

    with open(lr_q_path) as f:
        lr_questions = json.load(f)

    # Filter questions
    lr_filtered = [q for q in lr_questions if q["id"] in sample_ids]
    print(f"LinearRAG questions: {len(lr_filtered)}/{len(sample)}")

    # Build chunks from dev_full.json context (need fresh chunks for 100q subset)
    with open(DEV_FULL) as f:
        dev_data = json.load(f)
    dev_by_id = {q["_id"]: q for q in dev_data}

    # Collect all context paragraphs for the 100q, deduplicated by title
    seen_titles = set()
    chunks = []
    for q in lr_filtered:
        full_q = dev_by_id.get(q["id"])
        if not full_q:
            continue
        for title, sents in full_q["context"]:
            if title not in seen_titles:
                seen_titles.add(title)
                text = " ".join(sents).lower()
                chunks.append(f"{len(chunks)}:{text}")

    print(f"LinearRAG chunks: {len(chunks)} (deduplicated by title)")

    os.makedirs(LINEARRAG_OUT, exist_ok=True)
    with open(os.path.join(LINEARRAG_OUT, "questions.json"), "w") as f:
        json.dump(lr_filtered, f, indent=2, ensure_ascii=False)
    with open(os.path.join(LINEARRAG_OUT, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved LinearRAG dataset: {LINEARRAG_OUT}")

    # Summary
    print("\n--- Ablation 100q sample ready ---")
    print(f"  Sample: {sample_path}")
    print(f"  Hybrid coverage: {coverage_path}")
    print(f"  LinearRAG dataset: {LINEARRAG_OUT}")
    print("  LogicRAG: uses shared corpus (all 100q in LogicRAG dataset)")


if __name__ == "__main__":
    main()
