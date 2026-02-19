#!/usr/bin/env python3
"""Extract LinearRAG's 1K questions from dev_full.json, run DBpedia coverage analysis.

Outputs data/coverage_analysis_1000_linearrag.json with entity URIs and
answerability flags for each question. The answerable subset feeds into
SPARQL (Phase 5) and GRASP (Phase 6) evaluations.

Usage:
    python scripts/extract_linearrag_1k.py
"""

import json
import os
import sys

# Add project root + archive/scripts to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "archive", "scripts"))

from coverage_analysis import CoverageAnalyzer


def main():
    # 1. Load LinearRAG HF IDs
    lr_path = os.path.join(PROJECT_ROOT, "baselines/LinearRAG/dataset/2wikimultihop/questions.json")
    with open(lr_path) as f:
        lr_questions = json.load(f)
    lr_ids = {q["id"] for q in lr_questions}
    print(f"LinearRAG IDs: {len(lr_ids)}")

    # 2. Load dev_full.json
    dev_path = os.path.join(PROJECT_ROOT, "data/dev_full.json")
    with open(dev_path) as f:
        dev_full = json.load(f)
    print(f"dev_full.json: {len(dev_full)} questions")

    # 3. Filter to matching questions
    matched = [q for q in dev_full if q["_id"] in lr_ids]
    print(f"Matched: {len(matched)}")

    if len(matched) != 1000:
        print(f"WARNING: Expected 1000, got {len(matched)}")

    # Type breakdown
    from collections import Counter
    types = Counter(q["type"] for q in matched)
    print(f"Types: {dict(types)}")

    # 4. Run coverage analysis (bypass stratified_sample, feed directly)
    analyzer = CoverageAnalyzer(data_file=dev_path)
    results = analyzer.analyze_batch(matched)
    stats = analyzer.compute_statistics(results)
    analyzer.print_report(stats)

    # 5. Save to eval_1000q directory
    out_dir = os.path.join(PROJECT_ROOT, "data/eval_1000q")
    os.makedirs(out_dir, exist_ok=True)
    output = os.path.join(out_dir, "coverage_analysis_1000.json")
    analyzer.save_results(results, stats, output)

    answerable = [r for r in results if r.is_answerable]
    print(f"\nAnswerable: {len(answerable)}/{len(matched)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
