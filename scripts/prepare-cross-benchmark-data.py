#!/usr/bin/env python3
"""Download and verify cross-benchmark datasets (HotpotQA, MuSiQue) for LinearRAG.

Downloads from HuggingFace Zly0523/linear-rag and verifies question overlap
with LogicRAG's datasets. Idempotent — safe to re-run.

Usage:
    python scripts/prepare-cross-benchmark-data.py [--force]
"""

import argparse
import json
import os
import shutil
import sys

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGICRAG_DATA = os.path.join(PROJECT_ROOT, "baselines", "LogicRAG", "dataset")
LINEARRAG_DATA = os.path.join(PROJECT_ROOT, "baselines", "LinearRAG", "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "cross_benchmark")

BENCHMARKS = ["hotpotqa", "musique"]


def download_from_huggingface(force: bool = False):
    """Download LinearRAG HotpotQA/MuSiQue from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    for bench in BENCHMARKS:
        dest_dir = os.path.join(LINEARRAG_DATA, bench)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in ["questions.json", "chunks.json"]:
            dest_path = os.path.join(dest_dir, fname)
            if os.path.exists(dest_path) and not force:
                print(f"  SKIP {bench}/{fname} (exists, use --force to redownload)")
                continue

            path = hf_hub_download(
                "Zly0523/linear-rag",
                f"{bench}/{fname}",
                repo_type="dataset",
                local_dir="/tmp/linearrag_hf",
            )
            shutil.copy(path, dest_path)
            print(f"  Downloaded {bench}/{fname}")


def verify_logicrag_datasets():
    """Verify LogicRAG datasets exist and have expected counts."""
    print("\n=== LogicRAG Dataset Verification ===")
    for bench in BENCHMARKS:
        qfile = os.path.join(LOGICRAG_DATA, f"{bench}.json")
        cfile = os.path.join(LOGICRAG_DATA, f"{bench}_corpus.json")

        if not os.path.exists(qfile):
            print(f"  MISSING: {qfile}")
            continue

        questions = json.load(open(qfile))
        corpus = json.load(open(cfile))
        print(f"  {bench}: {len(questions)} questions, {len(corpus)} corpus entries")


def verify_overlap():
    """Verify question overlap between LogicRAG and LinearRAG for each benchmark."""
    print("\n=== Question Overlap Verification ===")
    results = {}

    for bench in BENCHMARKS:
        # Load LogicRAG questions
        lr_qs = json.load(open(os.path.join(LOGICRAG_DATA, f"{bench}.json")))
        # LogicRAG uses _id for HotpotQA, id for MuSiQue
        id_key = "_id" if bench == "hotpotqa" else "id"
        lr_ids = {q[id_key] for q in lr_qs}
        lr_answers = {q[id_key]: q["answer"] for q in lr_qs}

        # Load LinearRAG questions
        lin_qs = json.load(
            open(os.path.join(LINEARRAG_DATA, bench, "questions.json"))
        )
        # LinearRAG adds 'musique_' prefix for MuSiQue IDs
        if bench == "musique":
            lin_ids = {q["id"].replace("musique_", "") for q in lin_qs}
            lin_answers = {
                q["id"].replace("musique_", ""): q["answer"] for q in lin_qs
            }
        else:
            lin_ids = {q["id"] for q in lin_qs}
            lin_answers = {q["id"]: q["answer"] for q in lin_qs}

        # Compute overlap
        overlap = lr_ids & lin_ids
        answer_match = sum(
            1 for k in overlap if lr_answers[k] == lin_answers.get(k, "")
        )

        results[bench] = {
            "logicrag_count": len(lr_ids),
            "linearrag_count": len(lin_ids),
            "id_overlap": len(overlap),
            "answer_match": answer_match,
        }

        print(f"  {bench}:")
        print(f"    LogicRAG: {len(lr_ids)}, LinearRAG: {len(lin_ids)}")
        print(f"    ID overlap: {len(overlap)}/{max(len(lr_ids), len(lin_ids))}")
        print(f"    Answer match: {answer_match}/{len(overlap)}")

    return results


def create_output_dirs():
    """Create output directory structure for evaluation results."""
    for bench in BENCHMARKS:
        os.makedirs(os.path.join(OUTPUT_DIR, bench), exist_ok=True)
    print(f"\n  Output dirs: {OUTPUT_DIR}/{{hotpotqa,musique}}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Force redownload from HuggingFace"
    )
    args = parser.parse_args()

    print("=== Cross-Benchmark Data Preparation ===")
    download_from_huggingface(force=args.force)
    verify_logicrag_datasets()
    results = verify_overlap()
    create_output_dirs()

    # Summary
    print("\n=== Summary ===")
    for bench in BENCHMARKS:
        r = results[bench]
        status = "OK" if r["id_overlap"] == r["logicrag_count"] == 1000 else "WARN"
        print(f"  [{status}] {bench}: {r['id_overlap']}/1000 overlap, {r['answer_match']}/1000 answers match")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
