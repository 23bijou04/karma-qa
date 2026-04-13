#!/usr/bin/env python3
"""Unified experiment runner: batch eval → judge → comparison in one command.

Inspired by LinearRAG's single-command experiment flow (run.py → evaluate).
Chains the three evaluation stages with resume support at each stage.

Usage:
    # Full pipeline (eval + judge + compare)
    python scripts/run-full-pipeline.py --api-key $GROQ_API_KEY --limit 5

    # Skip to judging (eval already done)
    python scripts/run-full-pipeline.py --skip-eval

    # Skip to comparison (eval + judge already done)
    python scripts/run-full-pipeline.py --skip-eval --skip-judge

    # Cross-benchmark mode
    python scripts/run-full-pipeline.py --benchmark hotpotqa --api-key $GROQ_API_KEY
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime


def run_stage(name, cmd, dry_run=False):
    """Run a pipeline stage, printing the command and timing it.

    Why a subprocess instead of importing? Each stage is a standalone script
    with its own argparse, signal handlers, and global state. Running them
    as subprocesses keeps isolation clean and lets you Ctrl+C mid-stage
    without corrupting the parent process.
    """
    print(f"\n{'=' * 60}")
    print(f"STAGE: {name}")
    print(f"CMD:   {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return 0

    t0 = time.time()
    # Use the same Python interpreter that's running this script
    result = subprocess.run(cmd, cwd=None)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! Stage '{name}' failed with exit code {result.returncode}")
        print(f"  !! Elapsed: {elapsed / 60:.1f}min")
        print("  !! Fix the issue and re-run — resume support will skip completed work.")
        return result.returncode

    print(f"\n  Stage '{name}' completed in {elapsed / 60:.1f}min")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: batch eval → judge → comparison",
        # Show full help for all stages
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1. Batch eval   — Run hybrid_answer() on all questions (hybrid_batch_eval.py)
  2. Judge         — LLM-as-judge semantic equivalence (judge_hybrid_eval.py)
  3. Compare       — Cross-system comparison tables (unified_comparison.py)

Each stage is resume-safe: re-running skips already-completed questions.
Use --skip-eval and --skip-judge to start from a later stage.
        """,
    )

    # Shared args
    parser.add_argument("--api-key", default=None, help="API key for LLM provider")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="LLM model (default: llama-3.3-70b-versatile)")
    parser.add_argument("--provider", default="groq", choices=["groq", "github", "ollama"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--limit", type=int, default=0, help="Max questions for eval (0=all)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between questions during eval (default: 2)")

    # Stage control
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip batch eval (assume results exist)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judging (assume judged results exist)")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Skip comparison stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")

    # Cross-benchmark mode
    parser.add_argument("--benchmark", default="2wikimultihop",
                        choices=["2wikimultihop", "hotpotqa", "musique"],
                        help="Benchmark dataset (default: 2wikimultihop)")

    # Judge args
    parser.add_argument("--judge-provider", default="ollama",
                        choices=["ollama", "groq", "github"],
                        help="Judge LLM provider (default: ollama)")
    parser.add_argument("--judge-model", default="gemma3:12b",
                        help="Judge LLM model (default: gemma3:12b)")

    # File overrides
    parser.add_argument("--eval-output", default=None,
                        help="Override eval output JSONL path")
    args = parser.parse_args()

    print("=== Hybrid QA Full Pipeline ===")
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"LLM:        {args.model} ({args.provider})")
    print(f"Judge:      {args.judge_model} ({args.judge_provider})")
    if args.limit > 0:
        print(f"Limit:      {args.limit} questions")
    print(f"{'=' * 60}")

    pipeline_start = time.time()

    # Determine if this is cross-benchmark or standard 2WikiMultiHop
    is_cross = args.benchmark != "2wikimultihop"

    # Resolve file paths
    if is_cross:
        eval_output = args.eval_output or (
            f"data/cross_benchmark/{args.benchmark}/"
            f"hybrid_{args.benchmark}_1000q.jsonl"
        )
    else:
        eval_output = args.eval_output or "data/eval_1000q/hybrid/hybrid_eval_1000.jsonl"

    # Use sys.executable so we run the same Python that's executing this script
    python = sys.executable

    # ---------------------------------------------------------------
    # Stage 1: Batch evaluation
    # ---------------------------------------------------------------
    if not args.skip_eval:
        if is_cross:
            # Cross-benchmark uses a different runner script
            eval_cmd = [
                python, "scripts/run-hybrid-cross-benchmark.py",
                "--benchmark", args.benchmark,
                "--provider", args.provider,
                "--model", args.model,
                "--output", eval_output,
                "--delay", str(args.delay),
            ]
        else:
            eval_cmd = [
                python, "scripts/hybrid_batch_eval.py",
                "--provider", args.provider,
                "--model", args.model,
                "--output-file", eval_output,
                "--delay", str(args.delay),
            ]
        if args.api_key:
            eval_cmd += ["--api-key", args.api_key]
        if args.limit > 0:
            eval_cmd += ["--limit", str(args.limit)]

        rc = run_stage("Batch Evaluation", eval_cmd, dry_run=args.dry_run)
        if rc != 0:
            sys.exit(rc)
    else:
        print("\n  [SKIP] Batch evaluation (--skip-eval)")

    # ---------------------------------------------------------------
    # Stage 2: LLM Judge
    # ---------------------------------------------------------------
    if not args.skip_judge:
        judge_cmd = [
            python, "scripts/judge_hybrid_eval.py",
            eval_output,
            "--provider", args.judge_provider,
            "--model", args.judge_model,
        ]

        rc = run_stage("LLM Judge", judge_cmd, dry_run=args.dry_run)
        if rc != 0:
            sys.exit(rc)
    else:
        print("\n  [SKIP] LLM Judge (--skip-judge)")

    # ---------------------------------------------------------------
    # Stage 3: Unified comparison
    # ---------------------------------------------------------------
    if not args.skip_compare and not is_cross:
        # Comparison only makes sense for the main 2WikiMultiHop benchmark
        # where we have multiple systems to compare
        compare_cmd = [python, "scripts/unified_comparison.py"]

        rc = run_stage("Unified Comparison", compare_cmd, dry_run=args.dry_run)
        if rc != 0:
            sys.exit(rc)
    elif is_cross:
        print("\n  [SKIP] Comparison (cross-benchmark — run compute-cross-benchmark-metrics.py)")
    else:
        print("\n  [SKIP] Comparison (--skip-compare)")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"Total time: {total_elapsed / 60:.1f}min")
    print(f"Eval output:  {eval_output}")
    # The judge script creates a _judged.jsonl file next to the eval output
    judged_output = eval_output.replace(".jsonl", "_judged.jsonl")
    print(f"Judge output: {judged_output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
