#!/usr/bin/env python3
"""Rerun Hybrid QA on HotPotQA failures to measure fix recovery.

Loads failure IDs from hybrid_hotpotqa_failures.jsonl, reruns through
hybrid_answer() with the same benchmark_corpus, and compares old vs new.

Usage:
    python scripts/rerun-hotpotqa-failures.py --api-key $GROQ_API_KEY --limit 20
    python scripts/rerun-hotpotqa-failures.py --api-key $GROQ_API_KEY  # all 158
"""

import argparse
import importlib.util
import json
import os
import signal
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from dbpedia_qa.dbpedia import (  # noqa: E402
    DBpediaEntityLinker,
    DBpediaSPARQLClient,
    SchemaDiscovery,
    WikipediaResolver,
)
from dbpedia_qa.models import extract_final_answer  # noqa: E402

from hybrid_qa import hybrid_answer  # noqa: E402
from llm_provider import create_llm_client  # noqa: E402
from sparql_evaluation import force_ipv4  # noqa: E402

# Load metrics
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment

LOGICRAG_DIR = os.path.join(PROJECT_ROOT, "baselines", "LogicRAG")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cross_benchmark", "hotpotqa")
FAILURES_FILE = os.path.join(DATA_DIR, "hybrid_hotpotqa_failures.jsonl")
QUESTION_TIMEOUT = 300


class _QuestionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _QuestionTimeout(f"Question exceeded {QUESTION_TIMEOUT}s timeout")


def load_failures() -> list[dict]:
    """Load failure records with question_id, question, gold_answer."""
    failures = []
    with open(FAILURES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                failures.append(json.loads(line))
    return failures


def load_benchmark_corpus() -> list[dict]:
    """Load HotPotQA corpus from LogicRAG dataset."""
    path = os.path.join(LOGICRAG_DIR, "dataset", "hotpotqa_corpus.json")
    with open(path) as f:
        corpus = json.load(f)
    print(f"Loaded corpus: {len(corpus)} entries")
    return corpus


def load_completed_ids(output_path: str) -> set[str]:
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
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


def clean_answer(raw: str) -> str:
    if not raw:
        return ""
    answer = extract_final_answer(raw)
    for prefix in ["The answer is ", "Answer: ", "Based on the text, "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    return answer.strip().rstrip(".")


def main():
    force_ipv4()
    parser = argparse.ArgumentParser(description="Rerun HotPotQA failures")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--provider", default="groq", choices=["groq", "github", "ollama"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--failures", default=None, help="Custom failures JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Max failures to rerun (0=all)")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.join(DATA_DIR, "hybrid_hotpotqa_failures_rerun.jsonl")

    if args.failures:
        # Load from custom file instead of default
        failures = []
        with open(args.failures) as f:
            for line in f:
                line = line.strip()
                if line:
                    failures.append(json.loads(line))
    else:
        failures = load_failures()
    corpus = load_benchmark_corpus()

    # Resume support
    done_ids = load_completed_ids(args.output)
    remaining = [f for f in failures if f.get("id", f.get("question_id")) not in done_ids]
    if done_ids:
        print(f"RESUME: {len(done_ids)} done, {len(remaining)} remaining")
    if args.limit > 0:
        remaining = remaining[:args.limit]
    if not remaining:
        print("All failures already rerun.")
        return

    print(f"\n=== Rerunning {len(remaining)}/{len(failures)} HotPotQA failures ===")
    print(f"LLM: {args.model} ({args.provider})")
    print(f"Output: {args.output}\n")

    llm_client = create_llm_client(args.provider, api_key=args.api_key)
    linker = DBpediaEntityLinker(
        min_confidence=0.5, llm_client=llm_client, llm_model=args.model,
        wiki_resolver=WikipediaResolver(),
    )
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Tracking
    recovered, still_failed, errors = 0, 0, 0
    start = time.time()

    for i, fail_rec in enumerate(remaining):
        qid = fail_rec.get("id", fail_rec.get("question_id"))
        question = fail_rec["question"]
        gold = fail_rec["gold_answer"]
        old_method = fail_rec.get("method", "unknown")
        old_answer = fail_rec.get("predicted_answer", "")

        print(f"[{i+1}/{len(remaining)}] {qid[:12]}... ", end="", flush=True)

        record = {
            "question_id": qid,
            "question": question,
            "gold_answer": gold,
            "old_method": old_method,
            "old_answer": old_answer,
        }

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(QUESTION_TIMEOUT)
        try:
            result = hybrid_answer(
                question, args.api_key, model=args.model,
                llm_client=llm_client, linker=linker,
                kg_client=kg_client, schema_disc=schema_disc,
                benchmark_corpus=corpus,
            )
            record["new_answer"] = clean_answer(result.answer)
            record["new_method"] = result.method
            record["new_route_source"] = result.route_source
            if result.sub_question_log:
                record["sub_question_log"] = result.sub_question_log
            record["error"] = ""

            # Compute metrics
            aliases = fail_rec.get("gold_aliases", [])
            candidates = [gold] + (aliases or [])
            best_em = max(compute_em(record["new_answer"], c) for c in candidates if c)
            best_f1 = max(compute_f1(record["new_answer"], c) for c in candidates if c)
            best_containment = max(
                compute_containment(record["new_answer"], c) for c in candidates if c
            )
            record["em"] = best_em
            record["f1"] = best_f1
            record["containment"] = best_containment

            if best_em > 0:
                recovered += 1
                tag = "RECOVERED"
            else:
                still_failed += 1
                tag = "still_failed"

            print(f"{tag} | {old_method}->{result.method} | "
                  f"EM={best_em:.0f} F1={best_f1:.2f} | "
                  f"'{record['new_answer'][:40]}' vs '{gold[:40]}'")

        except _QuestionTimeout:
            record["new_answer"] = ""
            record["new_method"] = "timeout"
            record["error"] = "timeout"
            errors += 1
            still_failed += 1
            print("TIMEOUT")
        except Exception as exc:
            record["new_answer"] = ""
            record["new_method"] = "error"
            record["error"] = str(exc)[:200]
            errors += 1
            still_failed += 1
            print(f"ERROR: {exc}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        with open(args.output, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if args.delay and i < len(remaining) - 1:
            time.sleep(args.delay)

    elapsed = time.time() - start
    total = recovered + still_failed
    print(f"\n{'='*60}")
    print(f"RESULTS: {recovered}/{total} recovered ({recovered/total*100:.1f}%)")
    print(f"  Still failed: {still_failed}, Errors: {errors}")
    print(f"  Elapsed: {elapsed/60:.1f}min ({elapsed/total:.1f}s/q)")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
