#!/usr/bin/env python3
"""Rerun specific failed questions and replace their records in the JSONL.

Usage:
    python scripts/rerun_failed_text.py --api-key $GROQ_API_KEY
"""

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from dbpedia_qa.dbpedia import (
    DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery, WikipediaResolver,
)
from dbpedia_qa.models import extract_final_answer
from llm_provider import create_llm_client

# Load metrics
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment

from hybrid_qa import hybrid_answer  # noqa: E402

QUESTION_TIMEOUT = 300
EVAL_FILE = "data/eval_1000q/hybrid_eval_1000.jsonl"
LOGICRAG_CORPUS = os.path.join(
    PROJECT_ROOT, "baselines", "LogicRAG", "dataset", "2wikimultihopqa_corpus.json"
)

_HEDGE_RE = re.compile(
    r"(?:not sure|unclear|cannot determine|could not find|no information)", re.I,
)


class _QuestionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _QuestionTimeout(f"Question exceeded {QUESTION_TIMEOUT}s timeout")


def clean_answer(raw: str) -> str:
    if not raw:
        return ""
    answer = extract_final_answer(raw)
    for prefix in ["The answer is ", "Answer: ", "Based on the text, "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    return answer.strip().rstrip(".")


def main():
    from sparql_evaluation import force_ipv4
    force_ipv4()

    parser = argparse.ArgumentParser(description="Rerun failed TEXT-path questions")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--provider", default="groq", choices=["groq", "github"])
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--mode", default="text",
                        choices=["text", "logicrag", "all-wrong"],
                        help="Which failures to rerun")
    args = parser.parse_args()

    # Step 1: Load existing results and identify failures to rerun
    records = []
    with open(EVAL_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    failed_ids = set()
    failed_questions = {}
    for r in records:
        pred = r.get("predicted_answer", "").lower()
        if args.mode == "text":
            # Original: TEXT-path with no answer
            is_fail = (r["method"] == "text"
                       and pred in ("", "could not find answer", "unknown", "not found"))
        elif args.mode == "logicrag":
            # LogicRAG/TEXT failures: EM=0 and F1<0.5
            is_fail = (r.get("em", 0) == 0
                       and r.get("f1", 0) < 0.5
                       and r["method"] in ("logicrag_decomposed", "text"))
        else:  # all-wrong
            is_fail = r.get("em", 0) == 0 and r.get("f1", 0) < 0.5
        if is_fail:
            failed_ids.add(r["question_id"])
            failed_questions[r["question_id"]] = r

    print(f"Found {len(failed_ids)} failures to rerun (mode={args.mode})")

    # Step 2: Create shared clients
    llm_client = create_llm_client(args.provider, api_key=args.api_key)
    linker = DBpediaEntityLinker(
        min_confidence=0.5, llm_client=llm_client, llm_model=args.model,
        wiki_resolver=WikipediaResolver(),
    )
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()

    # Load benchmark corpus
    with open(LOGICRAG_CORPUS) as f:
        benchmark_corpus = json.load(f)
    print(f"Loaded benchmark corpus: {len(benchmark_corpus)} entries")

    # Step 3: Rerun each failed question
    new_records = {}
    em_sum, f1_sum = 0.0, 0.0

    for i, qid in enumerate(sorted(failed_ids)):
        orig = failed_questions[qid]
        question = orig["question"]
        gold = orig["gold_answer"]
        qtype = orig["question_type"]
        is_answerable = orig["is_answerable"]

        t0 = time.time()
        record = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold,
            "is_answerable": is_answerable,
        }

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(QUESTION_TIMEOUT)
        try:
            result = hybrid_answer(
                question, args.api_key, model=args.model,
                llm_client=llm_client, linker=linker,
                kg_client=kg_client, schema_disc=schema_disc,
                benchmark_corpus=benchmark_corpus,
            )
            record["raw_answer"] = result.answer
            record["predicted_answer"] = clean_answer(result.answer)
            record["method"] = result.method
            record["route_source"] = result.route_source
            record["sparql_fallback"] = result.sparql_fallback
            record["sparql_query"] = getattr(result, "sparql_query", None) or ""
            record["kg_context"] = getattr(result, "kg_context", None) or ""
            if result.sub_question_log:
                record["sub_question_log"] = result.sub_question_log
            record["debug_info"] = getattr(result, "debug_info", {})
            record["has_hedging"] = bool(_HEDGE_RE.search(record["raw_answer"]))
            record["error"] = ""
        except _QuestionTimeout:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "timeout"
            record["route_source"] = ""
            record["sparql_fallback"] = False
            record["sparql_query"] = ""
            record["kg_context"] = ""
            record["debug_info"] = {}
            record["has_hedging"] = False
            record["error"] = f"Timeout after {QUESTION_TIMEOUT}s"
            print(f"  !! TIMEOUT")
        except Exception:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "error"
            record["route_source"] = ""
            record["sparql_fallback"] = False
            record["sparql_query"] = ""
            record["kg_context"] = ""
            record["debug_info"] = {}
            record["has_hedging"] = False
            record["error"] = traceback.format_exc()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        pred = record["predicted_answer"]
        record["em"] = compute_em(pred, gold) if pred else 0.0
        record["f1"] = compute_f1(pred, gold) if pred else 0.0
        record["containment"] = compute_containment(
            record["raw_answer"], gold
        ) if record["raw_answer"] else 0.0
        record["latency_seconds"] = round(time.time() - t0, 2)

        new_records[qid] = record
        em_sum += record["em"]
        f1_sum += record["f1"]

        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        if record["containment"] > 0 and record["em"] == 0:
            em_tag += " cont=1"
        print(
            f"[{i+1}/{len(failed_ids)}] {question[:60]}...\n"
            f"  -> gold='{gold}' pred='{pred[:60]}' [{em_tag}] "
            f"method={record['method']} {record['latency_seconds']}s"
        )

        if i < len(failed_ids) - 1:
            time.sleep(args.delay)

    # Step 4: Replace records in JSONL (preserve order)
    updated = []
    for r in records:
        if r["question_id"] in new_records:
            updated.append(new_records[r["question_id"]])
        else:
            updated.append(r)

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    n = len(new_records)
    print(f"\n{'='*60}")
    print(f"Reran {n} questions")
    print(f"  EM:  {em_sum/n*100:.1f}%")
    print(f"  F1:  {f1_sum/n*100:.1f}%")
    answered = sum(1 for r in new_records.values() if r["predicted_answer"]
                   and r["predicted_answer"].lower() != "could not find answer")
    print(f"  Answered: {answered}/{n} ({answered/n*100:.0f}%)")
    print(f"\nUpdated: {EVAL_FILE}")


if __name__ == "__main__":
    main()
