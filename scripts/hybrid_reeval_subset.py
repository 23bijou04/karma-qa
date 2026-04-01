#!/usr/bin/env python3
"""Re-evaluate hybrid pipeline on a subset of question IDs.

Reuses hybrid_batch_eval machinery but filters to only specified IDs.
Useful for targeted re-eval after code changes (e.g., Phase 1 fixes).

Usage:
    python scripts/hybrid_reeval_subset.py \
        --ids-file data/eval_1000q/hybrid/phase1_reeval_147_ids.json \
        --output-file data/eval_1000q/hybrid/phase1_reeval_147.jsonl \
        --api-key $GROQ_API_KEY
"""

import argparse
import json
import os
import re
import signal
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Reuse metrics loader from batch eval
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_em = _metrics.compute_em
compute_f1 = _metrics.compute_f1
compute_containment = _metrics.compute_containment

from dbpedia_qa.dbpedia import (
    DBpediaEntityLinker, DBpediaSPARQLClient, SchemaDiscovery, WikipediaResolver,
)
from dbpedia_qa.models import extract_final_answer
from hybrid_qa import hybrid_answer
from llm_provider import create_llm_client
from sparql_evaluation import force_ipv4

QUESTION_TIMEOUT = 300
LOGICRAG_CORPUS = os.path.join(
    PROJECT_ROOT, "baselines", "LogicRAG", "dataset", "2wikimultihopqa_corpus.json"
)
DEFAULT_COVERAGE = "data/eval_1000q/ground-truth/coverage_analysis_1000.json"


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
    force_ipv4()
    parser = argparse.ArgumentParser(description="Re-eval hybrid on subset of IDs")
    parser.add_argument("--ids-file", required=True,
                        help="JSON file with list of question IDs to re-eval")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--provider", default="groq", choices=["groq", "github", "ollama"])
    parser.add_argument("--coverage-file", default=DEFAULT_COVERAGE)
    parser.add_argument("--output-file", required=True,
                        help="JSONL output file for re-eval results")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    # Load target IDs
    with open(args.ids_file) as f:
        target_ids = set(json.load(f))
    print(f"Target: {len(target_ids)} question IDs from {args.ids_file}")

    # Load all questions, filter to target IDs
    with open(args.coverage_file) as f:
        data = json.load(f)
    questions = [q for q in data["questions"] if q["question_id"] in target_ids]
    print(f"Matched: {len(questions)} questions from coverage file")

    # Resume support
    done_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("question_id"):
                            done_ids.add(rec["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if done_ids:
        print(f"RESUME: {len(done_ids)} done, {len(remaining)} remaining")

    if not remaining:
        print("All target questions already processed.")
        return

    # Create shared clients
    print("Creating shared clients...")
    llm_client = create_llm_client(args.provider, api_key=args.api_key)
    linker = DBpediaEntityLinker(
        min_confidence=0.5, llm_client=llm_client, llm_model=args.model,
        wiki_resolver=WikipediaResolver(),
    )
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()

    # Load benchmark corpus
    benchmark_corpus = None
    if os.path.exists(LOGICRAG_CORPUS):
        with open(LOGICRAG_CORPUS) as f:
            benchmark_corpus = json.load(f)
        print(f"Loaded benchmark corpus: {len(benchmark_corpus)} entries")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    _HEDGE_RE = re.compile(
        r"(?:not sure|unclear|cannot determine|could not find|no information)", re.I,
    )

    em_sum, f1_sum = 0.0, 0.0
    t_start = time.time()

    print(f"\n--- Re-evaluating {len(remaining)} questions (delay={args.delay}s) ---\n")

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]
        qtype = q["question_type"]

        t0 = time.time()
        record = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold,
            "is_answerable": q.get("is_answerable", False),
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
            print(f"  !! TIMEOUT after {QUESTION_TIMEOUT}s")
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

        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        em_sum += record["em"]
        f1_sum += record["f1"]
        done = i + 1

        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        elapsed = time.time() - t_start
        eta = (elapsed / done) * (len(remaining) - done) if done > 0 else 0
        print(
            f"[{len(done_ids)+done}/{len(target_ids)}] "
            f"{question[:60]}..."
        )
        print(
            f"  -> gold='{gold}' pred='{pred[:60]}' [{em_tag}] "
            f"method={record['method']} {record['latency_seconds']}s"
        )
        if done % 25 == 0:
            recovered = int(em_sum)
            print(
                f"  --- Running: {recovered}/{done} recovered (EM>0) "
                f"| Elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min ---"
            )

        if i < len(remaining) - 1:
            time.sleep(args.delay)

    # Final summary
    elapsed = time.time() - t_start
    total = len(done_ids) + len(remaining)
    recovered = int(em_sum)
    print(f"\n{'='*60}")
    print(f"Re-eval complete: {total} questions in {elapsed/60:.1f}min")
    print(f"Recovered (EM>0): {recovered}/{len(remaining)} ({recovered/len(remaining)*100:.1f}%)")
    print(f"Still failing: {len(remaining)-recovered}/{len(remaining)}")
    print(f"EM={em_sum/len(remaining)*100:.1f}% F1={f1_sum/len(remaining)*100:.1f}%")
    print(f"Results: {args.output_file}")


if __name__ == "__main__":
    main()
