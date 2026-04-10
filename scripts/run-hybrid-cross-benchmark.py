#!/usr/bin/env python3
"""Batch Hybrid QA evaluation on HotpotQA and MuSiQue benchmarks.

Loads questions/corpus from LogicRAG dataset format, runs hybrid_answer()
with benchmark_corpus, outputs JSONL with per-question metrics.
Logs KG hit rate per benchmark.

Usage:
    python scripts/run-hybrid-cross-benchmark.py --api-key $GROQ_API_KEY --benchmark hotpotqa --limit 5
    python scripts/run-hybrid-cross-benchmark.py --api-key $GROQ_API_KEY --benchmark musique
"""

import argparse
import importlib.util
import json
import os
import re
import signal
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

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
QUESTION_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Tee logger — writes every print to both stdout and a log file
# ---------------------------------------------------------------------------
_log_file = None


def log(msg=""):
    """Print to stdout and append to log file if open."""
    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()


def _open_log(benchmark: str, output_path: str):
    """Open log file next to the JSONL output. Returns file handle."""
    log_dir = os.path.dirname(output_path) or "."
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"hybrid_{benchmark}_run_{timestamp}.log")
    fh = open(log_path, "a", encoding="utf-8")
    fh.write(f"=== Log started {datetime.now().isoformat()} ===\n")
    fh.flush()
    return fh, log_path


class _QuestionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _QuestionTimeout(f"Question exceeded {QUESTION_TIMEOUT}s timeout")


def load_benchmark_questions(benchmark: str) -> list[dict]:
    """Load questions from LogicRAG dataset, normalized to unified format."""
    path = os.path.join(LOGICRAG_DIR, "dataset", f"{benchmark}.json")
    with open(path) as f:
        raw = json.load(f)

    questions = []
    for entry in raw:
        if benchmark == "hotpotqa":
            questions.append({
                "question_id": entry["_id"],
                "question": entry["question"],
                "gold_answer": entry["answer"],
                "gold_aliases": [],
                "question_type": entry.get("type", "unknown"),
            })
        elif benchmark == "musique":
            if not entry.get("answerable", True):
                continue
            qid = entry["id"]
            qtype = qid.split("__")[0] if "__" in qid else "unknown"
            questions.append({
                "question_id": qid,
                "question": entry["question"],
                "gold_answer": entry["answer"],
                "gold_aliases": entry.get("answer_aliases", []),
                "question_type": qtype,
            })
    return questions


def load_benchmark_corpus(benchmark: str) -> list[dict]:
    """Load corpus from LogicRAG dataset (title/text dicts)."""
    path = os.path.join(LOGICRAG_DIR, "dataset", f"{benchmark}_corpus.json")
    with open(path) as f:
        corpus = json.load(f)
    log(f"Loaded corpus: {len(corpus)} entries from {path}")
    return corpus


def compute_best_metrics(pred: str, gold: str, aliases: list) -> dict:
    """Compute EM/F1/containment, max over gold + aliases."""
    candidates = [gold] + (aliases or [])
    best_em, best_f1, best_cont = 0.0, 0.0, 0.0
    for c in candidates:
        if not c:
            continue
        best_em = max(best_em, compute_em(pred, c))
        best_f1 = max(best_f1, compute_f1(pred, c))
        best_cont = max(best_cont, compute_containment(pred, c))
    return {"em": best_em, "f1": best_f1, "containment": best_cont}


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
    """Normalize answer from both SPARQL and text routes."""
    if not raw:
        return ""
    answer = extract_final_answer(raw)
    for prefix in ["The answer is ", "Answer: ", "Based on the text, "]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    return answer.strip().rstrip(".")


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------
_stats = {"done": 0, "total": 0, "em_sum": 0.0, "f1_sum": 0.0, "start": 0.0}


def _signal_handler(sig, frame):
    d = _stats["done"]
    em = _stats["em_sum"] / d * 100 if d else 0
    f1 = _stats["f1_sum"] / d * 100 if d else 0
    elapsed = time.time() - _stats["start"] if _stats["start"] else 0
    log(f"\n--- Interrupted after {d}/{_stats['total']} questions ---")
    log(f"Running EM={em:.1f}%, F1={f1:.1f}%, Elapsed={elapsed / 60:.1f}min")
    log("Results saved. Resume by running the same command.")
    sys.exit(0)


def main():
    force_ipv4()
    parser = argparse.ArgumentParser(description="Hybrid QA cross-benchmark evaluation")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--provider", default="groq", choices=["groq", "github", "ollama"])
    parser.add_argument(
        "--benchmark", required=True, choices=["hotpotqa", "musique"],
        help="Benchmark to evaluate on",
    )
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between questions")
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.join(
            PROJECT_ROOT, "data", "cross_benchmark", args.benchmark,
            f"hybrid_{args.benchmark}_1000q.jsonl",
        )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Open log file next to JSONL output
    global _log_file
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    _log_file, log_path = _open_log(args.benchmark, args.output)

    log(f"=== Hybrid Cross-Benchmark: {args.benchmark} ===")
    log(f"Date:     {time.strftime('%Y-%m-%d')}")
    log(f"LLM:      {args.model} ({args.provider})")
    log(f"Output:   {args.output}")
    log(f"Log:      {log_path}")
    log("=" * 50)

    # Load questions and corpus
    questions = load_benchmark_questions(args.benchmark)
    benchmark_corpus = load_benchmark_corpus(args.benchmark)

    # Resume
    done_ids = load_completed_ids(args.output)
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    if done_ids:
        log(f"RESUME: {len(done_ids)} done, {len(remaining)} remaining")
    if args.limit > 0:
        remaining = remaining[:args.limit]
    if not remaining:
        log("All questions already processed.")
        return

    log(f"Processing: {len(remaining)} questions")

    # Create shared clients
    log("Creating shared clients...")
    try:
        llm_client = create_llm_client(args.provider, api_key=args.api_key)
    except ValueError as e:
        log(f"Error: {e}")
        sys.exit(1)
    linker = DBpediaEntityLinker(
        min_confidence=0.5, llm_client=llm_client, llm_model=args.model,
        wiki_resolver=WikipediaResolver(),
    )
    kg_client = DBpediaSPARQLClient()
    schema_disc = SchemaDiscovery()

    _stats["total"] = len(remaining)
    _stats["start"] = time.time()
    em_sum, f1_sum = 0.0, 0.0
    by_method = defaultdict(lambda: {"count": 0, "em": 0.0, "f1": 0.0})
    kg_linked_count = 0  # Track KG hit rate

    _HEDGE_RE = re.compile(
        r"(?:not sure|unclear|cannot determine|could not find|no information)", re.I,
    )

    log(f"\n--- Starting {len(remaining)} questions (delay={args.delay}s) ---\n")

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        question = q["question"]
        gold = q["gold_answer"]

        t0 = time.time()
        record = {
            "question_id": qid,
            "question_type": q["question_type"],
            "question": question,
            "gold_answer": gold,
            "gold_aliases": q["gold_aliases"],
            "benchmark": args.benchmark,
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
            if result.sub_question_log:
                record["sub_question_log"] = result.sub_question_log
            record["has_hedging"] = bool(_HEDGE_RE.search(record["raw_answer"]))
            record["error"] = ""

            # Track KG hit rate: check if any entities linked to DBpedia
            debug_info = getattr(result, "debug_info", {}) or {}
            entities_linked = debug_info.get("entities_linked", [])
            n_linked = len(entities_linked) if isinstance(entities_linked, list) else 0
            record["dbpedia_linked"] = n_linked > 0
            record["entities_linked_count"] = n_linked
            if n_linked > 0:
                kg_linked_count += 1

        except _QuestionTimeout:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "timeout"
            record["route_source"] = ""
            record["sparql_fallback"] = False
            record["sparql_query"] = ""
            record["has_hedging"] = False
            record["dbpedia_linked"] = False
            record["entities_linked_count"] = 0
            record["error"] = f"Timeout after {QUESTION_TIMEOUT}s"
        except Exception:
            record["raw_answer"] = ""
            record["predicted_answer"] = ""
            record["method"] = "error"
            record["route_source"] = ""
            record["sparql_fallback"] = False
            record["sparql_query"] = ""
            record["has_hedging"] = False
            record["dbpedia_linked"] = False
            record["entities_linked_count"] = 0
            record["error"] = traceback.format_exc()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        # Compute metrics (alias-aware)
        pred = record["predicted_answer"]
        raw = record.get("raw_answer", "")
        metrics = compute_best_metrics(pred, gold, q["gold_aliases"]) if pred else {
            "em": 0.0, "f1": 0.0, "containment": 0.0,
        }
        record["em"] = metrics["em"]
        record["f1"] = metrics["f1"]
        record["containment"] = (
            compute_best_metrics(raw, gold, q["gold_aliases"])["containment"]
            if raw else 0.0
        )
        record["latency_seconds"] = round(time.time() - t0, 2)

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        em_sum += record["em"]
        f1_sum += record["f1"]
        done = i + 1
        _stats["done"] = done
        _stats["em_sum"] = em_sum
        _stats["f1_sum"] = f1_sum

        method = record["method"]
        by_method[method]["count"] += 1
        by_method[method]["em"] += record["em"]
        by_method[method]["f1"] += record["f1"]

        em_tag = "EM=1" if record["em"] > 0 else f"F1={record['f1']:.2f}"
        log(f"[{done}/{len(remaining)}] {question[:60]}...")
        log(f"  -> gold='{gold}' pred='{pred[:60]}' [{em_tag}] "
            f"method={record['method']} {record['latency_seconds']}s "
            f"{'KG' if record['dbpedia_linked'] else 'noKG'}")

        if done % 25 == 0:
            elapsed = time.time() - _stats["start"]
            eta = (elapsed / done) * (len(remaining) - done) if done > 0 else 0
            log(f"  --- Running: EM={em_sum / done * 100:.1f}% "
                f"F1={f1_sum / done * 100:.1f}% "
                f"KG={kg_linked_count}/{done} ({kg_linked_count / done * 100:.0f}%) "
                f"| Elapsed={elapsed / 60:.1f}min ETA={eta / 60:.1f}min ---")

        if i < len(remaining) - 1:
            time.sleep(args.delay)

    # Final summary
    total = len(remaining)
    elapsed = time.time() - _stats["start"]
    log(f"\n{'=' * 60}")
    log(f"COMPLETE: {total} questions in {elapsed / 60:.1f}min")
    if total > 0:
        log(f"  EM:  {em_sum / total * 100:.1f}%")
        log(f"  F1:  {f1_sum / total * 100:.1f}%")
        log(f"  KG augmentation rate: {kg_linked_count}/{total} "
            f"({kg_linked_count / total * 100:.1f}%)")
    log("\nBy method:")
    for method, stats in sorted(by_method.items()):
        c = stats["count"]
        me = stats["em"] / c * 100 if c else 0
        mf = stats["f1"] / c * 100 if c else 0
        log(f"  {method}: {c} questions, EM={me:.1f}%, F1={mf:.1f}%")
    log(f"\nResults: {args.output}")
    log(f"Log:     {log_path}")

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
