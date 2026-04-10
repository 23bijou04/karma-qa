#!/usr/bin/env python3
"""100-question entity linking evaluation.

Evaluates the 4-phase EL cascade (Spotlight -> URI probe -> bif:contains -> Wikipedia)
against hop-verified gold URIs on a stratified 100-question sample.

Usage:
    python scripts/eval_entity_linking_100q.py --limit 3       # dry run
    python scripts/eval_entity_linking_100q.py                 # full 100q
    python scripts/eval_entity_linking_100q.py --report-only   # regenerate report
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from urllib.parse import unquote

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_parent = os.path.dirname(PROJECT_ROOT)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "eval_1000q", "hop_verification_results.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "eval_1000q", "el_eval_100q_results.jsonl")
REPORT_DIR = os.path.join(
    PROJECT_ROOT, "plans", "260325-2155-100q-entity-linking-eval", "reports",
)


# ---------------------------------------------------------------------------
# Phase 1: Question Selection
# ---------------------------------------------------------------------------

def _is_easy(q: dict) -> bool:
    """All hops resolved, no short/ambiguous subject names."""
    for hop in q["hops"]:
        if not hop["entity_resolved"]:
            return False
        name = hop["subject"]
        # Single-word name shorter than 12 chars = ambiguity risk
        if len(name) < 12 and len(name.split()) == 1:
            return False
        if "(" in name:  # parenthetical disambiguation = harder
            return False
    return True


def _is_hard(q: dict) -> bool:
    """Any hop with unresolved entity or parenthetical disambiguation."""
    # Only strong difficulty signals — short names go to remainder bucket
    for hop in q["hops"]:
        if not hop["entity_resolved"]:
            return True
        if "(" in hop["subject"]:
            return True
    return False


def select_100q(results: list[dict], seed: int = 42) -> list[dict]:
    """Select 100 questions with type-proportional stratification."""
    rng = random.Random(seed)
    # Proportional to 1000q type distribution
    targets = {"compositional": 41, "bridge_comparison": 24, "comparison": 24, "inference": 11}

    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r)

    selected = []
    for qtype, target in targets.items():
        pool = by_type.get(qtype, [])
        easy = [q for q in pool if _is_easy(q)]
        hard = [q for q in pool if _is_hard(q)]
        remainder = [q for q in pool if not _is_easy(q) and not _is_hard(q)]
        for bucket in (easy, hard, remainder):
            rng.shuffle(bucket)

        # ~1/3 per bucket, redistribute on shortfall
        n_easy = min(len(easy), target // 3)
        n_hard = min(len(hard), target // 3)
        n_rem = target - n_easy - n_hard
        if n_rem > len(remainder):
            extra = n_rem - len(remainder)
            n_rem = len(remainder)
            if len(easy) - n_easy >= len(hard) - n_hard:
                n_easy = min(len(easy), n_easy + extra)
            else:
                n_hard = min(len(hard), n_hard + extra)
            still = target - n_easy - n_hard - n_rem
            if still > 0:
                n_easy = min(len(easy), n_easy + still)
                still = target - n_easy - n_hard - n_rem
            if still > 0:
                n_hard = min(len(hard), n_hard + still)

        actual = min(len(easy), n_easy) + min(len(hard), n_hard) + min(len(remainder), n_rem)
        if actual < target:
            print(f"  WARNING: {qtype} has only {actual}/{target} questions available")

        for q in easy[:n_easy]:
            selected.append({**q, "_bucket": "easy"})
        for q in hard[:n_hard]:
            selected.append({**q, "_bucket": "hard"})
        for q in remainder[:n_rem]:
            selected.append({**q, "_bucket": "remainder"})

    # Ensure >=5 questions with unresolved hops
    has_unresolved = [s for s in selected if not s["all_entities_resolved"]]
    if len(has_unresolved) < 5:
        sel_ids = {s["question_id"] for s in selected}
        unr_pool = [r for r in results if not r["all_entities_resolved"] and r["question_id"] not in sel_ids]
        rng.shuffle(unr_pool)
        for ur in unr_pool[: 5 - len(has_unresolved)]:
            for i, s in enumerate(selected):
                if s["question_type"] == ur["question_type"] and s["_bucket"] == "remainder":
                    selected[i] = {**ur, "_bucket": "hard"}
                    break

    # Build output structure
    output = []
    for s in selected:
        golds = []
        has_unres = False
        for hop in s["hops"]:
            if not hop["entity_resolved"]:
                has_unres = True
            golds.append({"name": hop["subject"], "uri": hop["entity_uri"],
                          "resolved": hop["entity_resolved"], "hop_index": hop["hop_index"]})
        output.append({"question_id": s["question_id"], "question": s["question"],
                        "question_type": s["question_type"], "bucket": s["_bucket"],
                        "gold_entities": golds, "num_hops": s["num_hops"],
                        "has_unresolved": has_unres})
    return output


# ---------------------------------------------------------------------------
# Phase 2: Entity Linking Runner
# ---------------------------------------------------------------------------

def norm_uri(uri: str) -> str:
    """Normalize DBpedia URI for comparison."""
    return unquote(uri).lower().replace(" ", "_").rstrip("/")


def _subject_in_question(subject: str, question: str) -> bool:
    """Check if gold hop subject appears in question text."""
    # Normalize hyphens to spaces for robust matching (e.g. "Polish-Russian")
    s = subject.lower().strip().replace("-", " ")
    q = question.lower().replace("-", " ")
    if s in q:
        return True
    # Word-level recall >= 0.5 (handles slight variations)
    s_words = set(s.split())
    return len(s_words & set(q.split())) / len(s_words) >= 0.5 if s_words else False


_redirect_cache: dict[tuple, bool] = {}


def _check_redirect(uri_a: str, uri_b: str, sparql) -> bool:
    """Check if uri_a redirects to uri_b (or vice versa), with cache."""
    key = (uri_a, uri_b)
    if key in _redirect_cache:
        return _redirect_cache[key]
    result = False
    for src, tgt in [(uri_a, uri_b), (uri_b, uri_a)]:
        try:
            info = sparql.validate_entity_uri(src)
            if info.get("redirect_target") and norm_uri(info["redirect_target"]) == norm_uri(tgt):
                result = True
                break
        except Exception:
            pass
    _redirect_cache[key] = result
    return result


def match_entities(linked, gold_entities, question, sparql) -> list[dict]:
    """Match linked entities against gold hop entities. Returns match records."""
    matches = []
    for ge in gold_entities:
        evaluable = _subject_in_question(ge["name"], question)
        best_type, best_uri, best_source = "miss", None, None

        if evaluable and ge["uri"]:
            gold_norm = norm_uri(ge["uri"])
            # 1. Exact URI match
            for le in linked:
                if norm_uri(le.uri) == gold_norm:
                    best_type, best_uri, best_source = "exact", le.uri, le.source
                    break
            # 2. Redirect match
            if best_type == "miss":
                for le in linked:
                    if _check_redirect(le.uri, ge["uri"], sparql):
                        best_type, best_uri, best_source = "redirect", le.uri, le.source
                        break
            # 3. Label match (exact or Jaccard >= 0.5)
            if best_type == "miss":
                gn = ge["name"].lower().strip()
                for le in linked:
                    ll = le.label.lower().strip()
                    if ll == gn:
                        best_type, best_uri, best_source = "label_match", le.uri, le.source
                        break
                    lw, gw = set(ll.split()), set(gn.split())
                    if lw and gw and len(lw & gw) / len(lw | gw) >= 0.5:
                        best_type, best_uri, best_source = "label_match", le.uri, le.source
                        break

        matches.append({"gold_name": ge["name"], "gold_uri": ge["uri"],
                         "linked_uri": best_uri, "linked_source": best_source,
                         "match_type": best_type, "evaluable": evaluable,
                         "gold_resolved": ge["resolved"]})
    return matches


def _load_completed(path: str) -> set[str]:
    """Load question IDs already in JSONL output (for resume)."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def run_eval(selected: list[dict], output_file: str, limit: int | None = None):
    """Run 4-phase entity linking cascade on selected questions."""
    from dbpedia_qa.dbpedia.client import DBpediaSPARQLClient  # noqa: E402
    from dbpedia_qa.dbpedia.entity_linker import DBpediaEntityLinker  # noqa: E402
    from dbpedia_qa.dbpedia.wiki_resolver import WikipediaResolver  # noqa: E402

    sparql = DBpediaSPARQLClient()
    linker = DBpediaEntityLinker(
        sparql_client=sparql, wiki_resolver=WikipediaResolver(), llm_client=None,
    )

    done_ids = _load_completed(output_file)
    remaining = [q for q in selected if q["question_id"] not in done_ids]
    if limit:
        remaining = remaining[:limit]

    total = len(remaining)
    print(f"\n{'=' * 60}")
    print(f"EL Eval: {total} questions to process (already done: {len(done_ids)})")
    print(f"{'=' * 60}\n")

    for i, q in enumerate(remaining, 1):
        t0 = time.perf_counter()
        try:
            linked = linker.link_entities(q["question"], use_cache=False, question=q["question"])
            matches = match_entities(linked, q["gold_entities"], q["question"], sparql)
            ev = [m for m in matches if m["evaluable"]]
            tp = sum(1 for m in ev if m["match_type"] != "miss")
            recall = tp / len(ev) if ev else 0.0
            matched_uris = {m["linked_uri"] for m in matches if m["linked_uri"]}
            extra = sum(1 for le in linked if le.uri not in matched_uris)
            elapsed = (time.perf_counter() - t0) * 1000

            record = {"question_id": q["question_id"], "question": q["question"],
                       "question_type": q["question_type"], "bucket": q["bucket"],
                       "gold_entities": q["gold_entities"],
                       "linked_entities": [{"uri": e.uri, "label": e.label, "source": e.source} for e in linked],
                       "matches": matches, "recall": recall,
                       "extra_linked": extra, "elapsed_ms": elapsed, "error": None}
            status = f"recall={recall:.0%}"
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            record = {"question_id": q["question_id"], "question": q["question"],
                       "error": str(e), "elapsed_ms": elapsed}
            status = f"ERROR: {e}"

        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{i}/{total}] {q['question_id'][:8]}... {status} ({elapsed:.0f}ms)")

    print(f"\nDone. Results: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="100q Entity Linking Evaluation")
    parser.add_argument("--gold", default=GOLD_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--limit", type=int, help="Limit questions (dry run)")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report only")
    parser.add_argument("--select-only", action="store_true", help="Show selection only")
    args = parser.parse_args()

    with open(args.gold) as f:
        gold = json.load(f)
    selected = select_100q(gold["results"])

    tc = Counter(q["question_type"] for q in selected)
    bc = Counter(q["bucket"] for q in selected)
    unr = sum(1 for q in selected if q["has_unresolved"])
    print(f"Selected {len(selected)}q | Types: {dict(tc)} | Buckets: {dict(bc)} | Unresolved: {unr}")

    if args.select_only:
        for q in selected:
            print(f"  {q['question_id'][:8]} [{q['question_type']:20s}] [{q['bucket']:10s}] {q['question'][:60]}")
        return

    if not args.report_only:
        run_eval(selected, args.output, limit=args.limit)

    # Phase 3: Metrics & report
    if not os.path.exists(args.output):
        print("No results file. Run eval first.")
        return

    # Import report module (same directory)
    from el_eval_report import compute_metrics, generate_report, load_results, print_summary
    results_data = load_results(args.output)
    if not results_data:
        print("No valid results found.")
        return

    metrics = compute_metrics(results_data)
    print_summary(metrics)
    report_path = os.path.join(args.report_dir, "el-eval-100q-report.md")
    generate_report(metrics, report_path)


if __name__ == "__main__":
    main()
