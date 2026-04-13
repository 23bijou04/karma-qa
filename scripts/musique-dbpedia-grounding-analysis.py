#!/usr/bin/env python3
"""Phase 2: MuSiQue DBpedia grounding — 3-tier funnel (entity, connectivity, value).
Usage: python scripts/musique-dbpedia-grounding-analysis.py [--resume] [--limit N]
Output: data/cross_benchmark/musique/ablation/dbpedia_grounding_results.json
"""
import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

_base = Path(__file__).resolve().parent
_spec_mu = importlib.util.spec_from_file_location("mu", _base / "musique-analysis-utils.py")
_mu = importlib.util.module_from_spec(_spec_mu)
_spec_mu.loader.exec_module(_mu)
_spec_sp = importlib.util.spec_from_file_location("sp", _base / "musique-grounding-sparql-helpers.py")
_sp = importlib.util.module_from_spec(_spec_sp)
_spec_sp.loader.exec_module(_sp)

load_musique_gt, normalize_answer, ABLATION_DIR = _mu.load_musique_gt, _mu.normalize_answer, _mu.ABLATION_DIR
resolve_title_to_uri = _sp.resolve_title_to_uri
check_connection, retrieve_values = _sp.check_connection, _sp.retrieve_values
answer_grounded = _sp.answer_grounded
uri_cache = _sp.uri_cache

OUTPUT_PATH = ABLATION_DIR / "dbpedia_grounding_results.json"
CHECKPOINT_PATH = ABLATION_DIR / "grounding_checkpoint.json"


def extract_hop_titles(question: dict) -> list[tuple[int, str, str]]:
    """Return [(hop_idx, title, hop_answer)] for supporting paragraphs."""
    decomp = question["question_decomposition"]
    paras = question["paragraphs"]
    hops = []
    for i, hop in enumerate(decomp):
        pidx = hop.get("paragraph_support_idx")
        if pidx is not None and pidx < len(paras):
            hops.append((i, paras[pidx]["title"], hop["answer"]))
    return hops


def verify_question(question: dict) -> dict:
    """Run 3-tier funnel for one MuSiQue question."""
    qid = str(question["id"])
    # Infer question type from ID prefix (GT has no question_type field)
    qtype = qid.split("__")[0] if "__" in qid else "unknown"
    hops = extract_hop_titles(question)

    # Tier 1: resolve each supporting title
    hop_uris = []
    for i, title, ans in hops:
        uri = resolve_title_to_uri(title)
        hop_uris.append({"hop": i, "title": title, "uri": uri, "answer": ans})
    all_resolved = all(h["uri"] is not None for h in hop_uris)

    # Tier 2: check connectivity between consecutive hop entities
    connections = []
    all_connected = True
    for j in range(len(hop_uris) - 1):
        u1 = hop_uris[j]["uri"]
        u2 = hop_uris[j + 1]["uri"]
        if u1 and u2:
            connected, conn_type = check_connection(u1, u2)
            connections.append({
                "from_hop": j, "to_hop": j + 1,
                "connected": connected, "type": conn_type,
            })
            if not connected:
                all_connected = False
        else:
            connections.append({
                "from_hop": j, "to_hop": j + 1,
                "connected": False, "type": "uri_missing",
            })
            all_connected = False

    # Tier 3: per-hop answer grounding
    hop_grounded = []
    for h in hop_uris:
        if h["uri"]:
            values = retrieve_values(h["uri"])
            grounded, matched = answer_grounded(h["answer"], normalize_answer, values)
            hop_grounded.append({"hop": h["hop"], "grounded": grounded, "matched_value": matched})
        else:
            hop_grounded.append({"hop": h["hop"], "grounded": False, "matched_value": None})
    all_grounded = all(h["grounded"] for h in hop_grounded) if hop_grounded else False

    return {
        "question_id": qid,
        "question_type": qtype,
        "n_hops": len(hops),
        "tier1_all_resolved": all_resolved,
        "tier1_resolved_count": sum(1 for h in hop_uris if h["uri"]),
        "tier1_total": len(hop_uris),
        "tier2_all_connected": all_connected if len(hops) > 1 else True,
        "tier2_connections": connections,
        "tier3_all_grounded": all_grounded,
        "tier3_grounded_count": sum(1 for h in hop_grounded if h["grounded"]),
        "hop_details": hop_uris,
        "grounding_details": hop_grounded,
    }


def save_checkpoint(results: list[dict]):
    """Save checkpoint with processed results and URI cache."""
    data = {"results": results, "uri_cache": dict(uri_cache)}
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def load_checkpoint() -> tuple[list[dict], set[str]]:
    """Load checkpoint if exists. Returns (results, processed_ids)."""
    if not CHECKPOINT_PATH.exists():
        return [], set()
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("WARNING: corrupt checkpoint, starting fresh", file=__import__("sys").stderr)
        return [], set()
    uri_cache.update(data.get("uri_cache", {}))
    results = data.get("results", [])
    return results, {r["question_id"] for r in results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    args = parser.parse_args()

    gt = load_musique_gt()
    questions = sorted(gt.values(), key=lambda q: str(q["id"]))
    if args.limit > 0:
        questions = questions[:args.limit]

    results, processed = (load_checkpoint() if args.resume else ([], set()))
    if processed:
        print(f"Resumed: {len(processed)} already processed, {len(uri_cache)} cached URIs")

    remaining = [q for q in questions if str(q["id"]) not in processed]
    print(f"Processing {len(remaining)} questions ({len(questions)} total, {len(processed)} done)")
    t0 = time.time()

    for idx, q in enumerate(remaining, 1):
        result = verify_question(q)
        results.append(result)
        # Checkpoint every 50 questions
        if idx % 50 == 0:
            save_checkpoint(results)
            elapsed = time.time() - t0
            rate = idx / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{idx}/{len(remaining)}] {rate:.0f} q/min | "
                  f"T1={sum(1 for r in results if r['tier1_all_resolved'])}/{len(results)}")
        if idx % 200 == 0:
            print(f"  Checkpoint saved ({idx} processed)")

    # ── Aggregate ───────────────────────────────────────────────────────────
    t1 = sum(1 for r in results if r["tier1_all_resolved"])
    t2 = sum(1 for r in results if r["tier2_all_connected"])
    t3 = sum(1 for r in results if r["tier3_all_grounded"])
    n = len(results)

    def _agg(results_list, group_fn):
        buckets = defaultdict(lambda: {"total": 0, "t1": 0, "t2": 0, "t3": 0})
        for r in results_list:
            b = buckets[group_fn(r)]
            b["total"] += 1
            for tier, key in [("t1", "tier1_all_resolved"), ("t2", "tier2_all_connected"),
                              ("t3", "tier3_all_grounded")]:
                if r[key]:
                    b[tier] += 1
        return {k: dict(v) for k, v in sorted(buckets.items())}

    def pct(x):
        return round(100 * x / max(n, 1), 1)
    output = {
        "summary": {"total_questions": n,
                     "tier1_all_resolved": t1, "tier1_pct": pct(t1),
                     "tier2_all_connected": t2, "tier2_pct": pct(t2),
                     "tier3_all_grounded": t3, "tier3_pct": pct(t3)},
        "by_type": _agg(results, lambda r: r.get("question_type", "?")),
        "by_hop_count": _agg(results, lambda r: str(r["n_hops"])),
        "per_question": results,
    }
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"\nDBPEDIA GROUNDING (MuSiQue {n}q, {elapsed:.0f}s)\n{'='*50}")
    for tier, label in [("t1", "entity coverage"), ("t2", "connectivity"), ("t3", "answer grounding")]:
        v = {"t1": t1, "t2": t2, "t3": t3}[tier]
        print(f"  Tier {tier[1]} ({label}): {v}/{n} ({pct(v)}%)")
    print(f"Saved: {OUTPUT_PATH}")
    if len(results) >= len(questions) and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleaned up")


if __name__ == "__main__":
    main()
