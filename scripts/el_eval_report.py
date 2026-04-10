"""Metrics computation and report generation for EL evaluation.

Computes micro-averaged recall, per-type/bucket breakdowns, phase attribution,
error taxonomy, and generates a markdown report with Wilson confidence intervals.

Used by eval_entity_linking_100q.py (Phase 3).
"""

import json
import math
import os
import time
from collections import Counter


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    d = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / d
    margin = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / d
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_results(path: str) -> list[dict]:
    """Load JSONL results, skip error-only records."""
    results = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error") and "matches" not in rec:
                continue
            results.append(rec)
    return results


def _classify_miss(match: dict, linked_entities: list[dict]) -> str:
    """Classify a miss into error taxonomy."""
    gold_name = match["gold_name"].lower().strip()

    if not match.get("gold_resolved", True):
        return "unresolvable"

    # Check if any linked entity has significant name overlap
    gold_words = set(gold_name.split())
    for le in linked_entities:
        ll = le["label"].lower().strip()
        lw = set(ll.split())
        # Jaccard >= 0.5 or substring containment = wrong disambiguation
        if lw and gold_words:
            if len(lw & gold_words) / len(lw | gold_words) >= 0.5:
                return "wrong_disambiguation"
        if gold_name in ll or ll in gold_name:
            return "wrong_disambiguation"

    # Partial name: any significant gold word appears in a linked label
    for le in linked_entities:
        ll = le["label"].lower().strip()
        if any(w in ll for w in gold_words if len(w) > 3):
            return "partial_name"

    return "missed_mention"


def _recall_stats(matches: list[dict]) -> dict:
    """Compute recall stats from a list of evaluable match records."""
    tp = sum(1 for m in matches if m["match_type"] != "miss")
    fn = sum(1 for m in matches if m["match_type"] == "miss")
    n = tp + fn
    r = tp / n if n else 0.0
    return {"recall": r, "tp": tp, "fn": fn, "n": n, "ci": wilson_ci(tp, n)}


def compute_metrics(results: list[dict]) -> dict:
    """Compute all metrics from JSONL results."""
    all_ev = [m for r in results for m in r.get("matches", []) if m["evaluable"]]

    # Overall recall
    tp = sum(1 for m in all_ev if m["match_type"] != "miss")
    fn = sum(1 for m in all_ev if m["match_type"] == "miss")
    n_ev = tp + fn

    # Per-type and per-bucket breakdowns
    by_type: dict[str, list] = {}
    by_bucket: dict[str, list] = {}
    for r in results:
        for m in r.get("matches", []):
            if m["evaluable"]:
                by_type.setdefault(r["question_type"], []).append(m)
                by_bucket.setdefault(r.get("bucket", "unknown"), []).append(m)

    per_type = {k: _recall_stats(v) for k, v in by_type.items()}
    per_bucket = {k: _recall_stats(v) for k, v in by_bucket.items()}

    # Phase attribution (which cascade phase produced each TP)
    phase_counts = Counter(m["linked_source"] for m in all_ev if m["match_type"] != "miss")
    match_types = Counter(m["match_type"] for m in all_ev)

    # Error classification for misses
    error_counts: Counter = Counter()
    error_examples: dict[str, list] = {}
    for r in results:
        for m in r.get("matches", []):
            if m["evaluable"] and m["match_type"] == "miss":
                cat = _classify_miss(m, r.get("linked_entities", []))
                error_counts[cat] += 1
                if len(error_examples.get(cat, [])) < 3:
                    error_examples.setdefault(cat, []).append(
                        {"question": r["question"][:80], "gold_name": m["gold_name"],
                         "gold_uri": m["gold_uri"]})

    # Bonus: entities our linker resolved that gold missed
    bonus_count = 0
    bonus_examples = []
    for r in results:
        for m in r.get("matches", []):
            if not m.get("gold_resolved", True) and m["match_type"] != "miss":
                bonus_count += 1
                if len(bonus_examples) < 5:
                    bonus_examples.append({"question": r["question"][:80],
                                            "gold_name": m["gold_name"],
                                            "linked_uri": m["linked_uri"]})

    # Disambiguation disputes (redirect/label_match = different URI than gold)
    disputes = []
    for r in results:
        for m in r.get("matches", []):
            if m["evaluable"] and m["match_type"] in ("redirect", "label_match"):
                disputes.append({"question": r["question"][:80], "gold_name": m["gold_name"],
                                  "gold_uri": m["gold_uri"], "linked_uri": m["linked_uri"],
                                  "match_type": m["match_type"]})

    # Timing stats
    times = sorted(r["elapsed_ms"] for r in results if "elapsed_ms" in r)
    extra_total = sum(r.get("extra_linked", 0) for r in results)

    return {
        "n_questions": len(results), "n_evaluable": n_ev,
        "tp": tp, "fn": fn,
        "recall": tp / n_ev if n_ev else 0.0, "recall_ci": wilson_ci(tp, n_ev),
        "per_type": per_type, "per_bucket": per_bucket,
        "phase_counts": dict(phase_counts), "match_types": dict(match_types),
        "error_counts": dict(error_counts), "error_examples": error_examples,
        "bonus_count": bonus_count, "bonus_examples": bonus_examples,
        "disputes": disputes[:10], "extra_total": extra_total,
        "timing": {
            "total_ms": sum(times),
            "avg_ms": sum(times) / len(times) if times else 0,
            "p50_ms": times[len(times) // 2] if times else 0,
            "p95_ms": times[int(len(times) * 0.95)] if times else 0,
        },
    }


def print_summary(metrics: dict):
    """Print concise summary to console."""
    m = metrics
    lo, hi = m["recall_ci"]
    print(f"\n{'=' * 60}")
    print(f"EL EVAL SUMMARY ({m['n_questions']}q)")
    print(f"{'=' * 60}")
    print(f"Recall: {m['recall']:.1%} [{lo:.1%}, {hi:.1%}] (N={m['n_evaluable']})")
    print(f"  TP={m['tp']}, FN={m['fn']}, Extra linked={m['extra_total']}")
    print("\nPhase attribution:")
    for phase, cnt in sorted(m["phase_counts"].items(), key=lambda x: -x[1]):
        pct = cnt / m["tp"] * 100 if m["tp"] else 0
        print(f"  {phase:20s}: {cnt:3d} ({pct:.1f}%)")
    print("\nPer type:")
    for qt in ["compositional", "bridge_comparison", "comparison", "inference"]:
        if qt in m["per_type"]:
            d = m["per_type"][qt]
            print(f"  {qt:20s}: {d['recall']:.1%} (N={d['n']})")
    print(f"\nErrors ({m['fn']} misses):")
    for cat, cnt in sorted(m["error_counts"].items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {cnt}")


def generate_report(metrics: dict, output_path: str):
    """Generate markdown report from computed metrics."""
    m = metrics
    L = []  # report lines

    L.append("# Entity Linking Evaluation Report (100q)\n")
    L.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    L.append(f"**Sample**: {m['n_questions']} questions (seed=42), type-proportional stratified")
    L.append("**Linker config**: 4-phase cascade, no LLM reranker")
    L.append(f"**Runtime**: {m['timing']['total_ms']/1000:.0f}s "
             f"({m['timing']['total_ms']/60000:.1f}min)\n")

    # Overall
    lo, hi = m["recall_ci"]
    L.append("## Overall Metrics\n")
    L.append("| Metric | Value | 95% CI |")
    L.append("|--------|-------|--------|")
    L.append(f"| Recall | {m['recall']:.1%} | [{lo:.1%}, {hi:.1%}] |\n")
    L.append(f"- Evaluable gold entities: {m['n_evaluable']}")
    L.append(f"- TP: {m['tp']}, FN: {m['fn']}")
    L.append(f"- Extra linked (not in gold): {m['extra_total']}\n")

    # Match type breakdown
    L.append("### Match Type Breakdown\n")
    L.append("| Type | Count | % |")
    L.append("|------|-------|---|")
    for mt in ["exact", "redirect", "label_match", "miss"]:
        cnt = m["match_types"].get(mt, 0)
        pct = cnt / m["n_evaluable"] * 100 if m["n_evaluable"] else 0
        L.append(f"| {mt} | {cnt} | {pct:.1f}% |")
    L.append("")

    # Per-type
    L.append("## Per-Type Breakdown\n")
    L.append("| Type | N | Recall | 95% CI | TP | FN |")
    L.append("|------|---|--------|--------|----|----|")
    for qt in ["compositional", "bridge_comparison", "comparison", "inference"]:
        if qt in m["per_type"]:
            d = m["per_type"][qt]
            lo, hi = d["ci"]
            L.append(f"| {qt} | {d['n']} | {d['recall']:.1%} | [{lo:.1%}, {hi:.1%}] | {d['tp']} | {d['fn']} |")
    L.append("")

    # Per-bucket
    L.append("## Per-Bucket Breakdown\n")
    L.append("| Bucket | N | Recall | 95% CI |")
    L.append("|--------|---|--------|--------|")
    for bk in ["easy", "hard", "remainder"]:
        if bk in m["per_bucket"]:
            d = m["per_bucket"][bk]
            lo, hi = d["ci"]
            L.append(f"| {bk} | {d['n']} | {d['recall']:.1%} | [{lo:.1%}, {hi:.1%}] |")
    L.append("")

    # Phase attribution
    L.append("## Per-Phase Cascade Contribution\n")
    L.append("| Phase | TPs | % of all TPs |")
    L.append("|-------|-----|--------------|")
    for ph, cnt in sorted(m["phase_counts"].items(), key=lambda x: -x[1]):
        pct = cnt / m["tp"] * 100 if m["tp"] else 0
        L.append(f"| {ph} | {cnt} | {pct:.1f}% |")
    L.append("")

    # Error analysis
    L.append(f"## Error Analysis ({m['fn']} misses)\n")
    L.append("| Category | Count | % of FN |")
    L.append("|----------|-------|---------|")
    for cat in ["missed_mention", "wrong_disambiguation", "partial_name", "unresolvable"]:
        cnt = m["error_counts"].get(cat, 0)
        if cnt > 0:
            pct = cnt / m["fn"] * 100 if m["fn"] else 0
            L.append(f"| {cat} | {cnt} | {pct:.1f}% |")
    L.append("")
    for cat, examples in m["error_examples"].items():
        L.append(f"### {cat} examples\n")
        for ex in examples:
            L.append(f"- **{ex['gold_name']}** (`{ex['gold_uri']}`)")
            L.append(f"  Q: _{ex['question']}_")
        L.append("")

    # Bonus: Linker vs Gold
    L.append("## Bonus: Linker vs Gold\n")
    L.append(f"Entities our linker resolved that gold missed: **{m['bonus_count']}**\n")
    for ex in m["bonus_examples"]:
        L.append(f"- **{ex['gold_name']}** -> `{ex['linked_uri']}`")
        L.append(f"  Q: _{ex['question']}_")
    L.append("")

    # Disambiguation disputes
    if m["disputes"]:
        L.append("## Disambiguation Disputes\n")
        L.append("Cases where linker found different URI than gold (scored as TP):\n")
        for d in m["disputes"]:
            L.append(f"- **{d['gold_name']}** ({d['match_type']})")
            L.append(f"  Gold: `{d['gold_uri']}` | Linked: `{d['linked_uri']}`")
            L.append(f"  Q: _{d['question']}_")
        L.append("")

    # Timing
    t = m["timing"]
    L.append("## Timing\n")
    L.append(f"- Total: {t['total_ms']/1000:.0f}s | Avg: {t['avg_ms']:.0f}ms/q")
    L.append(f"- P50: {t['p50_ms']:.0f}ms | P95: {t['p95_ms']:.0f}ms\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nReport: {output_path}")
