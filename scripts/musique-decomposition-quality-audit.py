#!/usr/bin/env python3
"""Phase 5: Decomposition quality audit — 3-axis (depth, structure, semantic).
Output: data/cross_benchmark/musique/ablation/decomposition_quality.json
"""
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

_spec = importlib.util.spec_from_file_location(
    "mu", Path(__file__).resolve().parent / "musique-analysis-utils.py")
_mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mu)
load_hybrid, load_musique_gt = _mu.load_hybrid, _mu.load_musique_gt
normalize_answer, save_json = _mu.normalize_answer, _mu.save_json


def classify_depth(sys_hops: int, gt_hops: int) -> str:
    if sys_hops == 0:
        return "no_decomposition"
    if sys_hops == gt_hops:
        return "exact_match"
    return "over_decomposed" if sys_hops > gt_hops else "under_decomposed"


def build_dep_graph(questions: list[str], ref_pattern: str) -> dict[int, list[int]]:
    """Parse sub-Q references into dependency graph.

    ref_pattern: '#' for GT (1-indexed), '[Q' for system (0-indexed).
    Returns {hop_idx: [depends_on_idx, ...]}.
    """
    deps = {}
    for i, q in enumerate(questions):
        if ref_pattern == "#":
            refs = [int(m) - 1 for m in re.findall(r"#(\d+)", q)]
        else:
            refs = [int(m) for m in re.findall(r"\[Q(\d+)\]", q)]
        deps[i] = sorted(set(refs))
    return deps


def _edges(graph):
    """Convert dep graph {node: [parents]} to set of (parent, node) tuples."""
    return {(p, n) for n, ps in graph.items() for p in ps}


def compute_edge_match(sys_graph: dict, gt_graph: dict) -> dict:
    sys_e, gt_e = _edges(sys_graph), _edges(gt_graph)
    if not gt_e:
        return {"gt_edges": 0, "sys_edges": len(sys_e), "matched": 0, "match_rate": 1.0}
    matched = len(sys_e & gt_e)
    return {"gt_edges": len(gt_e), "sys_edges": len(sys_e), "matched": matched,
            "match_rate": round(matched / len(gt_e), 4)}


def resolve_refs(text: str, pattern: str) -> str:
    """Replace #N or [QN] with ENTITY_N for fair comparison."""
    if pattern == "#":
        return re.sub(r"#(\d+)", lambda m: f"ENTITY_{int(m.group(1))-1}", text)
    return re.sub(r"\[Q(\d+)\]", lambda m: f"ENTITY_{m.group(1)}", text)


def token_overlap_f1(text_a: str, text_b: str) -> float:
    """Set-based token F1 for question similarity (not multiset SQuAD F1)."""
    a_tokens = set(normalize_answer(text_a).split())
    b_tokens = set(normalize_answer(text_b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    common = a_tokens & b_tokens
    if not common:
        return 0.0
    precision = len(common) / len(a_tokens)
    recall = len(common) / len(b_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def main():
    hybrid = load_hybrid()
    gt = load_musique_gt()

    per_question = []
    depth_dist = defaultdict(int)
    depth_by_type = defaultdict(lambda: defaultdict(int))

    for qid in sorted(hybrid):
        h = hybrid[qid]
        g = gt[qid]
        sys_subqs = h.get("sub_question_log") or []
        gt_decomp = g["question_decomposition"]
        qtype = h.get("question_type", "unknown")
        judge_correct = h.get("judge_correct", False)

        sys_hops = len(sys_subqs)
        gt_hops = len(gt_decomp)
        depth_cat = classify_depth(sys_hops, gt_hops)
        delta = sys_hops - gt_hops

        depth_dist[f"delta_{delta}"] = depth_dist.get(f"delta_{delta}", 0) + 1
        depth_by_type[qtype][depth_cat] = depth_by_type[qtype].get(depth_cat, 0) + 1

        # Structural alignment (only if both have sub-Qs)
        edge_match = None
        if sys_hops > 0 and gt_hops > 0:
            sys_qs = [sq.get("sub_question", "") for sq in sys_subqs]
            gt_qs = [hop["question"] for hop in gt_decomp]
            sys_graph = build_dep_graph(sys_qs, "[Q")
            gt_graph = build_dep_graph(gt_qs, "#")
            edge_match = compute_edge_match(sys_graph, gt_graph)

        # Semantic similarity (positional alignment, min hops)
        sem_scores = []
        n_aligned = min(sys_hops, gt_hops)
        for i in range(n_aligned):
            sys_q = resolve_refs(sys_subqs[i].get("sub_question", ""), "[Q")
            gt_q = resolve_refs(gt_decomp[i]["question"], "#")
            sem_scores.append(token_overlap_f1(sys_q, gt_q))

        avg_sem = round(sum(sem_scores) / len(sem_scores), 4) if sem_scores else 0.0

        per_question.append({
            "question_id": qid,
            "question_type": qtype,
            "judge_correct": judge_correct,
            "sys_hops": sys_hops,
            "gt_hops": gt_hops,
            "depth_delta": delta,
            "depth_category": depth_cat,
            "edge_match": edge_match,
            "avg_semantic_sim": avg_sem,
            "per_hop_semantic_sim": sem_scores,
        })

    # ── Correlation: each axis vs accuracy ──────────────────────────────────
    def acc(subset):
        return round(100 * sum(1 for q in subset if q["judge_correct"]) / max(len(subset), 1), 1)

    has_subqs = [q for q in per_question if q["sys_hops"] > 0]
    has_edges = [q for q in has_subqs if q["edge_match"] is not None]
    depth_cats = defaultdict(int)
    for q in per_question:
        depth_cats[q["depth_category"]] += 1
    # Build axis buckets: (high_subset, low_subset) per axis
    axes = {
        "depth": ([q for q in has_subqs if q["depth_category"] == "exact_match"],
                  [q for q in has_subqs if q["depth_category"] != "exact_match"]),
        "edge": ([q for q in has_edges if q["edge_match"]["match_rate"] >= 0.7],
                 [q for q in has_edges if q["edge_match"]["match_rate"] < 0.7]),
        "sem": ([q for q in has_subqs if q["avg_semantic_sim"] >= 0.6],
                [q for q in has_subqs if q["avg_semantic_sim"] < 0.6]),
    }
    avg_sem = sum(q["avg_semantic_sim"] for q in has_subqs) / max(len(has_subqs), 1)
    avg_edge = (sum(q["edge_match"]["match_rate"] for q in has_edges)
                / max(len(has_edges), 1)) if has_edges else 0.0
    corr = {}
    for name, (hi, lo) in axes.items():
        corr[f"{name}_hi_acc"] = acc(hi)
        corr[f"{name}_hi_n"] = len(hi)
        corr[f"{name}_lo_acc"] = acc(lo)
        corr[f"{name}_lo_n"] = len(lo)
        # Fisher's exact test: 2x2 contingency table
        # [[hi_correct, hi_wrong], [lo_correct, lo_wrong]]
        a = sum(1 for q in hi if q["judge_correct"])
        b = len(hi) - a
        c = sum(1 for q in lo if q["judge_correct"])
        d = len(lo) - c
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
        corr[f"{name}_fisher_p"] = round(p_value, 6)
        corr[f"{name}_odds_ratio"] = round(odds_ratio, 4)
    output = {
        "summary": {
            "total_questions": len(per_question), "questions_with_subqs": len(has_subqs),
            "depth_exact_match": len(axes["depth"][0]),
            "depth_exact_match_pct": round(100 * len(axes["depth"][0]) / max(len(has_subqs), 1), 1),
            "over_decomposed": depth_cats.get("over_decomposed", 0),
            "under_decomposed": depth_cats.get("under_decomposed", 0),
            "no_decomposition": depth_cats.get("no_decomposition", 0),
            "avg_semantic_similarity": round(avg_sem, 4),
            "avg_edge_match_rate": round(avg_edge, 4),
        },
        "depth_distribution": dict(sorted(depth_dist.items())),
        "correlation_by_axis": corr,
        "per_type": {t: dict(depth_by_type[t]) for t in sorted(depth_by_type)},
        "per_question": per_question,
    }

    out = save_json(output, "decomposition_quality.json")

    # ── Console summary ─────────────────────────────────────────────────────
    s, c = output["summary"], output["correlation_by_axis"]
    print(f"DECOMPOSITION QUALITY AUDIT (MuSiQue 1000q)\n{'='*55}")
    print(f"With sub-Qs: {s['questions_with_subqs']}/1000 | Depth match: {s['depth_exact_match']} "
          f"({s['depth_exact_match_pct']}%) | Over: {s['over_decomposed']} | Under: "
          f"{s['under_decomposed']} | No decomp: {s['no_decomposition']}")
    print(f"Avg sem sim: {s['avg_semantic_similarity']:.3f} | Avg edge match: "
          f"{s['avg_edge_match_rate']:.3f}")
    print(f"\nCORRELATION: Quality → Accuracy\n{'-'*45}")
    for name in ["depth", "edge", "sem"]:
        print(f"  {name}: {c[name+'_hi_acc']:>5.1f}% (n={c[name+'_hi_n']}) vs "
              f"{c[name+'_lo_acc']:>5.1f}% (n={c[name+'_lo_n']}) | "
              f"OR={c[name+'_odds_ratio']:.4f} p={c[name+'_fisher_p']:.6f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
