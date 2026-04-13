#!/usr/bin/env python3
"""Phase 1: Per-hop accuracy audit for MuSiQue Hybrid evaluation.

Aligns system sub-questions to GT decomposition hops (positional alignment),
computes per-hop EM/F1/containment, and detects error cascades.

Output: data/cross_benchmark/musique/ablation/per_hop_accuracy.json
"""
import importlib.util
from collections import defaultdict
from pathlib import Path

# ── Import shared utils (hyphenated filename needs importlib) ───────────────
_spec = importlib.util.spec_from_file_location(
    "mu", Path(__file__).resolve().parent / "musique-analysis-utils.py"
)
_mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mu)

load_hybrid = _mu.load_hybrid
load_musique_gt = _mu.load_musique_gt
compute_em = _mu.compute_em
compute_f1 = _mu.compute_f1
compute_containment = _mu.compute_containment
save_json = _mu.save_json


def align_and_compare(system_subqs: list, gt_decomp: list) -> list[dict]:
    """Positional alignment: system hop i <-> GT hop i."""
    results = []
    n = min(len(system_subqs), len(gt_decomp))
    for i in range(n):
        sys_ans = str(system_subqs[i].get("answer", "") or "")
        gt_ans = gt_decomp[i]["answer"]
        results.append({
            "hop": i,
            "status": "aligned",
            "sys_question": system_subqs[i].get("sub_question", ""),
            "gt_question": gt_decomp[i]["question"],
            "sys_answer": sys_ans,
            "gt_answer": gt_ans,
            "method": system_subqs[i].get("method", "unknown"),
            "em": compute_em(sys_ans, gt_ans),
            "f1": compute_f1(sys_ans, gt_ans),
            "containment": compute_containment(sys_ans, gt_ans),
        })
    for i in range(n, len(system_subqs)):
        results.append({"hop": i, "status": "unmatched_system",
                         "method": system_subqs[i].get("method", "unknown")})
    for i in range(n, len(gt_decomp)):
        results.append({"hop": i, "status": "unmatched_gt"})
    return results


def detect_cascade(hop_results: list[dict]) -> dict:
    """Find first error hop and count subsequent failures."""
    aligned = [r for r in hop_results if r.get("status") == "aligned"]
    first_error = None
    for r in aligned:
        if r["em"] == 0.0 and first_error is None:
            first_error = r["hop"]
    subsequent = sum(
        1 for r in aligned if r["hop"] > (first_error if first_error is not None else 999)
        and r["em"] == 0.0
    )
    return {"first_error_hop": first_error, "cascade_count": subsequent}


def aggregate(bucket: list[dict]) -> dict:
    """Aggregate EM/F1/containment for a list of aligned hop results."""
    aligned = [r for r in bucket if r.get("status") == "aligned"]
    if not aligned:
        return {"n": 0, "em": 0.0, "f1": 0.0, "containment": 0.0}
    n = len(aligned)
    return {
        "n": n,
        "em": round(sum(r["em"] for r in aligned) / n, 4),
        "f1": round(sum(r["f1"] for r in aligned) / n, 4),
        "containment": round(sum(r["containment"] for r in aligned) / n, 4),
    }


def main():
    hybrid = load_hybrid()
    gt = load_musique_gt()

    all_hops = []           # flat list of per-hop results
    per_question = []       # per-question summary
    by_hop = defaultdict(list)
    by_method = defaultdict(list)
    by_type = defaultdict(list)
    by_hop_method = defaultdict(list)

    hop_count_matches = 0
    total_with_subqs = 0

    for qid in sorted(hybrid):
        h = hybrid[qid]
        g = gt[qid]
        sys_subqs = h.get("sub_question_log") or []
        gt_decomp = g["question_decomposition"]
        qtype = h.get("question_type", "unknown")

        hops = align_and_compare(sys_subqs, gt_decomp)
        cascade = detect_cascade(hops)
        judge_correct = h.get("judge_correct", False)

        if sys_subqs:
            total_with_subqs += 1
            if len(sys_subqs) == len(gt_decomp):
                hop_count_matches += 1

        per_question.append({
            "question_id": qid,
            "question_type": qtype,
            "sys_hops": len(sys_subqs),
            "gt_hops": len(gt_decomp),
            "judge_correct": judge_correct,
            **cascade,
        })

        for r in hops:
            r["question_id"] = qid
            r["question_type"] = qtype
            all_hops.append(r)
            if r.get("status") == "aligned":
                by_hop[r["hop"]].append(r)
                by_method[r["method"]].append(r)
                by_type[qtype].append(r)
                by_hop_method[(r["hop"], r["method"])].append(r)

    # ── Cascade analysis ────────────────────────────────────────────────────
    final_failures = [q for q in per_question if not q["judge_correct"]]
    first_hop_wrong = sum(1 for q in final_failures if q["first_error_hop"] == 0)
    no_subqs_failures = sum(
        1 for q in final_failures if hybrid[q["question_id"]].get("sub_question_log") is None
        or len(hybrid[q["question_id"]].get("sub_question_log", [])) == 0
    )
    # "Pure synthesis failure": all aligned hops correct but final answer wrong
    pure_synthesis = sum(
        1 for q in final_failures
        if q["first_error_hop"] is None and q["sys_hops"] > 0
    )

    output = {
        "summary": {
            "total_questions": len(hybrid),
            "total_subqs_system": sum(len(hybrid[q].get("sub_question_log") or []) for q in hybrid),
            "total_subqs_gt": sum(len(gt[q]["question_decomposition"]) for q in gt),
            "questions_with_subqs": total_with_subqs,
            "hop_count_matches": hop_count_matches,
            "hop_count_match_rate": round(hop_count_matches / max(total_with_subqs, 1), 4),
        },
        "per_hop_accuracy": {
            f"hop_{i}": aggregate(by_hop[i]) for i in sorted(by_hop)
        },
        "per_method": {m: aggregate(by_method[m]) for m in sorted(by_method)},
        "per_type": {t: aggregate(by_type[t]) for t in sorted(by_type)},
        "cascade_analysis": {
            "total_final_failures": len(final_failures),
            "first_hop_wrong": first_hop_wrong,
            "first_hop_wrong_pct": round(100 * first_hop_wrong / max(len(final_failures), 1), 1),
            "pure_synthesis_failure": pure_synthesis,
            "no_subqs_failures": no_subqs_failures,
        },
        "per_question": per_question,
    }

    out = save_json(output, "per_hop_accuracy.json")

    # ── Console summary ─────────────────────────────────────────────────────
    print("PER-HOP ACCURACY AUDIT (MuSiQue 1000q)")
    print("=" * 50)
    s = output["summary"]
    print(f"System sub-Qs: {s['total_subqs_system']}  |  GT hops: {s['total_subqs_gt']}")
    print(f"Hop-count match rate: {s['hop_count_match_rate']:.1%}")
    print()
    print(f"{'Hop':<8} {'N':>6} {'EM':>8} {'F1':>8} {'Cont':>8}")
    print("-" * 40)
    for k, v in output["per_hop_accuracy"].items():
        print(f"{k:<8} {v['n']:>6} {v['em']:>8.3f} {v['f1']:>8.3f} {v['containment']:>8.3f}")
    print()
    print(f"{'Method':<10} {'N':>6} {'EM':>8} {'F1':>8} {'Cont':>8}")
    print("-" * 44)
    for k, v in output["per_method"].items():
        print(f"{k:<10} {v['n']:>6} {v['em']:>8.3f} {v['f1']:>8.3f} {v['containment']:>8.3f}")
    print()
    c = output["cascade_analysis"]
    print(f"Cascade: {c['first_hop_wrong']}/{c['total_final_failures']} failures "
          f"({c['first_hop_wrong_pct']}%) had wrong first hop")
    print(f"Pure synthesis failures: {c['pure_synthesis_failure']}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
