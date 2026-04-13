"""Routing predicate validation: precision/recall of SPARQL routing vs KG-answerability."""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HYBRID_JSONL = ROOT / "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl"
GT_JSON = ROOT / "data/eval_1000q/ground-truth/hop_verification_results.json"
OUT_JSON = ROOT / "data/eval_1000q/ablation/routing_predicate_validation.json"
SEED = 42


def classify_route(rs: str | None) -> str:
    if not rs:
        return "UNKNOWN"
    if "no_entity_uris" in rs or "no_match" in rs:
        return "TEXT"
    # Check DECOMPOSE markers first (they appear in some llm_prop: prefixed strings too)
    if "ask_missing" in rs or "ask_partial" in rs or "incomplete_coverage" in rs:
        return "DECOMPOSE"
    if ("ask_verified" in rs or rs.startswith("rule:direct")
            or rs == "rule:direct_date_comparison"
            or rs == "llm_prop:rule:direct_property"):
        return "SPARQL"
    return "UNKNOWN"


def routing_stage(rs: str | None) -> str:
    if rs and (rs.startswith("llm_prop:") or rs.startswith("precheck_")):
        return "stage2_llm"
    return "stage1_keyword"


def compute_pr(rows: list[dict], gt: dict) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        pred = classify_route(r.get("route_source")) == "SPARQL"
        actual = gt.get(r["question_id"], False)
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 4), "recall": round(r_, 4), "f1": round(f1, 4)}


def pr_by_stage(rows: list[dict], gt: dict) -> dict:
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(routing_stage(r.get("route_source")), []).append(r)
    return {s: compute_pr(rs, gt) for s, rs in buckets.items()}


def stratified_sample(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list] = {"SPARQL": [], "DECOMPOSE": [], "TEXT": []}
    for r in rows:
        cat = classify_route(r.get("route_source"))
        if cat in buckets:
            buckets[cat].append(r)
    targets = {"SPARQL": 40, "DECOMPOSE": 40, "TEXT": 20}
    sample = []
    for cat, count in targets.items():
        for r in rng.sample(buckets[cat], min(count, len(buckets[cat]))):
            sample.append({**r, "_sample_cat": cat})
    return sample


def collect_failures(rows: list[dict], gt: dict, max_n: int = 10) -> dict:
    fps, fns = [], []
    for r in rows:
        cat = classify_route(r.get("route_source"))
        kg = gt.get(r["question_id"], False)
        entry = {k: r.get(k) for k in ("question_id", "question", "route_source", "judge_correct")}
        entry["kg_answerable"] = kg
        if cat == "SPARQL" and not kg:
            fps.append(entry)
        elif cat in ("DECOMPOSE", "TEXT") and kg:
            fns.append({**entry, "routing_category": cat})
    return {"false_positives": fps[:max_n], "false_negatives": fns[:max_n]}


def main():
    with open(HYBRID_JSONL) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(GT_JSON) as f:
        gt_data = json.load(f)
    gt = {r["question_id"]: bool(r.get("all_properties_exist")) for r in gt_data["results"]}

    route_dist = {"SPARQL": 0, "DECOMPOSE": 0, "TEXT": 0, "UNKNOWN": 0}
    for r in rows:
        route_dist[classify_route(r.get("route_source"))] += 1

    full_pr = compute_pr(rows, gt)
    stage_pr = pr_by_stage(rows, gt)
    sample = stratified_sample(rows, SEED)
    sample_pr = compute_pr(sample, gt)
    failures = collect_failures(rows, gt)

    print(f"Route distribution: {route_dist}")
    print(f"Full P/R  precision={full_pr['precision']:.1%}  recall={full_pr['recall']:.1%}  "
          f"f1={full_pr['f1']:.1%}")
    print(f"  TP={full_pr['tp']}  FP={full_pr['fp']}  FN={full_pr['fn']}  TN={full_pr['tn']}")
    for stage, pr in stage_pr.items():
        print(f"  [{stage}] P={pr['precision']:.1%} R={pr['recall']:.1%}  "
              f"TP={pr['tp']} FP={pr['fp']} FN={pr['fn']}")
    print(f"Sample P/R (n={len(sample)}) precision={sample_pr['precision']:.1%} "
          f"recall={sample_pr['recall']:.1%}")

    # ── accuracy impact: how routing errors affect final answer correctness ──
    fp_correct = fp_wrong = fn_correct = fn_wrong = 0
    for r in rows:
        pred_sparql = classify_route(r.get("route_source")) == "SPARQL"
        kg_ans = gt.get(r["question_id"], False)
        correct = r.get("judge_correct", False)
        if pred_sparql and not kg_ans:       # false positive
            if correct:
                fp_correct += 1
            else:
                fp_wrong += 1
        elif not pred_sparql and kg_ans:     # false negative
            if correct:
                fn_correct += 1
            else:
                fn_wrong += 1

    total_wrong = fp_wrong + fn_wrong
    n = len(rows)
    accuracy_impact = {
        "note": "How routing errors (FP/FN) translate to final answer correctness",
        "fp_correct": fp_correct,
        "fp_wrong": fp_wrong,
        "fp_correct_rate": (
            round(fp_correct / (fp_correct + fp_wrong), 4) if (fp_correct + fp_wrong) else None
        ),
        "fn_correct": fn_correct,
        "fn_wrong": fn_wrong,
        "fn_correct_rate": (
            round(fn_correct / (fn_correct + fn_wrong), 4) if (fn_correct + fn_wrong) else None
        ),
        "total_routing_errors_causing_wrong_answer": total_wrong,
        "routing_error_rate_pct": round(total_wrong / n * 100, 1),
        "perfect_routing_ceiling_pp": round(fn_wrong / n * 100, 1),
    }

    print("\nAccuracy impact:")
    if fp_correct + fp_wrong:
        print(f"  FP correct/wrong: {fp_correct}/{fp_wrong}  "
              f"({fp_correct / (fp_correct + fp_wrong):.1%} benign)")
    if fn_correct + fn_wrong:
        print(f"  FN correct/wrong: {fn_correct}/{fn_wrong}  "
              f"({fn_correct / (fn_correct + fn_wrong):.1%} recovered)")
    print(f"  Total routing errors causing wrong answer: {total_wrong}/{n} ({total_wrong / n:.1%})")
    print(f"  Perfect routing ceiling: +{fn_wrong / n * 100:.1f}pp")

    result = {
        "description": "Routing predicate validation: SPARQL-routed as positive prediction "
                       "vs all_properties_exist",
        "dataset": {"total": len(rows), "kg_answerable": int(sum(gt.values()))},
        "route_distribution": route_dist,
        "full_population": {"n": len(rows), "precision_recall": full_pr, "by_stage": stage_pr},
        "stratified_sample": {
            "n": len(sample), "seed": SEED,
            "counts": {"SPARQL": 40, "DECOMPOSE": 40, "TEXT": 20},
            "precision_recall": sample_pr, "by_stage": pr_by_stage(sample, gt),
        },
        "accuracy_impact": accuracy_impact,
        "failure_patterns": failures,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
