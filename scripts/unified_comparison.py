#!/usr/bin/env python3
"""Unified comparison of all QA systems on LinearRAG's 1K question set.

Compares Hybrid QA (primary), Schema-Guided SPARQL, GRASP, and LinearRAG.
Produces 5 tables with Wilson CIs and McNemar tests.

Usage:
    python scripts/unified_comparison.py
    python scripts/unified_comparison.py --output-dir data/eval_1000q
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import normalize_answer, compute_em, compute_f1, compute_containment
from ablation_study_v3.sampling import mcnemar_test


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_coverage(path):
    """Load coverage analysis, return answerable IDs and question metadata."""
    with open(path) as f:
        data = json.load(f)
    answerable_ids = set()
    qtype_map = {}
    gold_map = {}
    all_ids = set()
    for q in data["questions"]:
        qid = q["question_id"]
        all_ids.add(qid)
        qtype_map[qid] = q["question_type"]
        gold_map[qid] = q["gold_answer"]
        if q["is_answerable"]:
            answerable_ids.add(qid)
    return all_ids, answerable_ids, qtype_map, gold_map


def load_linearrag_predictions(path):
    """Load LinearRAG predictions (JSON array). Field: id, pred_answer."""
    with open(path) as f:
        preds = json.load(f)
    results = {}
    for p in preds:
        qid = p["id"]
        results[qid] = {
            "predicted_answer": p.get("pred_answer", ""),
            "gold_answer": p.get("gold_answer", ""),
            "question_type": p.get("question_type", ""),
        }
    return results


def load_jsonl_results(path):
    """Load SPARQL, GRASP, or Hybrid JSONL results. Field: question_id."""
    results = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r["question_id"]
            results[qid] = {
                "predicted_answer": r.get("predicted_answer", ""),
                "gold_answer": r.get("gold_answer", ""),
                "question_type": r.get("question_type", ""),
                "method": r.get("method", ""),
            }
    return results


def try_load(path, loader, label):
    """Try loading a file, return empty dict if missing."""
    if not os.path.exists(path):
        print(f"  {label}: FILE NOT FOUND ({path}) — skipping")
        return {}
    data = loader(path)
    print(f"  {label}: {len(data)} results")
    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bootstrap_ci(scores, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap percentile CI for continuous metrics (F1)."""
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(
        sum(rng.choices(scores, k=n)) / n for _ in range(n_boot)
    )
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


def compute_system_metrics(results, gold_override=None):
    """Compute EM, F1, Containment with CIs for a system's results."""
    ems, f1s, conts = [], [], []
    qids = sorted(results.keys())
    for qid in qids:
        r = results[qid]
        pred = r["predicted_answer"] or ""
        gold = gold_override[qid] if gold_override else (r["gold_answer"] or "")
        ems.append(compute_em(pred, gold))
        f1s.append(compute_f1(pred, gold))
        conts.append(compute_containment(pred, gold))

    n = len(ems)
    if n == 0:
        return {"n": 0}

    em_mean = sum(ems) / n
    f1_mean = sum(f1s) / n
    cont_mean = sum(conts) / n

    return {
        "n": n,
        "em": em_mean,
        "f1": f1_mean,
        "containment": cont_mean,
        "em_ci": wilson_ci(em_mean, n),
        "f1_ci": bootstrap_ci(f1s),
        "cont_ci": wilson_ci(cont_mean, n),
        "em_per_q": dict(zip(qids, ems)),
    }


def fmt_pct_ci(val, ci):
    """Format percentage with CI: XX.X% [XX.X-XX.X]"""
    return f"{val*100:.1f}% [{ci[0]*100:.1f}-{ci[1]*100:.1f}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified QA system comparison")
    parser.add_argument("--output-dir", default="data/eval_1000q")
    parser.add_argument("--coverage-file",
                        default="data/eval_1000q/coverage_analysis_1000.json")
    parser.add_argument("--hybrid-file",
                        default="data/eval_1000q/hybrid_eval_1000.jsonl")
    parser.add_argument("--sparql-file",
                        default="data/eval_1000q/sparql_eval_1000.jsonl")
    parser.add_argument("--grasp-file",
                        default="data/eval_1000q/grasp_eval_dbpedia.jsonl")
    parser.add_argument("--linearrag-file",
                        default="data/eval_1000q/linearrag_predictions_1000.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load coverage analysis
    print("Loading coverage analysis...")
    all_ids, answerable_ids, qtype_map, gold_map = load_coverage(args.coverage_file)
    print(f"  Total: {len(all_ids)}, Answerable: {len(answerable_ids)}")

    # 2. Load each system's results
    print("\nLoading system results...")
    hybrid_all = try_load(args.hybrid_file, load_jsonl_results, "Hybrid QA")
    sparql = try_load(args.sparql_file, load_jsonl_results, "Schema-Guided SPARQL")
    grasp = try_load(args.grasp_file, load_jsonl_results, "GRASP")
    linearrag_all = try_load(args.linearrag_file, load_linearrag_predictions, "LinearRAG")

    # Filter to answerable subset
    hybrid_ans = {qid: hybrid_all[qid] for qid in answerable_ids if qid in hybrid_all}
    linearrag_ans = {qid: linearrag_all[qid] for qid in answerable_ids
                     if qid in linearrag_all}

    # 3. Find common answerable set (only systems that have results)
    answerable_systems = {}
    if hybrid_ans:
        answerable_systems["Hybrid QA"] = hybrid_ans
    if sparql:
        answerable_systems["Schema-Guided SPARQL"] = sparql
    if grasp:
        answerable_systems["GRASP"] = grasp
    if linearrag_ans:
        answerable_systems["LinearRAG"] = linearrag_ans

    # Common set = intersection of all available systems
    if answerable_systems:
        common_ids = set.intersection(*(set(d.keys()) for d in answerable_systems.values()))
    else:
        common_ids = set()
    print(f"\n  Common answerable set: {len(common_ids)} questions")

    if len(common_ids) < len(answerable_ids) * 0.5 and len(answerable_systems) > 1:
        for name, data in answerable_systems.items():
            missing = answerable_ids - set(data.keys())
            print(f"  WARNING: {name} missing {len(missing)} answerable questions")

    # 4. Verify gold_answer consistency
    print("\nVerifying gold_answer consistency...")
    mismatches = 0
    for qid in common_ids:
        golds = set()
        for data in answerable_systems.values():
            if qid in data and data[qid].get("gold_answer"):
                golds.add(normalize_answer(data[qid]["gold_answer"]))
        if len(golds) > 1:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH {qid}: {golds}")
    if mismatches:
        print(f"  {mismatches} mismatches — using coverage gold_answer as ground truth")
    else:
        print("  All gold_answers consistent")

    # =========================================================================
    # TABLE 0: Per-system coverage (each on its own answerable results)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLE 0: Per-System Coverage on Answerable Questions")
    print("=" * 80)

    table0 = {}
    for name, data in answerable_systems.items():
        m = compute_system_metrics(data, gold_override=gold_map)
        table0[name] = m
        coverage_pct = m["n"] / len(answerable_ids) * 100 if answerable_ids else 0
        print(f"\n  {name} (N={m['n']}/{len(answerable_ids)}, {coverage_pct:.0f}% coverage)")
        print(f"    EM:          {fmt_pct_ci(m['em'], m['em_ci'])}")
        print(f"    F1:          {fmt_pct_ci(m['f1'], m['f1_ci'])}")
        print(f"    Containment: {fmt_pct_ci(m['containment'], m['cont_ci'])}")

    # =========================================================================
    # TABLE 1: Head-to-head on common answerable subset
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLE 1: System Comparison on Common Answerable Subset")
    print("=" * 80)

    systems_common = {
        name: {qid: data[qid] for qid in common_ids if qid in data}
        for name, data in answerable_systems.items()
    }

    table1 = {}
    for name, data in systems_common.items():
        m = compute_system_metrics(data, gold_override=gold_map)
        table1[name] = m
        print(f"\n  {name} (N={m['n']})")
        print(f"    EM:          {fmt_pct_ci(m['em'], m['em_ci'])}")
        print(f"    F1:          {fmt_pct_ci(m['f1'], m['f1_ci'])}")
        print(f"    Containment: {fmt_pct_ci(m['containment'], m['cont_ci'])}")

    # =========================================================================
    # TABLE 2: By question type (answerable)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLE 2: By Question Type — EM% / F1% (Answerable Subset)")
    print("=" * 80)

    types = sorted(set(qtype_map.get(qid, "unknown") for qid in common_ids))
    table2 = {}
    for name, data in systems_common.items():
        table2[name] = {}
        for qtype in types:
            type_data = {qid: data[qid] for qid in common_ids
                         if qtype_map.get(qid) == qtype and qid in data}
            if type_data:
                table2[name][qtype] = compute_system_metrics(
                    type_data, gold_override=gold_map
                )

    header = f"  {'System':<25}"
    for qtype in types:
        header += f" | {qtype:>18}"
    print(header)
    print("  " + "-" * len(header))
    for name in systems_common:
        row = f"  {name:<25}"
        for qtype in types:
            m = table2[name].get(qtype, {})
            if m:
                row += f" | {m['em']*100:5.1f}% / {m['f1']*100:5.1f}%"
            else:
                row += f" | {'N/A':>18}"
        print(row)

    # =========================================================================
    # TABLE 3: Hybrid QA on ALL 1,000 questions (by route)
    # =========================================================================
    table3_hybrid = {}
    if hybrid_all:
        print("\n" + "=" * 80)
        print("TABLE 3: Hybrid QA — Full 1,000 Questions (by Route)")
        print("=" * 80)

        # Overall
        hybrid_all_metrics = compute_system_metrics(hybrid_all, gold_override=gold_map)
        table3_hybrid["all"] = hybrid_all_metrics
        print(f"\n  Overall (N={hybrid_all_metrics['n']})")
        print(f"    EM:          {fmt_pct_ci(hybrid_all_metrics['em'], hybrid_all_metrics['em_ci'])}")
        print(f"    F1:          {fmt_pct_ci(hybrid_all_metrics['f1'], hybrid_all_metrics['f1_ci'])}")
        print(f"    Containment: {fmt_pct_ci(hybrid_all_metrics['containment'], hybrid_all_metrics['cont_ci'])}")

        # By route (method)
        by_route = defaultdict(dict)
        for qid, r in hybrid_all.items():
            method = r.get("method", "unknown")
            by_route[method][qid] = r

        for route in sorted(by_route.keys()):
            route_data = by_route[route]
            m = compute_system_metrics(route_data, gold_override=gold_map)
            table3_hybrid[route] = m
            print(f"\n  Route: {route} (N={m['n']})")
            print(f"    EM:          {fmt_pct_ci(m['em'], m['em_ci'])}")
            print(f"    F1:          {fmt_pct_ci(m['f1'], m['f1_ci'])}")

        # By question type
        print(f"\n  By question type (all routes):")
        for qtype in types:
            type_data = {qid: hybrid_all[qid] for qid in hybrid_all
                         if qtype_map.get(qid) == qtype}
            if type_data:
                m = compute_system_metrics(type_data, gold_override=gold_map)
                print(f"    {qtype}: EM={m['em']*100:.1f}%, F1={m['f1']*100:.1f}% (N={m['n']})")

    # =========================================================================
    # TABLE 4: LinearRAG full 1,000 vs subset
    # =========================================================================
    table4_linearrag = {}
    if linearrag_all:
        print("\n" + "=" * 80)
        print("TABLE 4: LinearRAG — Full 1,000 vs Answerable Subset")
        print("=" * 80)

        lr_all_m = compute_system_metrics(linearrag_all)
        lr_sub_m = compute_system_metrics(
            {qid: linearrag_ans[qid] for qid in common_ids if qid in linearrag_ans},
            gold_override=gold_map,
        )
        table4_linearrag = {"all_1000": lr_all_m, "answerable_subset": lr_sub_m}

        print(f"\n  {'Metric':<15} | {'All 1,000':>25} | {'Answerable Subset':>25}")
        print("  " + "-" * 70)
        for metric in ["em", "f1", "containment"]:
            ci_key = f"{metric}_ci" if metric != "containment" else "cont_ci"
            if lr_all_m.get("n", 0) > 0 and lr_sub_m.get("n", 0) > 0:
                print(f"  {metric.upper():<15} | "
                      f"{fmt_pct_ci(lr_all_m[metric], lr_all_m[ci_key]):>25} | "
                      f"{fmt_pct_ci(lr_sub_m[metric], lr_sub_m[ci_key]):>25}")

    # =========================================================================
    # TABLE 5: McNemar pairwise significance
    # =========================================================================
    system_names = list(table1.keys())
    n_systems = len(system_names)

    if n_systems >= 2 and common_ids:
        print("\n" + "=" * 80)
        print("TABLE 5: McNemar Pairwise Significance Tests")
        print("=" * 80)

        n_pairs = n_systems * (n_systems - 1) // 2
        ALPHA = 0.05 / n_pairs  # Bonferroni
        table5 = []
        qid_list = sorted(common_ids)

        for sys_a, sys_b in combinations(system_names, 2):
            em_a = [table1[sys_a]["em_per_q"].get(qid, 0) > 0 for qid in qid_list]
            em_b = [table1[sys_b]["em_per_q"].get(qid, 0) > 0 for qid in qid_list]
            chi2, p = mcnemar_test(em_a, em_b)
            sig = "Yes" if p < ALPHA else "No"
            table5.append({
                "pair": f"{sys_a} vs {sys_b}",
                "chi2": chi2, "p_value": p, "significant": sig,
            })
            print(f"  {sys_a} vs {sys_b}: chi2={chi2:.2f}, p={p:.4f}, sig={sig}")

        print(f"\n  Bonferroni-corrected alpha = {ALPHA:.4f} ({n_pairs} pairs)")
    else:
        table5 = []
        ALPHA = 0.05

    # =========================================================================
    # Save JSON
    # =========================================================================
    output = {
        "metadata": {
            "total_questions": len(all_ids),
            "answerable_count": len(answerable_ids),
            "common_question_count": len(common_ids),
            "systems_evaluated": system_names,
            "bonferroni_alpha": ALPHA,
        },
        "table0_per_system_coverage": {
            name: {k: v for k, v in m.items() if k != "em_per_q"}
            for name, m in table0.items()
        },
        "table1_answerable_comparison": {
            name: {k: v for k, v in m.items() if k != "em_per_q"}
            for name, m in table1.items()
        },
        "table2_by_type": {
            name: {
                qtype: {k: v for k, v in m.items() if k != "em_per_q"}
                for qtype, m in types_data.items()
            }
            for name, types_data in table2.items()
        },
        "table3_hybrid_full": {
            route: {k: v for k, v in m.items() if k != "em_per_q"}
            for route, m in table3_hybrid.items()
        } if table3_hybrid else {},
        "table4_linearrag_full": {
            k: {k2: v2 for k2, v2 in m.items() if k2 != "em_per_q"}
            for k, m in table4_linearrag.items()
        } if table4_linearrag else {},
        "table5_mcnemar": table5,
        "per_question_data": {
            qid: {
                "question_type": qtype_map.get(qid, ""),
                "gold_answer": gold_map.get(qid, ""),
                "hybrid_pred": hybrid_all.get(qid, {}).get("predicted_answer", ""),
                "hybrid_method": hybrid_all.get(qid, {}).get("method", ""),
                "sparql_pred": sparql.get(qid, {}).get("predicted_answer", ""),
                "grasp_pred": grasp.get(qid, {}).get("predicted_answer", ""),
                "linearrag_pred": linearrag_all.get(qid, {}).get("predicted_answer", ""),
            }
            for qid in all_ids
        },
    }

    json_path = os.path.join(args.output_dir, "unified_comparison_1000.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # =========================================================================
    # Markdown report
    # =========================================================================
    report_dir = os.path.join(
        PROJECT_ROOT,
        "plans/260218-1800-unified-1000q-baseline-evaluation/reports",
    )
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, "unified-comparison-results.md")

    with open(md_path, "w") as f:
        f.write("# Unified Baseline Comparison: 1,000-Question Evaluation\n\n")
        f.write("## Evaluation Setup\n\n")
        f.write(f"- **Question set**: 1,000 questions from LinearRAG HF dataset "
                f"(`Zly0523/linear-rag`)\n")
        f.write(f"- **Answerable subset**: {len(answerable_ids)} questions "
                f"({len(answerable_ids)/10:.1f}%) with DBpedia entity+property coverage\n")
        f.write(f"- **Common evaluation set**: {len(common_ids)} questions "
                f"(intersection of all systems with results)\n")
        f.write(f"- **Systems**: {', '.join(system_names)}\n")
        f.write(f"- **LLM**: Groq `llama-3.3-70b-versatile` (Hybrid, SPARQL, LinearRAG); "
                f"`gpt-5-mini` (GRASP)\n\n")

        # GRASP coverage note
        if grasp:
            grasp_n = len(grasp)
            grasp_pct = grasp_n / len(answerable_ids) * 100 if answerable_ids else 0
            if grasp_pct < 95:
                f.write(f"> **Note**: GRASP evaluated {grasp_n}/{len(answerable_ids)} "
                        f"answerable questions ({grasp_pct:.0f}% coverage) due to WebSocket "
                        f"server errors (HTTP 503). Common-set comparisons are restricted "
                        f"to the {len(common_ids)}-question intersection.\n\n")

        # Table 0
        f.write("## Table 0: Per-System Coverage\n\n")
        f.write("Each system evaluated on all its available answerable results "
                "(no intersection requirement).\n\n")
        f.write("| System | Coverage | EM (95% CI) | F1 (95% CI) "
                "| Containment (95% CI) |\n")
        f.write("|--------|----------|-------------|-------------|"
                "----------------------|\n")
        for name in answerable_systems:
            m = table0[name]
            cov = f"{m['n']}/{len(answerable_ids)}"
            f.write(f"| {name} | {cov} | "
                    f"{fmt_pct_ci(m['em'], m['em_ci'])} | "
                    f"{fmt_pct_ci(m['f1'], m['f1_ci'])} | "
                    f"{fmt_pct_ci(m['containment'], m['cont_ci'])} |\n")

        # Table 1
        f.write(f"\n## Table 1: System Comparison on Common Subset (N={len(common_ids)})\n\n")
        f.write("| System | N | EM (95% CI) | F1 (95% CI) | Containment (95% CI) |\n")
        f.write("|--------|---|-------------|-------------|----------------------|\n")
        for name in systems_common:
            m = table1[name]
            f.write(f"| {name} | {m['n']} | "
                    f"{fmt_pct_ci(m['em'], m['em_ci'])} | "
                    f"{fmt_pct_ci(m['f1'], m['f1_ci'])} | "
                    f"{fmt_pct_ci(m['containment'], m['cont_ci'])} |\n")

        # Table 2
        f.write("\n## Table 2: By Question Type — EM% / F1% (Answerable)\n\n")
        header = "| System |"
        sep_line = "|--------|"
        for qtype in types:
            header += f" {qtype} |"
            sep_line += "------------|"
        f.write(header + "\n" + sep_line + "\n")
        for name in systems_common:
            row = f"| {name} |"
            for qtype in types:
                m = table2[name].get(qtype, {})
                if m:
                    row += f" {m['em']*100:.1f} / {m['f1']*100:.1f} |"
                else:
                    row += " N/A |"
            f.write(row + "\n")

        # Table 3: Hybrid full
        if table3_hybrid:
            f.write("\n## Table 3: Hybrid QA — Full 1,000 Questions\n\n")
            f.write("| Scope | N | EM (95% CI) | F1 (95% CI) | Containment (95% CI) |\n")
            f.write("|-------|---|-------------|-------------|----------------------|\n")
            for route in ["all"] + sorted(k for k in table3_hybrid if k != "all"):
                m = table3_hybrid[route]
                if m.get("n", 0) > 0:
                    label = "Overall" if route == "all" else f"Route: {route}"
                    f.write(f"| {label} | {m['n']} | "
                            f"{fmt_pct_ci(m['em'], m['em_ci'])} | "
                            f"{fmt_pct_ci(m['f1'], m['f1_ci'])} | "
                            f"{fmt_pct_ci(m['containment'], m['cont_ci'])} |\n")

        # Table 4: LinearRAG full
        if table4_linearrag:
            f.write("\n## Table 4: LinearRAG — Full 1,000 vs Answerable\n\n")
            f.write("| Metric | All 1,000 | Answerable Subset |\n")
            f.write("|--------|-----------|-------------------|\n")
            a = table4_linearrag.get("all_1000", {})
            s = table4_linearrag.get("answerable_subset", {})
            if a.get("n", 0) > 0 and s.get("n", 0) > 0:
                for metric in ["em", "f1", "containment"]:
                    ci_key = f"{metric}_ci" if metric != "containment" else "cont_ci"
                    f.write(f"| {metric.upper()} | "
                            f"{fmt_pct_ci(a[metric], a[ci_key])} | "
                            f"{fmt_pct_ci(s[metric], s[ci_key])} |\n")

        # Table 5: McNemar
        if table5:
            f.write(f"\n## Table 5: McNemar Pairwise Significance\n\n")
            f.write(f"Bonferroni-corrected alpha = {ALPHA:.4f} ({n_pairs} pairs)\n\n")
            f.write("| Pair | chi2 | p-value | Significant? |\n")
            f.write("|------|------|---------|-------------|\n")
            for t in table5:
                f.write(f"| {t['pair']} | {t['chi2']:.2f} | "
                        f"{t['p_value']:.4f} | {t['significant']} |\n")

    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
