#!/usr/bin/env python3
"""Statistical analysis for cross-benchmark comparison (3 systems x 3 benchmarks).

Computes McNemar tests (pairwise, Bonferroni-corrected), Wilson CIs,
per-question-type breakdowns, KG augmentation analysis, and generates
a paper-ready markdown report.

Usage:
    python scripts/cross-benchmark-statistical-analysis.py
    python scripts/cross-benchmark-statistical-analysis.py --output data/cross_benchmark/cross_benchmark_comparison.md
"""

import argparse
import importlib.util
import json
import math
import os
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load project metrics for on-the-fly EM/F1/containment computation
_spec = importlib.util.spec_from_file_location(
    "metrics", os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
_compute_em = _metrics.compute_em
_compute_f1 = _metrics.compute_f1
_compute_containment = _metrics.compute_containment
CROSS_DIR = os.path.join(PROJECT_ROOT, "data", "cross_benchmark")

SYSTEMS = ["hybrid", "logicrag", "linearrag"]
BENCHMARKS = ["2wiki", "hotpotqa", "musique"]
PAIRS = [("hybrid", "logicrag"), ("hybrid", "linearrag"), ("logicrag", "linearrag")]

# Bonferroni correction: 9 tests (3 pairs x 3 benchmarks)
ALPHA = 0.05 / 9  # ~0.0056

# File paths
EVAL_PATHS = {
    "hotpotqa": {
        "hybrid": os.path.join(CROSS_DIR, "hotpotqa", "hybrid_hotpotqa_1000q_judged.jsonl"),
        "logicrag": os.path.join(CROSS_DIR, "hotpotqa", "logicrag_hotpotqa_1000q_judged.jsonl"),
        "linearrag": os.path.join(CROSS_DIR, "hotpotqa", "linearrag_hotpotqa_1000q_judged.jsonl"),
    },
    "musique": {
        "hybrid": os.path.join(CROSS_DIR, "musique", "hybrid_musique_1000q_judged.jsonl"),
        "logicrag": os.path.join(CROSS_DIR, "musique", "logicrag_musique_1000q_judged.jsonl"),
        "linearrag": os.path.join(CROSS_DIR, "musique", "linearrag_musique_1000q_judged.jsonl"),
    },
    "2wiki": {
        "hybrid": os.path.join(PROJECT_ROOT, "data", "eval_1000q", "hybrid",
                               "hybrid_eval_1000_judged.jsonl"),
        "logicrag": os.path.join(PROJECT_ROOT, "data", "eval_1000q", "logicrag",
                                 "logicrag_1000q_traced_judged.jsonl"),
        "linearrag": os.path.join(PROJECT_ROOT, "data", "eval_1000q", "linearrag",
                                  "linearrag_1000q_judged.jsonl"),
    },
}


def load_judged(path: str) -> dict[str, dict]:
    """Load judged JSONL into {question_id: record} dict.

    Computes em/f1/containment from predicted_answer + gold_answer
    when those metric fields are absent (e.g. LinearRAG 2Wiki data).
    """
    results = {}
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    qid = d.get("question_id", "")
                    if not qid:
                        continue
                    # Compute string metrics when absent
                    if "em" not in d and d.get("predicted_answer") and d.get("gold_answer"):
                        pred = d["predicted_answer"]
                        gold = d["gold_answer"]
                        aliases = d.get("gold_aliases", [])
                        candidates = [gold] + [a for a in aliases if a]
                        d["em"] = max(_compute_em(pred, c) for c in candidates)
                        d["f1"] = max(_compute_f1(pred, c) for c in candidates)
                        d["containment"] = max(_compute_containment(pred, c) for c in candidates)
                    results[qid] = d
                except json.JSONDecodeError:
                    pass
    return results


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_test(data_a: dict, data_b: dict) -> dict:
    """McNemar test on paired judge_correct outcomes with continuity correction."""
    shared_ids = set(data_a.keys()) & set(data_b.keys())
    # b = A correct + B wrong, c = A wrong + B correct
    b, c = 0, 0
    for qid in shared_ids:
        ac = data_a[qid].get("judge_correct", False)
        bc = data_b[qid].get("judge_correct", False)
        if ac and not bc:
            b += 1
        elif not ac and bc:
            c += 1

    if (b + c) == 0:
        chi2, p_value = 0.0, 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # Exact p-value from chi-squared distribution with 1 df
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, df=1))

    return {
        "shared": len(shared_ids),
        "a_only_correct": b,
        "b_only_correct": c,
        "net_b_minus_a": c - b,
        "chi2": round(chi2, 2),
        "p_value": p_value,
        "significant": p_value < ALPHA,
    }


def aggregate_accuracy(data: dict) -> dict:
    """Compute Judge EM, EM, F1, Containment from records."""
    n = len(data)
    if n == 0:
        return {"n": 0}
    judge_correct = sum(1 for r in data.values() if r.get("judge_correct", False))
    em = sum(r.get("em", 0.0) for r in data.values())
    f1 = sum(r.get("f1", 0.0) for r in data.values())
    cont = sum(r.get("containment", 0.0) for r in data.values())
    lo, hi = wilson_ci(judge_correct, n)
    return {
        "n": n,
        "judge_em": round(judge_correct / n * 100, 1),
        "judge_em_ci": f"{lo * 100:.1f}-{hi * 100:.1f}",
        "em": round(em / n * 100, 1),
        "f1": round(f1 / n * 100, 1),
        "containment": round(cont / n * 100, 1),
    }


def per_type_breakdown(data: dict) -> dict:
    """Compute Judge EM per question_type."""
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in data.values():
        qtype = r.get("question_type", "unknown")
        by_type[qtype]["total"] += 1
        if r.get("judge_correct", False):
            by_type[qtype]["correct"] += 1
    return {
        k: {
            "total": v["total"],
            "correct": v["correct"],
            "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
        }
        for k, v in sorted(by_type.items())
    }


def kg_augmentation_analysis(data: dict) -> dict:
    """Analyze KG augmentation rate for Hybrid system."""
    n = len(data)
    if n == 0:
        return {}
    kg_linked = sum(1 for r in data.values() if r.get("dbpedia_linked", False))
    # Accuracy with vs without KG
    with_kg = [r for r in data.values() if r.get("dbpedia_linked", False)]
    without_kg = [r for r in data.values() if not r.get("dbpedia_linked", False)]
    kg_acc = sum(1 for r in with_kg if r.get("judge_correct", False))
    nokg_acc = sum(1 for r in without_kg if r.get("judge_correct", False))
    return {
        "kg_hit_rate": round(kg_linked / n * 100, 1),
        "kg_linked": kg_linked,
        "with_kg_accuracy": round(kg_acc / len(with_kg) * 100, 1) if with_kg else 0,
        "without_kg_accuracy": round(nokg_acc / len(without_kg) * 100, 1) if without_kg else 0,
        "kg_advantage": round(
            (kg_acc / len(with_kg) - nokg_acc / len(without_kg)) * 100, 1
        ) if with_kg and without_kg else 0,
    }


def generate_report(all_data: dict, all_agg: dict, all_mcnemar: dict,
                    all_per_type: dict, all_kg: dict) -> str:
    """Generate markdown report."""
    lines = [
        "# Cross-Benchmark Comparison Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Setup",
        "- LLM backbone: llama-3.3-70b-versatile (Groq) for all systems",
        "- Judge: gemma3:12b (Ollama) for all evaluations",
        "- Metrics: EM, F1, Containment (automated), Judge EM (semantic judge)",
        f"- Statistical: McNemar test, Bonferroni-corrected alpha = {ALPHA:.4f}",
        "- Retrieval corpora: 2Wiki ~6.1K snippets, HotpotQA ~9.8K passages, MuSiQue ~11.7K passages",
        "- Hybrid additionally queries live DBpedia SPARQL + Spotlight APIs for entity linking and schema discovery",
        "",
        "## 3x3 Comparison Table (Judge EM %)",
        "",
    ]

    # Main table
    header = "| System    | 2Wiki | HotpotQA† | MuSiQue |"
    sep = "|-----------|------:|----------:|--------:|"
    lines.append(header)
    lines.append(sep)

    for sys_name in SYSTEMS:
        row = f"| {sys_name.capitalize():<9s} |"
        for bench in BENCHMARKS:
            d = all_agg.get(bench, {}).get(sys_name, {})
            if d.get("n", 0) == 0:
                row += " — |"
            else:
                row += f" {d['judge_em']}% ({d['judge_em_ci']}) |"
        lines.append(row)

    lines.append("| LinearRAG (pub.‡) | 63.7% | 66.5% | 37.0% |")
    lines.append("")
    lines.append("† Closed retrieval corpus; Hybrid additionally queries live DBpedia APIs")
    lines.append("‡ LinearRAG paper: GPT-4o-mini backbone + GPT-4o-mini judge (GPT-Acc). "
                 "Published GPT-Acc not directly comparable to our Judge EM "
                 "(different judge model and metric).")
    lines.append("")

    # EM and F1 table
    lines.append("## Automated Metrics (EM / F1 %)")
    lines.append("")
    lines.append("| System | 2Wiki EM/F1 | HotpotQA EM/F1 | MuSiQue EM/F1 |")
    lines.append("|--------|-------------|----------------|---------------|")
    for sys_name in SYSTEMS:
        row = f"| {sys_name.capitalize():<7s} |"
        for bench in BENCHMARKS:
            d = all_agg.get(bench, {}).get(sys_name, {})
            if d.get("n", 0) == 0:
                row += " — |"
            else:
                row += f" {d['em']}/{d['f1']} |"
        lines.append(row)
    lines.append("")

    # Pairwise significance
    lines.append("## Pairwise Significance (McNemar, Bonferroni α=0.0056)")
    lines.append("")
    lines.append("| Comparison | 2Wiki | HotpotQA | MuSiQue |")
    lines.append("|------------|------:|---------:|--------:|")
    for (a, b) in PAIRS:
        row = f"| {a.capitalize()} vs {b.capitalize()} |"
        for bench in BENCHMARKS:
            mc = all_mcnemar.get(bench, {}).get(f"{a}_vs_{b}", {})
            if not mc:
                row += " — |"
            else:
                sig = "**sig**" if mc.get("significant") else "ns"
                row += f" χ²={mc['chi2']}, {sig} (net {mc['net_b_minus_a']:+d}) |"
        lines.append(row)
    lines.append("")

    # Per-type breakdowns
    for bench in BENCHMARKS:
        types_data = all_per_type.get(bench, {})
        if not types_data:
            continue
        label = "2WikiMultiHopQA" if bench == "2wiki" else bench.capitalize()
        lines.append(f"## Per-Type Breakdown: {label}")
        lines.append("")

        # Collect all types
        all_types = set()
        for sys_name in SYSTEMS:
            all_types.update(types_data.get(sys_name, {}).keys())

        header = "| Type |"
        sep = "|------|"
        for sys_name in SYSTEMS:
            header += f" {sys_name.capitalize():>10s} |"
            sep += "-----------:|"
        lines.append(header)
        lines.append(sep)

        for qtype in sorted(all_types):
            row = f"| {qtype:<15s} |"
            for sys_name in SYSTEMS:
                td = types_data.get(sys_name, {}).get(qtype, {})
                if not td:
                    row += f" {'—':>10s} |"
                else:
                    row += f" {td['pct']}% ({td['correct']}/{td['total']}) |"
            lines.append(row)
        lines.append("")

    # KG augmentation analysis
    lines.append("## KG Augmentation Analysis (Hybrid only)")
    lines.append("")
    lines.append("| Benchmark | KG Hit Rate | With KG Acc | Without KG Acc | Δ |")
    lines.append("|-----------|------------:|------------:|---------------:|--:|")
    for bench in ["hotpotqa", "musique"]:
        kg = all_kg.get(bench, {})
        if not kg:
            continue
        lines.append(
            f"| {bench.capitalize()} | {kg.get('kg_hit_rate', 0)}% | "
            f"{kg.get('with_kg_accuracy', 0)}% | "
            f"{kg.get('without_kg_accuracy', 0)}% | "
            f"{kg.get('kg_advantage', 0):+.1f}pp |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark statistical analysis")
    parser.add_argument(
        "--output",
        default=os.path.join(CROSS_DIR, "cross_benchmark_comparison.md"),
    )
    parser.add_argument(
        "--json-output",
        default=os.path.join(CROSS_DIR, "statistical_analysis.json"),
    )
    args = parser.parse_args()

    # Load all data
    all_data = {}  # {bench: {system: {qid: record}}}
    for bench in BENCHMARKS:
        all_data[bench] = {}
        for sys_name in SYSTEMS:
            path = EVAL_PATHS[bench][sys_name]
            data = load_judged(path)
            all_data[bench][sys_name] = data
            if data:
                print(f"  Loaded {sys_name}/{bench}: {len(data)} records from {path}")
            else:
                print(f"  WARN: No data for {sys_name}/{bench} at {path}")

    # Compute aggregates
    all_agg = {}
    for bench in BENCHMARKS:
        all_agg[bench] = {}
        for sys_name in SYSTEMS:
            all_agg[bench][sys_name] = aggregate_accuracy(all_data[bench][sys_name])

    # McNemar tests
    all_mcnemar = {}
    for bench in BENCHMARKS:
        all_mcnemar[bench] = {}
        for (a, b) in PAIRS:
            mc = mcnemar_test(all_data[bench][a], all_data[bench][b])
            all_mcnemar[bench][f"{a}_vs_{b}"] = mc
            if mc["shared"] > 0:
                sig = "SIG" if mc["significant"] else "ns"
                print(f"  McNemar {a} vs {b} ({bench}): "
                      f"χ²={mc['chi2']}, net={mc['net_b_minus_a']:+d}, {sig}")

    # Per-type breakdowns
    all_per_type = {}
    for bench in BENCHMARKS:
        all_per_type[bench] = {}
        for sys_name in SYSTEMS:
            all_per_type[bench][sys_name] = per_type_breakdown(all_data[bench][sys_name])

    # KG augmentation analysis (Hybrid only, cross-benchmark)
    all_kg = {}
    for bench in ["hotpotqa", "musique"]:
        hybrid_data = all_data[bench].get("hybrid", {})
        if hybrid_data:
            all_kg[bench] = kg_augmentation_analysis(hybrid_data)

    # Generate report
    report = generate_report(all_data, all_agg, all_mcnemar, all_per_type, all_kg)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport: {args.output}")

    # Save JSON
    json_out = {
        "aggregates": all_agg,
        "mcnemar": all_mcnemar,
        "per_type": all_per_type,
        "kg_augmentation": all_kg,
    }
    with open(args.json_output, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON:   {args.json_output}")


if __name__ == "__main__":
    main()
