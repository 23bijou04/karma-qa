#!/usr/bin/env python3
"""Cross-system Judge EM comparison across all baselines.

Loads judged JSONL files for Hybrid, LogicRAG, SPARQL, GRASP, LinearRAG.
Produces:
- Overall comparison table (hop-368 and full-1000q subsets)
- Per-question-type breakdown
- Wilson CIs for all Judge EM rates
- McNemar pairwise tests on Judge EM

Usage:
    python scripts/judge-comparison.py [--output docs/linearrag-baseline-comparison.md]
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

# Add project root to path for evaluation.metrics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- File paths for each system's judged output ---
JUDGED_FILES = {
    "Hybrid": "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl",
    "LogicRAG": "data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl",
    "SPARQL": "data/eval_1000q/sparql/hop368/sparql_eval_1000_hop368_judged.jsonl",
    "GRASP": "data/eval_1000q/grasp/hop368/grasp_eval_dbpedia_hop368_judged.jsonl",
    "LinearRAG": "data/eval_1000q/linearrag/linearrag_1000q_judged.jsonl",
}

HOP_VERIFICATION = "data/eval_1000q/ground-truth/hop_verification_results.json"


def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for proportion p with n trials."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, centre - spread), min(1, centre + spread)


def mcnemar_chi2(b, c):
    """McNemar chi-squared with continuity correction."""
    if b + c == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)


def load_judged(path):
    """Load judged JSONL → {question_id: record}."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                records[d["question_id"]] = d
    return records


def load_hop368_ids():
    """Load the 368 hop-verified question IDs (all_properties_exist=True)."""
    with open(HOP_VERIFICATION) as f:
        data = json.load(f)
    return {r["question_id"] for r in data["results"] if r.get("all_properties_exist")}


def compute_stats(records, subset_ids=None):
    """Compute Judge EM overall and per question type."""
    if subset_ids is not None:
        records = {qid: r for qid, r in records.items() if qid in subset_ids}

    total = len(records)
    if total == 0:
        return {"total": 0, "correct": 0, "pct": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "by_type": {}}

    correct = sum(1 for r in records.values() if r.get("judge_correct"))
    p = correct / total
    ci_lo, ci_hi = wilson_ci(p, total)

    # Per question type
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records.values():
        qtype = r.get("question_type", "unknown")
        by_type[qtype]["total"] += 1
        if r.get("judge_correct"):
            by_type[qtype]["correct"] += 1

    by_type_out = {}
    for qtype, stats in by_type.items():
        tp = stats["correct"] / stats["total"] if stats["total"] else 0
        tlo, thi = wilson_ci(tp, stats["total"])
        by_type_out[qtype] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "pct": round(tp * 100, 1),
            "ci_lo": round(tlo * 100, 1),
            "ci_hi": round(thi * 100, 1),
        }

    return {
        "total": total,
        "correct": correct,
        "pct": round(p * 100, 1),
        "ci_lo": round(ci_lo * 100, 1),
        "ci_hi": round(ci_hi * 100, 1),
        "by_type": by_type_out,
    }


def pairwise_mcnemar(records_a, records_b, subset_ids=None):
    """McNemar test between two systems on shared questions."""
    if subset_ids is not None:
        records_a = {qid: r for qid, r in records_a.items() if qid in subset_ids}
        records_b = {qid: r for qid, r in records_b.items() if qid in subset_ids}

    shared = set(records_a) & set(records_b)
    b, c = 0, 0  # b = A correct & B wrong, c = A wrong & B correct
    for qid in shared:
        ac = records_a[qid].get("judge_correct", False)
        bc = records_b[qid].get("judge_correct", False)
        if ac and not bc:
            b += 1
        elif not ac and bc:
            c += 1

    chi2 = mcnemar_chi2(b, c)
    # Bonferroni-corrected significance (6 pairwise comparisons → alpha=0.0083)
    sig = chi2 > 6.63  # chi2 > 6.63 ≈ p<0.01 (conservative for Bonferroni)

    return {
        "shared": len(shared),
        "a_only": b,
        "b_only": c,
        "net": c - b,
        "chi2": round(chi2, 2),
        "significant": sig,
    }


def format_table(headers, rows, align=None):
    """Format a markdown table."""
    if align is None:
        align = ["l"] + ["r"] * (len(headers) - 1)

    # Column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Header
    hdr = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_map = {"l": ":---", "r": "---:", "c": ":---:"}
    sep = "| " + " | ".join(
        sep_map[align[i]].ljust(widths[i], "-") for i in range(len(headers))
    ) + " |"

    lines = [hdr, sep]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cross-system Judge EM comparison")
    parser.add_argument("--output", default="docs/linearrag-baseline-comparison.md",
                        help="Output markdown report path")
    args = parser.parse_args()

    # Load all judged files
    all_systems = {}
    for name, path in JUDGED_FILES.items():
        if os.path.exists(path):
            all_systems[name] = load_judged(path)
            print(f"Loaded {name}: {len(all_systems[name])} records from {path}")
        else:
            print(f"SKIP {name}: {path} not found")

    if not all_systems:
        print("No judged files found. Exiting.")
        sys.exit(1)

    hop368_ids = load_hop368_ids()
    print(f"\nHop-368 IDs: {len(hop368_ids)}")

    report = []
    report.append("# Cross-System Judge EM Comparison")
    report.append("")
    report.append("All systems judged with gemma3:12b via Ollama (consistent judge).")
    report.append("Metrics from `evaluation/metrics.py`. Judge EM = semantic equivalence.")
    report.append("")

    # --- Table 1: Overall Judge EM (hop-368) ---
    report.append("## Table 1: Judge EM on Hop-368 (KG-Answerable Subset)")
    report.append("")

    headers = ["System", "N", "Correct", "Judge EM%", "95% CI"]
    rows = []
    for name in ["Hybrid", "LogicRAG", "SPARQL", "GRASP", "LinearRAG"]:
        if name not in all_systems:
            continue
        s = compute_stats(all_systems[name], hop368_ids)
        rows.append([
            name, s["total"], s["correct"],
            f"{s['pct']}%",
            f"[{s['ci_lo']}%, {s['ci_hi']}%]",
        ])
    report.append(format_table(headers, rows))
    report.append("")

    # --- Table 2: Full 1000q Judge EM (only systems with 1000q data) ---
    full_systems = {n: r for n, r in all_systems.items() if len(r) >= 1000}
    if full_systems:
        report.append("## Table 2: Judge EM on Full 1000q")
        report.append("")
        report.append("Only systems evaluated on all 1000 questions shown.")
        report.append("")
        rows = []
        for name in ["Hybrid", "LogicRAG", "LinearRAG"]:
            if name not in full_systems:
                continue
            s = compute_stats(full_systems[name])
            rows.append([
                name, s["total"], s["correct"],
                f"{s['pct']}%",
                f"[{s['ci_lo']}%, {s['ci_hi']}%]",
            ])
        report.append(format_table(headers, rows))
        report.append("")

    # --- Table 3: Per-type breakdown (hop-368) ---
    report.append("## Table 3: Judge EM by Question Type (Hop-368)")
    report.append("")
    type_headers = ["System"]
    # Collect all types
    all_types = set()
    for name, records in all_systems.items():
        s = compute_stats(records, hop368_ids)
        all_types.update(s["by_type"].keys())
    all_types = sorted(all_types)
    type_headers += all_types

    type_rows = []
    for name in ["Hybrid", "LogicRAG", "SPARQL", "GRASP", "LinearRAG"]:
        if name not in all_systems:
            continue
        s = compute_stats(all_systems[name], hop368_ids)
        row = [name]
        for t in all_types:
            ts = s["by_type"].get(t, {"pct": "-", "total": 0})
            if ts["total"] > 0:
                row.append(f"{ts['pct']}% ({ts['total']})")
            else:
                row.append("-")
        type_rows.append(row)
    report.append(format_table(type_headers, type_rows))
    report.append("")

    # --- Table 4: Per-type breakdown (full 1000q) ---
    if full_systems:
        report.append("## Table 4: Judge EM by Question Type (Full 1000q)")
        report.append("")
        type_rows_full = []
        for name in ["Hybrid", "LogicRAG", "LinearRAG"]:
            if name not in full_systems:
                continue
            s = compute_stats(full_systems[name])
            row = [name]
            for t in all_types:
                ts = s["by_type"].get(t, {"pct": "-", "total": 0})
                if ts["total"] > 0:
                    row.append(f"{ts['pct']}% ({ts['total']})")
                else:
                    row.append("-")
            type_rows_full.append(row)
        report.append(format_table(type_headers, type_rows_full))
        report.append("")

    # --- Table 5: McNemar Pairwise Tests (hop-368) ---
    report.append("## Table 5: McNemar Pairwise Tests — Judge EM (Hop-368)")
    report.append("")
    report.append("Bonferroni-corrected alpha=0.0083 (6 comparisons).")
    report.append("")

    mc_headers = ["Comparison", "N", "A only", "B only", "Net", "Chi2", "Significant"]
    mc_rows = []
    pairs = [
        ("Hybrid", "LinearRAG"),
        ("Hybrid", "LogicRAG"),
        ("Hybrid", "SPARQL"),
        ("Hybrid", "GRASP"),
        ("SPARQL", "GRASP"),
        ("SPARQL", "LinearRAG"),
    ]
    for a_name, b_name in pairs:
        if a_name not in all_systems or b_name not in all_systems:
            continue
        mc = pairwise_mcnemar(all_systems[a_name], all_systems[b_name], hop368_ids)
        mc_rows.append([
            f"{a_name} vs {b_name}",
            mc["shared"],
            mc["a_only"],
            mc["b_only"],
            f"{mc['net']:+d}",
            mc["chi2"],
            "Yes" if mc["significant"] else "No",
        ])
    report.append(format_table(mc_headers, mc_rows))
    report.append("")

    # --- Table 6: McNemar on full 1000q ---
    if len(full_systems) >= 2:
        report.append("## Table 6: McNemar Pairwise Tests — Judge EM (Full 1000q)")
        report.append("")
        mc_rows_full = []
        full_pairs = [
            ("Hybrid", "LinearRAG"),
            ("Hybrid", "LogicRAG"),
            ("LogicRAG", "LinearRAG"),
        ]
        for a_name, b_name in full_pairs:
            if a_name not in full_systems or b_name not in full_systems:
                continue
            mc = pairwise_mcnemar(all_systems[a_name], all_systems[b_name])
            mc_rows_full.append([
                f"{a_name} vs {b_name}",
                mc["shared"],
                mc["a_only"],
                mc["b_only"],
                f"{mc['net']:+d}",
                mc["chi2"],
                "Yes" if mc["significant"] else "No",
            ])
        report.append(format_table(mc_headers, mc_rows_full))
        report.append("")

    # --- Methodology notes ---
    report.append("## Methodology")
    report.append("")
    report.append("- **Judge**: gemma3:12b via Ollama (local, same model for all systems)")
    report.append("- **LLM**: llama-3.3-70b-versatile via Groq (all systems except GRASP)")
    report.append("- **GRASP**: uses openai/gpt-5-mini (external system, LLM not controllable)")
    report.append("- **Metrics**: EM, F1, containment from `evaluation/metrics.py`")
    report.append("- **Hop-368**: questions where all properties exist in DBpedia "
                   "(KG-answerable subset)")
    report.append("- **McNemar**: chi-squared with continuity correction, "
                   "Bonferroni alpha=0.0083 for 6 tests")
    report.append("- **Wilson CI**: 95% confidence intervals for proportions")
    report.append("")

    # Write report
    report_text = "\n".join(report)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report_text)
    print(f"\nReport written to: {args.output}")

    # Also print to stdout
    print("\n" + report_text)


if __name__ == "__main__":
    main()
