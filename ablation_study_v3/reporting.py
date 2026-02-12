"""Report generation for Ablation Study V3."""

import json
import os
from collections import defaultdict

from .sampling import wilson_confidence_interval, mcnemar_test, bootstrap_ci

LEVEL_NAMES = {
    0: "L0: No KG",
    1: "L1: Entity-Only KG",
    2: "L2: Entity + Important Relations",
    3: "L3: Full KG",
    4: "L4: Oracle KG",
}

BONFERRONI_ALPHA = 0.0125  # 0.05 / 4 tests


def _load_level_results(output_dir: str, level: int) -> list[dict]:
    path = os.path.join(output_dir, f"level_{level}_results.jsonl")
    results = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def generate_report(output_dir: str, levels: list[int]):
    """Generate comprehensive ablation study report."""
    all_results = {}
    for level in levels:
        all_results[level] = _load_level_results(output_dir, level)

    lines = ["# Ablation Study V3: Results Report\n"]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Level | N | EM | F1 | 95% CI (EM) |")
    lines.append("|-------|---|----|----|-------------|")

    for level in levels:
        results = all_results[level]
        if not results:
            continue
        n = len(results)
        em_sum = sum(r["em"] for r in results)
        f1_scores = [r["f1"] for r in results]
        em_mean = em_sum / n
        f1_mean = sum(f1_scores) / n
        ci_low, ci_high = wilson_confidence_interval(int(em_sum), n)
        name = LEVEL_NAMES.get(level, f"L{level}")
        lines.append(
            f"| {name} | {n} | {em_mean:.3f} | {f1_mean:.3f} | "
            f"[{ci_low:.3f}, {ci_high:.3f}] |"
        )

    # By question type
    lines.append("\n## By Question Type\n")
    types_seen = set()
    for results in all_results.values():
        for r in results:
            types_seen.add(r["question_type"])

    type_list = sorted(types_seen)
    header = "| Type | " + " | ".join(LEVEL_NAMES.get(l, f"L{l}") for l in levels) + " |"
    sep = "|------|" + "|".join("------" for _ in levels) + "|"
    lines.append(header)
    lines.append(sep)

    for qtype in type_list:
        row = f"| {qtype} |"
        for level in levels:
            results = [r for r in all_results.get(level, []) if r["question_type"] == qtype]
            if results:
                em = sum(r["em"] for r in results) / len(results)
                row += f" {em:.3f} |"
            else:
                row += " - |"
        lines.append(row)

    # Statistical tests
    lines.append("\n## Statistical Tests (McNemar, Bonferroni alpha=0.0125)\n")
    test_pairs = [(0, 1), (1, 2), (2, 3), (1, 3)]
    lines.append("| Comparison | chi2 | p-value | Significant |")
    lines.append("|------------|------|---------|-------------|")

    for la, lb in test_pairs:
        if la not in all_results or lb not in all_results:
            continue
        ra, rb = all_results[la], all_results[lb]
        if not ra or not rb:
            continue
        # Align by question_id
        id_to_a = {r["question_id"]: r["em"] > 0 for r in ra}
        id_to_b = {r["question_id"]: r["em"] > 0 for r in rb}
        common_ids = sorted(set(id_to_a) & set(id_to_b))
        if not common_ids:
            continue
        a_correct = [id_to_a[qid] for qid in common_ids]
        b_correct = [id_to_b[qid] for qid in common_ids]
        chi2, p = mcnemar_test(a_correct, b_correct)
        sig = "Yes" if p < BONFERRONI_ALPHA else "No"
        na = LEVEL_NAMES.get(la, f"L{la}")
        nb = LEVEL_NAMES.get(lb, f"L{lb}")
        lines.append(f"| {na} vs {nb} | {chi2:.2f} | {p:.4f} | {sig} |")

    # KG contribution analysis (Level 2)
    if 2 in all_results and all_results[2]:
        lines.append("\n## KG Contribution Analysis (Level 2)\n")
        kg_hops = 0
        wiki_hops = 0
        for r in all_results[2]:
            sources = r.get("metadata", {}).get("hop_sources", {})
            for _, src in sources.items():
                if src == "kg":
                    kg_hops += 1
                else:
                    wiki_hops += 1
        total_hops = kg_hops + wiki_hops
        if total_hops > 0:
            lines.append(f"- KG answered: {kg_hops}/{total_hops} hops ({kg_hops/total_hops*100:.1f}%)")
            lines.append(f"- Wikipedia fallback: {wiki_hops}/{total_hops} hops ({wiki_hops/total_hops*100:.1f}%)")

    # Thesis validation
    if 1 in all_results and all_results[1]:
        lines.append("\n## Thesis Validation\n")
        l1_em = sum(r["em"] for r in all_results[1]) / len(all_results[1])

        if 3 in all_results and all_results[3]:
            l3_em = sum(r["em"] for r in all_results[3]) / len(all_results[3])
            delta = l3_em - l1_em
            lines.append(f"- L1 (Entity-Only) EM: {l1_em:.3f}")
            lines.append(f"- L3 (Full KG) EM: {l3_em:.3f}")
            lines.append(f"- Delta (L3 - L1): {delta:+.3f}")
            if abs(delta) <= 0.10:
                lines.append(f"- **Thesis SUPPORTED**: L1 is within 10 points of L3")
            else:
                lines.append(f"- **Thesis NOT supported**: L1 is {abs(delta)*100:.1f} points from L3")

        if 4 in all_results and all_results[4]:
            l4_em = sum(r["em"] for r in all_results[4]) / len(all_results[4])
            lines.append(f"- L4 (Oracle) EM: {l4_em:.3f}")
            lines.append(f"- L1 achieves {l1_em/l4_em*100:.1f}% of oracle ceiling")

    # Error analysis
    lines.append("\n## Error Analysis\n")
    for level in levels:
        results = all_results.get(level, [])
        failures = [r for r in results if r["em"] == 0]
        if not failures:
            continue
        errors = defaultdict(int)
        for r in failures:
            err = r.get("error", "") or r.get("answer_source", "unknown")
            if not err:
                err = "wrong_answer"
            errors[err] += 1

        name = LEVEL_NAMES.get(level, f"L{level}")
        lines.append(f"### {name} ({len(failures)} failures)")
        for err, count in sorted(errors.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- {err}: {count}")

    # Limitations
    lines.append("\n## Limitations\n")
    lines.append(
        "- **Perfect NER assumption**: All levels use entity names from evidence triples, "
        "simulating perfect entity recognition. Real-world performance would additionally "
        "depend on NER/entity extraction quality."
    )
    lines.append(
        "- **L2 vs L3 statistical power**: With n=200, small deltas (2-5 points) may not "
        "reach significance. This comparison is exploratory."
    )

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")

    # Save summary JSON
    summary = {}
    for level in levels:
        results = all_results.get(level, [])
        if results:
            n = len(results)
            em_mean = sum(r["em"] for r in results) / n
            f1_mean = sum(r["f1"] for r in results) / n
            summary[f"level_{level}"] = {
                "name": LEVEL_NAMES.get(level, f"L{level}"),
                "n": n, "em": em_mean, "f1": f1_mean,
            }

    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

    # Print report to console
    print("\n" + report)
