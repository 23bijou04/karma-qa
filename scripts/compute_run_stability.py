#!/usr/bin/env python3
"""Compute run1 vs run2 stability metrics for the ThiRAG paper (xp.tex).

Loads judged JSONL files for every (system, benchmark) pair that has two
runs, recomputes Cont. using the canonical normalize_answer() / normalize_date()
from evaluation/metrics.py (the stale metrics_summary.json is ignored),
and reports:

  - mean ± half-range of Cont. and Acc. across the two runs,
  - per-question agreement rate (fraction of IDs where both runs return
    the same judge_correct verdict).

Writes a Markdown report ready to drop into input/xp.tex.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_containment, compute_em  # noqa: E402

REPO = ROOT
REPORT_OUT = REPO.parent / "thirag-paper" / "plans" / "reports" / "run-stability-260420-1605-xp-tables.md"

# (system, benchmark) -> (run1_path, run2_path_or_None)
FILES: dict[tuple[str, str], tuple[Path, Path | None]] = {
    # 2WikiMHQA
    ("ThiRAG", "2WikiMHQA"): (
        REPO / "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl",
        REPO / "data/eval_1000q/hybrid/hybrid_eval_1000_run2_judged.jsonl",
    ),
    ("LinearRAG", "2WikiMHQA"): (
        REPO / "data/eval_1000q/linearrag/linearrag_1000q_judged.jsonl",
        REPO / "data/eval_1000q/linearrag/linearrag_1000q_run2_judged.jsonl",
    ),
    ("LogicRAG", "2WikiMHQA"): (
        REPO / "data/eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl",
        REPO / "data/eval_1000q/logicrag/logicrag_1000q_traced_run2_judged.jsonl",
    ),
    # HotpotQA
    ("ThiRAG", "HotpotQA"): (
        REPO / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl",
        REPO / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_run2_judged.jsonl",
    ),
    ("LinearRAG", "HotpotQA"): (
        REPO / "data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_judged.jsonl",
        REPO / "data/cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_run2_judged.jsonl",
    ),
    ("LogicRAG", "HotpotQA"): (
        REPO / "data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_judged.jsonl",
        REPO / "data/cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_run2_judged.jsonl",
    ),
    # MuSiQue
    ("ThiRAG", "MuSiQue"): (
        REPO / "data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl",
        REPO / "data/cross_benchmark/musique/hybrid_musique_1000q_run2_judged.jsonl",
    ),
    ("LinearRAG", "MuSiQue"): (
        REPO / "data/cross_benchmark/musique/linearrag_musique_1000q_judged.jsonl",
        REPO / "data/cross_benchmark/musique/linearrag_musique_1000q_run2_judged.jsonl",
    ),
    ("LogicRAG", "MuSiQue"): (
        REPO / "data/cross_benchmark/musique/logicrag_musique_1000q_judged.jsonl",
        REPO / "data/cross_benchmark/musique/logicrag_musique_1000q_run2_judged.jsonl",
    ),
}


@dataclass
class RunMetrics:
    n: int
    cont_pct: float  # substring containment, %
    acc_pct: float   # LLM judge accuracy, %
    per_q_correct: dict[str, int]  # qid -> judge_correct (0/1)
    per_type_acc: dict[str, tuple[int, int]]  # qtype -> (n_correct, n_total)


def load_run(path: Path) -> RunMetrics:
    per_q: dict[str, int] = {}
    per_type: dict[str, list[int]] = {}
    cont_sum = 0.0
    acc_sum = 0.0
    n = 0
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            qid = str(row.get("question_id"))
            pred = row.get("predicted_answer") or ""
            gold = row.get("gold_answer") or ""
            cont = compute_containment(pred, gold)
            if cont == 0.0 and row.get("gold_aliases"):
                for alias in row["gold_aliases"]:
                    if compute_containment(pred, alias) == 1.0:
                        cont = 1.0
                        break
            jc = int(bool(row.get("judge_correct")))
            per_q[qid] = jc
            qtype = row.get("question_type") or "unknown"
            per_type.setdefault(qtype, []).append(jc)
            cont_sum += cont
            acc_sum += jc
            n += 1
    return RunMetrics(
        n=n,
        cont_pct=100.0 * cont_sum / n if n else 0.0,
        acc_pct=100.0 * acc_sum / n if n else 0.0,
        per_q_correct=per_q,
        per_type_acc={t: (sum(v), len(v)) for t, v in per_type.items()},
    )


def pair_stats(r1: RunMetrics, r2: RunMetrics | None) -> dict[str, float | str]:
    if r2 is None:
        return {
            "n_run1": r1.n, "n_run2": 0,
            "cont_mean": r1.cont_pct, "cont_half_range": 0.0,
            "acc_mean": r1.acc_pct, "acc_half_range": 0.0,
            "agreement_pct": float("nan"),
            "note": "run1 only",
        }
    # Agreement over shared qids.
    shared = set(r1.per_q_correct) & set(r2.per_q_correct)
    if not shared:
        agreement = float("nan")
    else:
        agree = sum(1 for q in shared if r1.per_q_correct[q] == r2.per_q_correct[q])
        agreement = 100.0 * agree / len(shared)
    cont_mean = statistics.mean([r1.cont_pct, r2.cont_pct])
    cont_half = abs(r1.cont_pct - r2.cont_pct) / 2.0
    acc_mean = statistics.mean([r1.acc_pct, r2.acc_pct])
    acc_half = abs(r1.acc_pct - r2.acc_pct) / 2.0
    return {
        "n_run1": r1.n, "n_run2": r2.n,
        "cont_run1": r1.cont_pct, "cont_run2": r2.cont_pct,
        "acc_run1": r1.acc_pct, "acc_run2": r2.acc_pct,
        "cont_mean": cont_mean, "cont_half_range": cont_half,
        "acc_mean": acc_mean, "acc_half_range": acc_half,
        "agreement_pct": agreement,
        "note": "",
    }


def fmt(x: float, digits: int = 1) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x:.{digits}f}"


def main() -> None:
    results: dict[tuple[str, str], dict] = {}
    for key, (p1, p2) in FILES.items():
        if not p1.exists():
            print(f"[WARN] missing run1: {p1}", file=sys.stderr)
            continue
        r1 = load_run(p1)
        r2 = load_run(p2) if (p2 and p2.exists()) else None
        results[key] = pair_stats(r1, r2)

    lines: list[str] = []
    lines.append("# Run-stability results for xp.tex Tables 1–4\n")
    lines.append("Generated by `scripts/compute_run_stability.py`. "
                 "Cont. and Acc. recomputed via `evaluation/metrics.py::normalize_answer` "
                 "(stale `metrics_summary.json` ignored).\n")

    # Main table (Table 1): mean ± half-range.
    lines.append("## Table 1 — Main results (mean ± half-range across 2 runs)\n")
    lines.append("| System | 2Wiki Cont. | 2Wiki Acc. | Hotpot Cont. | Hotpot Acc. | MuSiQue Cont. | MuSiQue Acc. |")
    lines.append("|---|---|---|---|---|---|---|")
    for sys_name in ["LogicRAG", "LinearRAG", "ThiRAG"]:
        row = [sys_name]
        for bench in ["2WikiMHQA", "HotpotQA", "MuSiQue"]:
            s = results.get((sys_name, bench))
            if not s:
                row.extend(["n/a", "n/a"])
                continue
            mark = "" if not s.get("note") else "†"
            row.append(f"{fmt(s['cont_mean'])}±{fmt(s['cont_half_range'])}{mark}")
            row.append(f"{fmt(s['acc_mean'])}±{fmt(s['acc_half_range'])}{mark}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Raw per-run numbers for auditability.
    lines.append("## Per-run raw numbers\n")
    lines.append("| System | Benchmark | Cont r1 | Cont r2 | Acc r1 | Acc r2 | Agreement |")
    lines.append("|---|---|---|---|---|---|---|")
    for (sys_name, bench), s in sorted(results.items()):
        lines.append(
            f"| {sys_name} | {bench} | "
            f"{fmt(s.get('cont_run1', s['cont_mean']))} | {fmt(s.get('cont_run2', float('nan')))} | "
            f"{fmt(s.get('acc_run1', s['acc_mean']))} | {fmt(s.get('acc_run2', float('nan')))} | "
            f"{fmt(s.get('agreement_pct', float('nan')))}% |"
        )

    # Per-type table (Table 2) on 2WikiMHQA across all three systems.
    lines.append("\n## Table 2 — Per-type judge accuracy on 2WikiMHQA (mean ± half-range)\n")
    type_order = ["bridge_comparison", "comparison", "compositional", "inference"]
    type_label = {
        "bridge_comparison": "bridge\\_comp.",
        "comparison": "comparison",
        "compositional": "compositional",
        "inference": "inference",
    }
    # Recompute per-type acc per run for 2Wiki.
    per_type_cells: dict[tuple[str, str], tuple[float, float, int]] = {}
    for sys_name in ["LogicRAG", "LinearRAG", "ThiRAG"]:
        p1, p2 = FILES[(sys_name, "2WikiMHQA")]
        r1 = load_run(p1)
        r2 = load_run(p2) if p2 and p2.exists() else None
        for t in type_order:
            c1, n1 = r1.per_type_acc.get(t, (0, 0))
            a1 = 100.0 * c1 / n1 if n1 else 0.0
            if r2:
                c2, n2 = r2.per_type_acc.get(t, (0, 0))
                a2 = 100.0 * c2 / n2 if n2 else 0.0
                mean = (a1 + a2) / 2
                half = abs(a1 - a2) / 2
            else:
                mean, half = a1, 0.0
            per_type_cells[(sys_name, t)] = (mean, half, n1)
    lines.append("| Type | NB | ThiRAG | LogicRAG | LinearRAG |")
    lines.append("|---|---|---|---|---|")
    for t in type_order:
        nb = per_type_cells[("ThiRAG", t)][2]
        cells = []
        for sys_name in ["ThiRAG", "LogicRAG", "LinearRAG"]:
            m, h, _ = per_type_cells[(sys_name, t)]
            cells.append(f"{fmt(m)}±{fmt(h)}")
        lines.append(f"| {type_label[t]} | {nb} | " + " | ".join(cells) + " |")
    # All-row:
    allrow = []
    for sys_name in ["ThiRAG", "LogicRAG", "LinearRAG"]:
        s = results[(sys_name, "2WikiMHQA")]
        allrow.append(f"{fmt(s['acc_mean'])}±{fmt(s['acc_half_range'])}")
    lines.append(f"| All | 1000 | " + " | ".join(allrow) + " |")

    # Global stability summary.
    drifts_acc = [s['acc_half_range'] * 2 for s in results.values() if s.get('n_run2')]
    drifts_cont = [s['cont_half_range'] * 2 for s in results.values() if s.get('n_run2')]
    agreements = [s['agreement_pct'] for s in results.values() if s.get('n_run2') and s['agreement_pct'] == s['agreement_pct']]
    lines.append("\n## Global stability\n")
    if drifts_acc:
        lines.append(f"- Acc. drift across runs: mean={fmt(statistics.mean(drifts_acc))} pp, "
                     f"max={fmt(max(drifts_acc))} pp (over {len(drifts_acc)} system×benchmark cells with both runs).")
    if drifts_cont:
        lines.append(f"- Cont. drift across runs: mean={fmt(statistics.mean(drifts_cont))} pp, "
                     f"max={fmt(max(drifts_cont))} pp.")
    if agreements:
        lines.append(f"- Per-question run-to-run agreement: mean={fmt(statistics.mean(agreements))}%, "
                     f"min={fmt(min(agreements))}%.")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_OUT}")
    # Also echo to stdout for quick review.
    print("\n".join(lines))


if __name__ == "__main__":
    main()
