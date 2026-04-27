"""Per-type accuracy plot (paper supplement for Table 3).

Loads 9 jsonl (3 benchmarks x 3 systems), computes per-type accuracy with
Wilson 95% CIs, runs paired McNemar (Hybrid vs best baseline) with per-panel
Bonferroni correction, writes CSV audit, asserts 2Wiki values match paper
Table 3.

Renders ONE horizontal grouped-bar figure with all 3 benchmarks stacked
vertically (one row per question_type per benchmark). MuSiQue types merged
to 2hop/3hop/4hop in main figure; see plot-per-type-musique-6types.py for
appendix breakdown.

Usage:
    python scripts/plot-per-type-cross-benchmark.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load sibling _plot_utils.py (kebab-case filenames aren't directly importable).
_UTILS_PATH = Path(__file__).parent / "_plot_utils.py"
_spec = importlib.util.spec_from_file_location("_plot_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
sys.modules["_plot_utils"] = _utils
_spec.loader.exec_module(_utils)  # type: ignore[union-attr]

OKABE_ITO = _utils.OKABE_ITO
MARKERS = _utils.MARKERS
LNCS_RCPARAMS = _utils.LNCS_RCPARAMS
wilson_ci = _utils.wilson_ci
mcnemar_exact = _utils.mcnemar_exact

# ---------------------------------------------------------------------------
# Configuration

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = Path("~/Documents/2026/2026_02/thirag-paper/figures").expanduser()
DATA_OUT_DIR = FIG_DIR / "data"

# Paper Table 2/3 reports mean +/- half-range across two runs. Provide run1
# and run2 files per (benchmark, system); run2=None = single-run cell (e.g.
# MuSiQue Hybrid, marked with double-dagger in Table 2).
_D = REPO / "data"
SOURCES = {
    "2WikiMHQA": {
        "Hybrid":    (_D / "eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl",
                      _D / "eval_1000q/hybrid/hybrid_eval_1000_run2_judged.jsonl"),
        "LogicRAG":  (_D / "eval_1000q/logicrag/logicrag_1000q_traced_judged.jsonl",
                      _D / "eval_1000q/logicrag/logicrag_1000q_traced_run2_judged.jsonl"),
        "LinearRAG": (_D / "eval_1000q/linearrag/linearrag_1000q_judged.jsonl",
                      _D / "eval_1000q/linearrag/linearrag_1000q_run2_judged.jsonl"),
    },
    "HotpotQA": {
        "Hybrid":    (_D / "cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl",
                      _D / "cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_run2_judged.jsonl"),
        "LogicRAG":  (_D / "cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_judged.jsonl",
                      _D / "cross_benchmark/hotpotqa/logicrag_hotpotqa_1000q_run2_judged.jsonl"),
        "LinearRAG": (_D / "cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_judged.jsonl",
                      _D / "cross_benchmark/hotpotqa/linearrag_hotpotqa_1000q_run2_judged.jsonl"),
    },
    "MuSiQue": {
        # Hybrid: run2 pending, paper marks with double-dagger.
        "Hybrid":    (_D / "cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl", None),
        "LogicRAG":  (_D / "cross_benchmark/musique/logicrag_musique_1000q_judged.jsonl",
                      _D / "cross_benchmark/musique/logicrag_musique_1000q_run2_judged.jsonl"),
        "LinearRAG": (_D / "cross_benchmark/musique/linearrag_musique_1000q_judged.jsonl",
                      _D / "cross_benchmark/musique/linearrag_musique_1000q_run2_judged.jsonl"),
    },
}

TYPE_ORDER = {
    "2WikiMHQA": ["bridge_comparison", "comparison", "compositional", "inference"],
    "HotpotQA":  ["bridge", "comparison"],
    "MuSiQue":   ["2hop", "3hop", "4hop"],  # merged
}

TYPE_ORDER_6TYPE = {
    "2WikiMHQA": ["bridge_comparison", "comparison", "compositional", "inference"],
    "HotpotQA":  ["bridge", "comparison"],
    "MuSiQue":   ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"],
}

TYPE_LABEL = {
    "bridge_comparison": "bridge_comp",
    "comparison": "comparison",
    "compositional": "compositional",
    "inference": "inference",
    "bridge": "bridge",
    "2hop": "2-hop",
    "3hop": "3-hop",
    "4hop": "4-hop",
    "3hop1": "3-hop-1",
    "3hop2": "3-hop-2",
    "4hop1": "4-hop-1",
    "4hop2": "4-hop-2",
    "4hop3": "4-hop-3",
}

# Paper Table 3 Hybrid 2Wiki values (mean across 2 runs) — audit target.
PAPER_TABLE3_HYBRID_2WIKI = {
    "bridge_comparison": 93.4,
    "comparison": 95.5,
    "compositional": 83.4,
    "inference": 94.4,
}
TOLERANCE_PP = 0.15
MIN_CELL_N = 20


# ---------------------------------------------------------------------------
# Data loading

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def musique_merge_type(t: str) -> str:
    """Fold 3hop1/3hop2 -> 3hop, 4hop1/4hop2/4hop3 -> 4hop; leave 2hop alone."""
    m = re.match(r"^(\d)hop", t)
    if not m:
        raise ValueError(f"unexpected MuSiQue type: {t!r}")
    return f"{m.group(1)}hop"


def _load_one(
    path: Path, bench: str, system: str,
    type_map: dict[str, str] | None = None,
    merge_musique: bool = True,
) -> list[dict]:
    """Load jsonl and normalize question_type.

    ``type_map`` lets run2 inherit canonical (pre-merge) question_type from
    run1 via question_id — some run2 files (e.g. linearrag MuSiQue) carry
    empty type strings. ``merge_musique=False`` preserves 6-way MuSiQue hop
    labels for the appendix figure.
    """
    if not path.exists():
        raise FileNotFoundError(f"{bench}/{system}: {path}")
    records = load_jsonl(path)
    if len(records) < 950:
        print(f"WARN {bench}/{system} {path.name}: only {len(records)} rows", file=sys.stderr)
    for r in records:
        for k in ("question_id", "judge_correct"):
            if k not in r:
                raise KeyError(f"{bench}/{system}: missing {k} in {path}")
        raw_type = r.get("question_type") or ""
        if not raw_type and type_map:
            raw_type = type_map.get(r["question_id"], "")
        if not raw_type:
            raise KeyError(f"{bench}/{system}: missing question_type for {r['question_id']} in {path}")
        if bench == "MuSiQue" and merge_musique:
            r["question_type"] = musique_merge_type(raw_type)
        else:
            r["question_type"] = raw_type
    return records


def load_per_system_records(sources: dict, merge_musique: bool = True) -> dict:
    """Return {benchmark: {system: {'run1': [...], 'run2': [...]|None}}}."""
    out = {}
    for bench, systems in sources.items():
        out[bench] = {}
        for system, (p1, p2) in systems.items():
            run1 = _load_one(p1, bench, system, merge_musique=merge_musique)
            run2 = None
            if p2 is not None:
                raw_map = {}
                for line in open(p1):
                    r = json.loads(line)
                    raw_map[r["question_id"]] = r.get("question_type") or ""
                run2 = _load_one(p2, bench, system, type_map=raw_map, merge_musique=merge_musique)
            out[bench][system] = {"run1": run1, "run2": run2}
    return out


# ---------------------------------------------------------------------------
# Aggregation + paired stats

def per_type_accuracy(records: list[dict]) -> dict[str, tuple[int, int]]:
    agg = defaultdict(lambda: [0, 0])
    for r in records:
        t = r["question_type"]
        agg[t][1] += 1
        if r["judge_correct"]:
            agg[t][0] += 1
    return {t: (v[0], v[1]) for t, v in agg.items()}


def paired_outcomes(records_a: list[dict], records_b: list[dict]) -> dict[str, tuple[list[bool], list[bool]]]:
    by_id_a = {r["question_id"]: r for r in records_a}
    by_id_b = {r["question_id"]: r for r in records_b}
    common_ids = set(by_id_a) & set(by_id_b)
    out = defaultdict(lambda: ([], []))
    for qid in common_ids:
        ra, rb = by_id_a[qid], by_id_b[qid]
        if ra["question_type"] != rb["question_type"]:
            continue
        out[ra["question_type"]][0].append(bool(ra["judge_correct"]))
        out[ra["question_type"]][1].append(bool(rb["judge_correct"]))
    return {t: (a, b) for t, (a, b) in out.items()}


def best_baseline_per_type(per_sys_accs: dict[str, dict[str, tuple[int, int]]]) -> dict[str, str]:
    logicrag = per_sys_accs["LogicRAG"]
    linearrag = per_sys_accs["LinearRAG"]
    best = {}
    for t in set(logicrag) | set(linearrag):
        k_l, n_l = logicrag.get(t, (0, 1))
        k_r, n_r = linearrag.get(t, (0, 1))
        best[t] = "LinearRAG" if (k_r / max(n_r, 1)) >= (k_l / max(n_l, 1)) else "LogicRAG"
    return best


def _cell_stats(
    runs: dict,
    type_order: list[str],
) -> dict[str, dict]:
    """Return per-type cell info with mean accuracy +/- half-range across runs.

    If run2 is absent, mean == run1 accuracy and half_range == 0.
    Wilson CI computed from run1 (snapshot) for statistical whiskers.
    """
    run1_accs = per_type_accuracy(runs["run1"])
    run2_accs = per_type_accuracy(runs["run2"]) if runs["run2"] else None
    out = {}
    for t in type_order:
        k1, n1 = run1_accs.get(t, (0, 0))
        p1 = k1 / n1 if n1 else 0.0
        if run2_accs is not None and t in run2_accs:
            k2, n2 = run2_accs[t]
            p2 = k2 / n2 if n2 else 0.0
            mean = (p1 + p2) / 2
            half_range = abs(p1 - p2) / 2
        else:
            mean, half_range = p1, 0.0
            n2 = None
        lo, hi = wilson_ci(k1, n1)  # Wilson from run1 N.
        out[t] = {
            "n": n1, "n_run2": n2,
            "acc_run1": p1,
            "acc_mean": mean,
            "half_range": half_range,
            "ci_lo": lo, "ci_hi": hi,
        }
    return out


def compute_benchmark_stats(bench: str, systems_runs: dict, type_order: list[str]) -> dict:
    # Cell stats (mean acc + half-range + Wilson CI) per system.
    per_sys_cells = {sys_: _cell_stats(runs, type_order) for sys_, runs in systems_runs.items()}

    # Paired McNemar on run1 (each run's question order is the gold standard).
    hybrid_r1 = systems_runs["Hybrid"]["run1"]
    pairs_vs_logic = paired_outcomes(hybrid_r1, systems_runs["LogicRAG"]["run1"])
    pairs_vs_linear = paired_outcomes(hybrid_r1, systems_runs["LinearRAG"]["run1"])

    # Best baseline by run1 accuracy (for McNemar pairing).
    best = {}
    for t in type_order:
        l_acc = per_sys_cells["LogicRAG"][t]["acc_run1"]
        r_acc = per_sys_cells["LinearRAG"][t]["acc_run1"]
        best[t] = "LinearRAG" if r_acc >= l_acc else "LogicRAG"

    # Bonferroni family: types within this benchmark panel.
    alpha_corrected = 0.05 / len(type_order)

    rows = []
    per_type_results = {}
    for t in type_order:
        cell = {}
        for sys_, cells in per_sys_cells.items():
            c = cells[t]
            if c["n"] < MIN_CELL_N:
                print(f"WARN {bench}/{sys_}/{t}: n={c['n']} < {MIN_CELL_N}", file=sys.stderr)
            cell[sys_] = {
                "n": c["n"], "n_run2": c["n_run2"],
                "acc": c["acc_mean"],      # plot uses mean across runs
                "acc_run1": c["acc_run1"],
                "half_range": c["half_range"],
                "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"],
            }
            rows.append({
                "benchmark": bench, "system": sys_, "question_type": t,
                "n": c["n"],
                "accuracy_pct_mean": round(c["acc_mean"] * 100, 2),
                "half_range_pp": round(c["half_range"] * 100, 2),
                "accuracy_pct_run1": round(c["acc_run1"] * 100, 2),
                "ci_lo_pct_run1": round(c["ci_lo"] * 100, 2),
                "ci_hi_pct_run1": round(c["ci_hi"] * 100, 2),
            })

        baseline_sys = best[t]
        pairs = pairs_vs_linear[t] if baseline_sys == "LinearRAG" else pairs_vs_logic[t]
        if pairs[0]:
            mc = mcnemar_exact(pairs[0], pairs[1])
            sig = mc.p_value < alpha_corrected
            # Per user decision: only display stars if Bonferroni-corrected significant.
            stars = ""
            if sig:
                if mc.p_value < 0.001:
                    stars = "***"
                elif mc.p_value < 0.01:
                    stars = "**"
                else:
                    stars = "*"
            cell["mcnemar"] = {
                "vs": baseline_sys, "b": mc.b, "c": mc.c,
                "p": mc.p_value, "significant": sig, "stars": stars,
            }
        per_type_results[t] = cell

    return {
        "per_type": per_type_results,
        "alpha_corrected": alpha_corrected,
        "csv_rows": rows,
    }


def assert_paper_match(bench_stats: dict) -> None:
    """Check that plotted mean (run1+run2 avg) matches paper Table 3 values."""
    errors = []
    for t, expected in PAPER_TABLE3_HYBRID_2WIKI.items():
        actual = bench_stats["per_type"][t]["Hybrid"]["acc"] * 100
        if abs(actual - expected) > TOLERANCE_PP:
            errors.append(f"  {t}: got {actual:.2f}, paper {expected} (delta {abs(actual-expected):.2f}pp)")
    if errors:
        raise AssertionError(
            f"2Wiki Hybrid (run1+run2 mean) vs paper Table 3 diverges beyond {TOLERANCE_PP}pp:\n"
            + "\n".join(errors)
        )
    print(f"PASS: 2Wiki Hybrid run-mean matches paper Table 3 within {TOLERANCE_PP}pp")


# ---------------------------------------------------------------------------
# Rendering (horizontal grouped bars, all 3 benchmarks stacked vertically)

# Faint background shading alternated per benchmark group to aid grouping.
BENCH_SHADE = {
    "2WikiMHQA": "#F4F4F4",
    "HotpotQA":  "#FFFFFF",
    "MuSiQue":   "#F4F4F4",
}


def _yerr_halfrange(cell: dict) -> float:
    """Whisker length: half-range across runs (paper convention).

    Zero-valued when only run1 exists (e.g. MuSiQue Hybrid).
    """
    return cell["half_range"] * 100


def render_horizontal_grouped(
    all_stats: dict,
    type_order_by_bench: dict[str, list[str]],
    out: Path,
) -> None:
    """One figure, horizontal grouped bars, all 3 benchmarks stacked top-to-bottom.

    Each y-row corresponds to one (benchmark, question_type). Within a row
    three bars (Hybrid/LogicRAG/LinearRAG) are dodged vertically. Benchmark
    group boundaries are marked with faint background shading and a bold
    benchmark label at the left.
    """
    plt.rcParams.update(LNCS_RCPARAMS)
    systems = ["Hybrid", "LogicRAG", "LinearRAG"]
    # Vertical dodge so 3 bars fit inside one row (row height = 1.0 unit).
    offsets = [0.26, 0.0, -0.26]  # Hybrid at top within a row.
    bar_h = 0.24

    # Flatten (benchmark, type) into rows, preserving paper table order.
    rows: list[tuple[str, str, dict]] = []
    for bench in all_stats:
        for t in type_order_by_bench[bench]:
            rows.append((bench, t, all_stats[bench]["per_type"][t]))
    n_rows = len(rows)

    # Figure sizing — slightly wider than LNCS column to fit rotated bench labels.
    fig_h = 0.9 + n_rows * 0.44
    fig, ax = plt.subplots(figsize=(5.6, fig_h))
    y = np.arange(n_rows)

    # 1) benchmark background shading + group label on the left.
    bench_ranges: dict[str, list[int]] = {}
    for i, (b, _, _) in enumerate(rows):
        bench_ranges.setdefault(b, [i, i])[1] = i
    for b, (s, e) in bench_ranges.items():
        ax.axhspan(s - 0.5, e + 0.5, color=BENCH_SHADE[b], zorder=0)
        # Benchmark label: rotated 90deg, well outside the y-tick labels.
        ax.text(
            -0.28, (s + e) / 2, b,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            rotation=90, transform=ax.get_yaxis_transform(),
        )

    # 2) bars per system with error bars + value labels.
    for sys_, dy in zip(systems, offsets):
        vals = np.array([r[2][sys_]["acc"] * 100 for r in rows])
        errs = np.array([_yerr_halfrange(r[2][sys_]) for r in rows])
        ax.barh(
            y + dy, vals, height=bar_h,
            color=OKABE_ITO[sys_], edgecolor="white", linewidth=0.4,
            xerr=errs, capsize=1.4, ecolor="#333", label=sys_, zorder=3,
        )
        for yi, v in zip(y + dy, vals):
            ax.text(v + 0.6, yi, f"{v:.1f}", va="center", ha="left", fontsize=5.8, zorder=4)

    # 3) y-tick labels: type name + sample size.
    labels = [f"{TYPE_LABEL.get(r[1], r[1])}\nN={r[2]['Hybrid']['n']}" for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()  # first row at top.

    # 4) significance stars on the right margin.
    for i, r in enumerate(rows):
        mc = r[2].get("mcnemar")
        if mc and mc.get("stars"):
            ax.text(
                1.015, i, mc["stars"],
                ha="left", va="center", fontsize=9, fontweight="bold",
                transform=ax.get_yaxis_transform(),
            )

    # 5) x-axis + grid. Auto-floor: include 0 when any value < 20%
    # (MuSiQue 4-hop-2 can be <10% in the 6-type appendix view).
    all_vals = [r[2][s]["acc"] * 100 for r in rows for s in systems]
    x_lo = 0 if min(all_vals) < 20 else 20
    ax.set_xlabel("Judge Accuracy (gemma-3-12b), %")
    ax.set_xlim(x_lo, 102)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10 if x_lo == 20 else 20))
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    # 6) horizontal divider lines between benchmark groups.
    prev = rows[0][0]
    for i, (b, _, _) in enumerate(rows[1:], 1):
        if b != prev:
            ax.axhline(i - 0.5, color="#888", linewidth=0.6, zorder=2)
            prev = b

    # 7) shared legend below (figure-level so it doesn't crowd the x-axis label).
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=3, frameon=False, fontsize=7.5,
    )
    # Dynamic margins — enough room on the left for rotated bench labels and
    # on the bottom for x-label + legend.
    bottom_frac = min(0.14, (0.6 + 1.2) / fig_h)
    plt.subplots_adjust(left=0.27, right=0.95, top=0.99, bottom=bottom_frac)

    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def write_csv(all_rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark", "system", "question_type", "n",
        "accuracy_pct_mean", "half_range_pp",
        "accuracy_pct_run1", "ci_lo_pct_run1", "ci_hi_pct_run1",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out}")


def _build_all_stats(type_order: dict[str, list[str]], merge_musique: bool) -> tuple[dict, list[dict]]:
    records = load_per_system_records(SOURCES, merge_musique=merge_musique)
    all_stats: dict = {}
    all_rows: list[dict] = []
    for bench, systems_records in records.items():
        stats = compute_benchmark_stats(bench, systems_records, type_order[bench])
        all_stats[bench] = stats
        all_rows.extend(stats["csv_rows"])
        if bench == "2WikiMHQA":
            assert_paper_match(stats)
    return all_stats, all_rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Main figure: MuSiQue merged to 2/3/4-hop (paper Table 3 convention).
    all_stats, all_rows = _build_all_stats(TYPE_ORDER, merge_musique=True)
    render_horizontal_grouped(all_stats, TYPE_ORDER, FIG_DIR / "per-type-dot.pdf")
    write_csv(all_rows, DATA_OUT_DIR / "per-type-values.csv")

    # Appendix figure: MuSiQue unmerged (6 hop types), same 3-benchmark layout.
    all_stats_6, all_rows_6 = _build_all_stats(TYPE_ORDER_6TYPE, merge_musique=False)
    render_horizontal_grouped(
        all_stats_6, TYPE_ORDER_6TYPE,
        FIG_DIR / "per-type-musique-6types.pdf",
    )
    write_csv(all_rows_6, DATA_OUT_DIR / "per-type-values-6types.csv")

    # Report significance summary (merged view).
    print("\n=== McNemar (per-benchmark Bonferroni) ===")
    for bench, stats in all_stats.items():
        print(f"\n{bench}  alpha={stats['alpha_corrected']:.4f}")
        for t in TYPE_ORDER[bench]:
            mc = stats["per_type"][t].get("mcnemar")
            if mc:
                tag = "SIG" if mc["significant"] else "ns"
                star = mc["stars"] or "-"
                print(f"  {t:<20} vs {mc['vs']:<9}  b={mc['b']:3d} c={mc['c']:3d}  p={mc['p']:.4f}  {tag} {star}")


if __name__ == "__main__":
    main()
