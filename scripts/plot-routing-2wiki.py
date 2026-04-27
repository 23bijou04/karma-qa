"""Cross-benchmark routing breakdown (replaces old 2Wiki-only figure).

For each of 3 benchmarks, classify hybrid runs into 4 routing modes:
  - Direct SPARQL: single-hop SPARQL, no decomposition fired
  - Decomp: all SPARQL: decomposition fired, every sub-Q solved by SPARQL
  - Decomp: SPARQL+text: decomposition fired, mix of SPARQL and text sub-Qs
  - Text only: decomposition fired with all-text sub-Qs OR no decomp with no SPARQL

Emits 2-panel figure: Judge Accuracy (left), Containment (right). Grouped
bars per mode, coloured by benchmark (same palette as routing-cross-benchmark).

Output: routing-2wiki.pdf (filename kept for paper backward compat).

Usage:
    python scripts/plot-routing-2wiki.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_UTILS_PATH = Path(__file__).parent / "_plot_utils.py"
_spec = importlib.util.spec_from_file_location("_plot_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
sys.modules["_plot_utils"] = _utils
_spec.loader.exec_module(_utils)  # type: ignore[union-attr]

LNCS_RCPARAMS = _utils.LNCS_RCPARAMS

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = Path("~/Documents/2026/2026_02/thirag-paper/figures").expanduser()

HYBRID_SOURCES = {
    "2WikiMHQA": REPO / "data/eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl",
    "HotpotQA":  REPO / "data/cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl",
    "MuSiQue":   REPO / "data/cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl",
}

# Mode display order — increasing text-reliance.
MODES = [
    ("direct_sparql",      "Direct\nSPARQL"),
    ("decomp_all_sparql",  "Decomp:\nall SPARQL"),
    ("decomp_sparql_text", "Decomp:\nSPARQL+text"),
    ("text_only",          "Text\nonly"),
]

# Palette matches routing-cross-benchmark figure for visual continuity.
BENCH_COLORS = {
    "2WikiMHQA": "#0072B2",
    "HotpotQA":  "#E69F00",
    "MuSiQue":   "#009E73",
}


def classify_4mode(r: dict) -> str:
    """Collapse hybrid route into 4-mode taxonomy used by paper Table 4/5."""
    sqlog = r.get("sub_question_log") or []
    if not sqlog:
        # No decomposition fired.
        return "direct_sparql" if r.get("sparql_query") else "text_only"
    methods = [s.get("method", "?") for s in sqlog]
    n_sparql = sum(1 for m in methods if m == "sparql")
    if n_sparql == len(methods):
        return "decomp_all_sparql"
    if n_sparql > 0:
        return "decomp_sparql_text"
    return "text_only"


def compute_per_bench() -> dict:
    """Return {bench: {mode: {'n', 'pct', 'judge_acc', 'containment'}}}."""
    out = {}
    for bench, path in HYBRID_SOURCES.items():
        agg = defaultdict(lambda: {"n": 0, "correct": 0, "cont": 0})
        total = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                m = classify_4mode(r)
                agg[m]["n"] += 1
                if r.get("judge_correct"):
                    agg[m]["correct"] += 1
                if r.get("containment"):
                    agg[m]["cont"] += 1
                total += 1
        out[bench] = {}
        for mode_key, _ in MODES:
            a = agg.get(mode_key, {"n": 0, "correct": 0, "cont": 0})
            n = a["n"]
            out[bench][mode_key] = {
                "n": n,
                "pct": 100 * n / total if total else 0.0,
                "judge_acc": 100 * a["correct"] / n if n else 0.0,
                "containment": 100 * a["cont"] / n if n else 0.0,
            }
    return out


def _panel(ax, stats: dict, metric_key: str, metric_title: str) -> None:
    benchmarks = list(HYBRID_SOURCES.keys())
    x = np.arange(len(MODES))
    width = 0.26
    offsets = [-width, 0.0, width]
    for i, bench in enumerate(benchmarks):
        vals = [stats[bench][m[0]][metric_key] for m in MODES]
        bars = ax.bar(
            x + offsets[i], vals, width,
            color=BENCH_COLORS[bench], edgecolor="white", linewidth=0.4,
            label=bench,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 1.0,
                f"{v:.1f}", ha="center", va="bottom", fontsize=5.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in MODES], fontsize=7)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.set_ylabel(metric_title + ", %")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title(metric_title, fontsize=9)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(LNCS_RCPARAMS)
    stats = compute_per_bench()

    # Echo table for reproducibility audit.
    print("Routing breakdown (hybrid, 1000q):")
    for bench in HYBRID_SOURCES:
        print(f"  {bench}")
        for mode_key, label in MODES:
            s = stats[bench][mode_key]
            print(f"    {mode_key:20s}  N={s['n']:4d} ({s['pct']:5.1f}%)  "
                  f"acc={s['judge_acc']:5.1f}  cont={s['containment']:5.1f}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.8, 3.2), sharey=True,
    )
    _panel(ax1, stats, "judge_acc",   "Judge Accuracy")
    _panel(ax2, stats, "containment", "Containment")

    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=3, frameon=False, fontsize=7,
    )
    plt.subplots_adjust(bottom=0.22, top=0.90, wspace=0.08, left=0.08, right=0.98)

    out = FIG_DIR / "routing-2wiki.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
