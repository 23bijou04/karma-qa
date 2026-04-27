"""Cross-benchmark routing distribution (paper Fig. 4, supports Table 5).

Single-panel, two 100%-stacked bars per benchmark:
  - Bar 1 — Questions (n=1000): 4-segment question-level partition,
    reuses Fig 3 taxonomy (classify_4mode in plot-routing-2wiki.py):
      Direct SPARQL / Decomp: all SPARQL / Decomp: SPARQL+text / Text only
  - Bar 2 — Sub-Qs (n = sub-questions generated): 2-segment partition:
      Sub-SPARQL / Sub-text

Both bars share the %-axis (0..100). Denominators printed on each bar so
the two stacks are not confused. "Any-SPARQL touch" is a derived union,
kept in paper prose rather than the chart.

Usage:
    python scripts/plot-cross-benchmark-routing.py
Output:
    ~/Documents/2026/2026_02/thirag-paper/figures/routing-cross-benchmark.pdf
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

# Question-level 4-mode partition (bottom -> top, symbolic -> textual).
# Keys mirror plot-routing-2wiki.py:classify_4mode() for taxonomy parity.
QUESTION_SEGMENTS = [
    ("direct_sparql",      "Direct SPARQL",          "#08519c"),  # deep blue
    ("decomp_all_sparql",  "Decomp: all SPARQL",     "#0072B2"),  # mid blue (Okabe-Ito)
    ("decomp_sparql_text", "Decomp: SPARQL+text",    "#009E73"),  # green (mixed)
    ("text_only",          "Text only",              "#E69F00"),  # orange
]

# Sub-question 2-mode partition (bottom -> top).
SUBQ_SEGMENTS = [
    ("sub_sparql", "SPARQL", "#0072B2"),
    ("sub_text",   "Text",   "#E69F00"),
]


def classify_question(r: dict) -> str:
    """Question-level 4-mode taxonomy.

    Uses top-level `method` (canonical routing label) to stay consistent with
    compute-routing-breakdown.py and the paper Table~\\ref{tab:routing}. A
    previous revision keyed on `sparql_query` presence + empty `sub_question_log`
    and diverged by ~1 question on HotpotQA / 12 questions on 2WikiMHQA.
    """
    top = r.get("method")
    if top == "sparql":
        return "direct_sparql"
    if top == "text":
        return "text_only"
    if top == "logicrag_decomposed":
        subs = r.get("sub_question_log") or []
        sub_methods = {s.get("method") for s in subs}
        if not sub_methods:
            return "text_only"
        has_sparql = "sparql" in sub_methods
        has_text = "text" in sub_methods
        if has_sparql and not has_text:
            return "decomp_all_sparql"
        if has_sparql and has_text:
            return "decomp_sparql_text"
        return "text_only"
    # Legacy / unexpected label: fall back to structural signals.
    sqlog = r.get("sub_question_log") or []
    if not sqlog:
        return "direct_sparql" if r.get("sparql_query") else "text_only"
    methods = [s.get("method", "?") for s in sqlog]
    n_sparql = sum(1 for m in methods if m == "sparql")
    if n_sparql == len(methods):
        return "decomp_all_sparql"
    if n_sparql > 0:
        return "decomp_sparql_text"
    return "text_only"


def compute_stats() -> dict:
    """Return {bench: {'q': {seg: pct}, 'sub': {seg: pct}, 'n_q': int, 'n_sub': int}}."""
    out: dict = {}
    for bench, path in HYBRID_SOURCES.items():
        q_counts: dict = defaultdict(int)
        sub_counts = {"sub_sparql": 0, "sub_text": 0}
        n_q = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                q_counts[classify_question(r)] += 1
                n_q += 1
                # Count sub-question methods only for decomposed rows.
                for s in r.get("sub_question_log") or []:
                    m = s.get("method")
                    if m == "sparql":
                        sub_counts["sub_sparql"] += 1
                    else:
                        sub_counts["sub_text"] += 1
        n_sub = sub_counts["sub_sparql"] + sub_counts["sub_text"]
        out[bench] = {
            "n_q": n_q,
            "n_sub": n_sub,
            "q": {k: 100 * q_counts.get(k, 0) / n_q if n_q else 0.0
                  for k, _, _ in QUESTION_SEGMENTS},
            "sub": {k: 100 * sub_counts[k] / n_sub if n_sub else 0.0
                    for k, _, _ in SUBQ_SEGMENTS},
        }
    return out


def _stack_bar(ax, x: float, width: float, segments: list, values: dict,
               min_label_pct: float = 5.0) -> None:
    """Draw a 100%-stacked bar at x with `segments`=(key,label,color)."""
    bottom = 0.0
    for key, _label, color in segments:
        v = values.get(key, 0.0)
        if v <= 0:
            continue
        ax.bar(x, v, width, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.6)
        # Numeric label centred on segment when large enough.
        if v >= min_label_pct:
            ax.text(x, bottom + v / 2, f"{v:.1f}",
                    ha="center", va="center", fontsize=6.5, color="white")
        bottom += v


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(LNCS_RCPARAMS)
    stats = compute_stats()

    # Reproducibility audit to stdout.
    print("Cross-benchmark routing partition (hybrid, 1000q each):")
    for bench, s in stats.items():
        print(f"  {bench}  n_q={s['n_q']}  n_sub={s['n_sub']}")
        for key, label, _ in QUESTION_SEGMENTS:
            print(f"    Q  {label:24s} {s['q'][key]:5.1f}%")
        for key, label, _ in SUBQ_SEGMENTS:
            print(f"    S  {label:24s} {s['sub'][key]:5.1f}%")

    benchmarks = list(HYBRID_SOURCES.keys())
    # Two bars per group, grouped per benchmark.
    group_w = 1.0
    bar_w = 0.38
    group_centers = np.arange(len(benchmarks)) * (group_w + 0.7)
    x_q = group_centers - bar_w / 2 - 0.02
    x_s = group_centers + bar_w / 2 + 0.02

    fig, ax = plt.subplots(figsize=(6.4, 2.6))

    for i, bench in enumerate(benchmarks):
        _stack_bar(ax, x_q[i], bar_w, QUESTION_SEGMENTS, stats[bench]["q"])
        _stack_bar(ax, x_s[i], bar_w, SUBQ_SEGMENTS, stats[bench]["sub"])

    # Thin "Q" / "Sub-Q" tags directly above each bar (no denominators here; those
    # move into the benchmark tick labels below).
    for i in range(len(benchmarks)):
        ax.text(x_q[i], 101, "Q", ha="center", va="bottom", fontsize=6)
        ax.text(x_s[i], 101, "Sub", ha="center", va="bottom", fontsize=6)

    # Benchmark tick labels carry the denominators, so the x-axis title is redundant.
    tick_labels = [
        f"{b}\n(Q$=${stats[b]['n_q']}, Sub$=${stats[b]['n_sub']})"
        for b in benchmarks
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(plt.FixedLocator([0, 20, 40, 60, 80, 100]))
    ax.set_ylabel("%")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Two separate legends: Q-bar uses 4-mode question taxonomy; Sub-bar uses
    # 2-mode sub-question taxonomy. Sub-SPARQL aggregates SPARQL-touching subs
    # across Direct, Decomp:all, and the SPARQL share of Decomp:SPARQL+text;
    # Sub-text is the complement. Colours intentionally reuse the SPARQL-blue
    # and text-orange palette to keep the two stacks visually coherent.
    q_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in QUESTION_SEGMENTS]
    q_labels = [lab for _, lab, _ in QUESTION_SEGMENTS]
    leg_q = ax.legend(q_handles, q_labels,
                      title="Whole question (Q)",
                      loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      fontsize=6.5, title_fontsize=6.5,
                      frameon=False,
                      handlelength=1.2, handletextpad=0.5,
                      labelspacing=0.5, borderaxespad=0.0,
                      alignment="left")
    leg_q._legend_box.align = "left"
    ax.add_artist(leg_q)

    s_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SUBQ_SEGMENTS]
    s_labels = [lab for _, lab, _ in SUBQ_SEGMENTS]
    leg_s = ax.legend(s_handles, s_labels,
                      title="Sub-question (Sub)",
                      loc="lower left", bbox_to_anchor=(1.01, 0.0),
                      fontsize=6.5, title_fontsize=6.5,
                      frameon=False,
                      handlelength=1.2, handletextpad=0.5,
                      labelspacing=0.5, borderaxespad=0.0,
                      alignment="left")
    leg_s._legend_box.align = "left"

    plt.subplots_adjust(left=0.08, right=0.72, bottom=0.22, top=0.92)

    out = FIG_DIR / "routing-cross-benchmark.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
