"""
Shared rendering helpers for companion analytics report builder (Phase 05).

Extracted to keep build_report.py under 200 LOC.
All functions are pure (no side effects, no file writes).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]

# Float column names that should be rounded to 3 decimals for display
DISPLAY_FLOAT_COLS: frozenset[str] = frozenset({
    "acc", "wilson_lo", "wilson_hi", "r1_acc", "r1_lo", "r1_hi",
    "r2_acc", "r2_lo", "r2_hi", "flip_pct", "pct_of_failures", "pct_of_total",
    "acc_a", "acc_a_lo", "acc_a_hi", "acc_b", "acc_b_lo", "acc_b_hi",
    "em_mean", "f1_mean", "containment_mean", "hedge_rate", "extraction_rate",
    "mean_s", "median_s", "p95_s", "p0", "p50", "p95",
    "chi2", "p_value", "p_bonferroni",
})

# Dim notes appended after section header
DIM_NOTES: dict[str, str] = {
    "flip_rate": (
        "_Note: flip-rate p-values raw — single paired test per cell, not a family; "
        "Bonferroni applies only to Dim 9._"
    ),
    "significance": (
        "_Correction: Bonferroni family size = 9 per run (3 pairs × 3 benchmarks). "
        "R1/R2 treated as independent families — never pooled to 18._"
    ),
    "failure_modes": "_Rules version: v1 (see failure_modes.py docstring)_",
}

# (dim_id, title) per dim key
DIM_META: dict[str, tuple[int, str]] = {
    "route_x_type":    (1, "Route × Question-Type (Hybrid)"),
    "decomp_stats":    (2, "Decomposition Stats (Hybrid + LogicRAG)"),
    "failure_modes":   (3, "Failure Modes"),
    "flip_rate":       (4, "R1↔R2 Flip Rate + McNemar"),
    "nonjudge_metrics": (5, "Non-Judge Metrics per Type/Method"),
    "fallback_chains": (6, "Hybrid Fallback Chains"),
    "latency":         (7, "Latency (Hybrid only)"),
    "subq_metrics":    (8, "Sub-Question Metrics"),
    "significance":    (9, "Between-System Significance"),
}

# CSV file templates per dim key ({bench} substituted at render time)
CSV_MAP: dict[str, list[str]] = {
    "route_x_type":    ["dim1_{bench}_r1.csv", "dim1_{bench}_r2.csv"],
    "decomp_stats":    ["dim2_{bench}_r1.csv", "dim2_{bench}_r2.csv"],
    "failure_modes":   ["dim3_failure_modes.csv"],
    "flip_rate":       ["dim4_{bench}.csv"],
    "nonjudge_metrics": ["dim5_nonjudge_metrics.csv"],
    "fallback_chains": ["dim6_{bench}_r1.csv", "dim6_{bench}_r2.csv"],
    "latency":         ["dim7_latency.csv"],
    "subq_metrics":    ["dim8_{bench}_r1.csv", "dim8_{bench}_r2.csv"],
    "significance":    ["dim9_{bench}_r1.csv", "dim9_{bench}_r2.csv"],
}


def sha256_file(path: Path, n: int = 16) -> str:
    """Return first n hex chars of SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV; dim3 has comment lines starting with '#'."""
    return pd.read_csv(path, comment="#")


def to_markdown(df: pd.DataFrame, max_rows: int = 30, csv_path: Path | None = None) -> str:
    """Render DataFrame as markdown table, capped at max_rows."""
    df = df.copy()
    for col in df.columns:
        if col in DISPLAY_FLOAT_COLS or df[col].dtype == float:
            df[col] = df[col].apply(lambda x: round(x, 3) if pd.notna(x) else x)
    truncated = len(df) > max_rows
    display_df = df.iloc[:max_rows]
    try:
        table = display_df.to_markdown(index=False)
    except ImportError:
        # Manual pipe-table fallback (tabulate absent)
        cols = list(display_df.columns)
        rows_md = [" | ".join(str(c) for c in cols), " | ".join(["---"] * len(cols))]
        for _, row in display_df.iterrows():
            rows_md.append(" | ".join(str(v) for v in row))
        table = "\n".join(rows_md)
    if truncated:
        link = f"[full CSV]({csv_path})" if csv_path else "full CSV"
        table += f"\n\n_(showing {max_rows}/{len(df)} rows — see {link})_"
    return table


def source_footer(csv_paths: list[Path], n_rows: int) -> str:
    """Generate italicised source footer line for a markdown section."""
    parts = []
    for p in csv_paths:
        sha = sha256_file(p, 8) if p.exists() else "MISSING"
        parts.append(f"`{p.relative_to(_REPO)}` ({sha})")
    return f"_Source: {', '.join(parts)} — {n_rows} rows displayed_"


# ---------------------------------------------------------------------------
# Cross-report data extraction helpers (pure, no file writes)
# ---------------------------------------------------------------------------

KEY_FINDINGS = (
    "- **2Wiki**: Hybrid (89.4%) > LinearRAG (84.0%) > LogicRAG (74.5%) — all pairs Bonferroni-significant.\n"
    "- **HotPotQA**: Hybrid ≈ LinearRAG (86.1% vs 86.9%, not significant). Both >> LogicRAG.\n"
    "- **MuSiQue**: Hybrid (65.6%) significantly above LinearRAG/LogicRAG (~51%); LinearRAG vs LogicRAG indistinguishable.\n"
    "- **HotPotQA R2 anomaly**: rule+multihop route bucket at 39.5% — flagged in Phase 02.\n"
    "- **MuSiQue under-decomp**: 172 questions with sq < expected hops — flagged in Phase 03.\n"
    "- **MuSiQue 3hop2/4hop3 flip rates** elevated (>19%) — judge borderline on hard questions."
)


def collect_em_rows(data_dir: Path, benchmarks: list[str]) -> pd.DataFrame:
    """Extract per-system Judge EM + Wilson CI from dim9 CSVs across all benchmarks/runs."""
    rows: list[dict] = []
    for bench in benchmarks:
        for run in ["r1", "r2"]:
            path = data_dir / f"dim9_{bench}_{run}.csv"
            if not path.exists():
                continue
            for _, row in load_csv(path).iterrows():
                for s, acc, lo, hi in [
                    (row["sys_a"], row["acc_a"], row["acc_a_lo"], row["acc_a_hi"]),
                    (row["sys_b"], row["acc_b"], row["acc_b_lo"], row["acc_b_hi"]),
                ]:
                    rows.append({"bench": bench, "system": s, "run": run,
                                 "judge_em": round(acc, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .drop_duplicates(["bench", "system", "run"])
              .sort_values(["bench", "system", "run"])
              .reset_index(drop=True))


def collect_flip_rows(data_dir: Path, benchmarks: list[str]) -> pd.DataFrame:
    """Extract max flip_pct per (bench, system) from dim4 CSVs."""
    rows: list[dict] = []
    for bench in benchmarks:
        path = data_dir / f"dim4_{bench}.csv"
        if not path.exists():
            continue
        df = load_csv(path)
        for sys_ in df["system"].unique():
            rows.append({"bench": bench, "system": sys_,
                         "max_flip_pct": round(df[df["system"] == sys_]["flip_pct"].max(), 3)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["bench", "system"]).reset_index(drop=True)


def collect_sig_rows(data_dir: Path, benchmarks: list[str]) -> pd.DataFrame:
    """Extract between-system McNemar rows from dim9 CSVs."""
    rows: list[dict] = []
    for bench in benchmarks:
        for run in ["r1", "r2"]:
            path = data_dir / f"dim9_{bench}_{run}.csv"
            if not path.exists():
                continue
            for _, row in load_csv(path).iterrows():
                rows.append({"bench": bench, "run": run,
                             "pair": f"{row['sys_a']} vs {row['sys_b']}",
                             "acc_a": round(row["acc_a"], 3), "acc_b": round(row["acc_b"], 3),
                             "p_bonferroni": round(row["p_bonferroni"], 4),
                             "significant_05": row["significant_05"]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["bench", "run", "pair"]).reset_index(drop=True)
