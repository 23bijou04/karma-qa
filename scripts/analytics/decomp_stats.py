"""
Dimension 2: Decomposition statistics.

Hybrid: sub-question count distribution by question_type (p0/p50/p95),
        per-count accuracy. MuSiQue under-decomposition flag.
LogicRAG: rounds count distribution by question_type (top-level int or trace.rounds list).

Output: plans/260422-1027-companion-analytics-18-judged-jsonls/reports/data/
        dim2_{bench}_{run}.csv

Columns:
  system, benchmark, run, question_type, sub_q_count (int, -1=N/A),
  n_rows, correct, acc, is_underdecomp_flag (bool, Hybrid/MuSiQue only)
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))

from analytics.load_judged import FILE_INVENTORY, load_judged  # noqa: E402

_OUT_DIR = (
    _REPO
    / "plans/260422-1027-companion-analytics-18-judged-jsonls/reports/data"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expected_hops(question_type: str | None) -> int | None:
    """Extract expected hop count from MuSiQue question_type (e.g. '3hop1' -> 3)."""
    m = re.match(r"^(\d+)hop", question_type or "")
    return int(m.group(1)) if m else None


def _rounds_count(row: dict) -> int:
    """Extract rounds count from a LogicRAG row (top-level int or via trace.rounds list)."""
    # Top-level 'rounds' is an int in cross-benchmark files
    top = row.get("rounds")
    if isinstance(top, int):
        return top
    # 2Wiki uses trace.rounds list
    trace = row.get("trace")
    if isinstance(trace, dict):
        r = trace.get("rounds", [])
        if isinstance(r, list):
            return len(r)
    return 0


# ---------------------------------------------------------------------------
# Hybrid decomp stats
# ---------------------------------------------------------------------------
def _hybrid_records(rows: dict, bench: str, run: str) -> list[dict]:
    records = []
    for row in rows.values():
        if row.get("method") != "logicrag_decomposed":
            continue
        sql = row.get("sub_question_log") or []
        sq_count = len(sql)
        qtype = row.get("question_type") or "unknown"
        correct = int(bool(row.get("judge_correct")))
        exp = _expected_hops(qtype)
        under = bool(exp is not None and sq_count < exp)
        records.append(
            {
                "system": "hybrid",
                "benchmark": bench,
                "run": run,
                "question_type": qtype,
                "sub_q_count": sq_count,
                "correct": correct,
                "is_underdecomp": under,
            }
        )
    return records


# ---------------------------------------------------------------------------
# LogicRAG decomp stats
# ---------------------------------------------------------------------------
def _logicrag_records(rows: dict, bench: str, run: str) -> list[dict]:
    records = []
    for row in rows.values():
        sq_count = _rounds_count(row)
        qtype = row.get("question_type") or "unknown"
        correct = int(bool(row.get("judge_correct")))
        records.append(
            {
                "system": "logicrag",
                "benchmark": bench,
                "run": run,
                "question_type": qtype,
                "sub_q_count": sq_count,
                "correct": correct,
                "is_underdecomp": False,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Aggregate to percentiles + per-count accuracy
# ---------------------------------------------------------------------------
def _aggregate(records: list[dict]) -> pd.DataFrame:
    """Aggregate to (system, benchmark, run, question_type, sub_q_count) rows."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)

    # Per (qtype, sub_q_count): count rows, sum correct, underdecomp flag
    grp = (
        df.groupby(["system", "benchmark", "run", "question_type", "sub_q_count"])
        .agg(
            n_rows=("correct", "count"),
            correct=("correct", "sum"),
            n_underdecomp=("is_underdecomp", "sum"),
        )
        .reset_index()
    )
    grp["acc"] = grp["correct"] / grp["n_rows"]
    grp["is_underdecomp_flag"] = grp["n_underdecomp"] > 0

    # Append percentile summary rows (sub_q_count=-1 signals summary)
    summaries = []
    for (sys_, bench, run, qtype), sub in df.groupby(
        ["system", "benchmark", "run", "question_type"]
    ):
        counts = sub["sub_q_count"].values
        import numpy as np

        summaries.append(
            {
                "system": sys_,
                "benchmark": bench,
                "run": run,
                "question_type": qtype,
                "sub_q_count": -1,  # sentinel: summary row
                "n_rows": len(counts),
                "correct": int(sub["correct"].sum()),
                "n_underdecomp": int(sub["is_underdecomp"].sum()),
                "acc": sub["correct"].sum() / len(counts) if len(counts) else float("nan"),
                "is_underdecomp_flag": sub["is_underdecomp"].any(),
                "p0": int(counts.min()),
                "p50": float(np.percentile(counts, 50)),
                "p95": float(np.percentile(counts, 95)),
            }
        )

    summary_df = pd.DataFrame(summaries)
    # Merge: grp doesn't have p0/p50/p95; fill with nan
    for col in ("p0", "p50", "p95"):
        grp[col] = float("nan")

    combined = pd.concat([grp, summary_df], ignore_index=True, sort=False)
    cols = [
        "system", "benchmark", "run", "question_type", "sub_q_count",
        "n_rows", "correct", "acc", "is_underdecomp_flag",
        "p0", "p50", "p95",
    ]
    for c in cols:
        if c not in combined.columns:
            combined[c] = float("nan")
    return combined[cols].sort_values(
        ["system", "question_type", "sub_q_count"]
    )


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_check(df: pd.DataFrame, bench: str, run: str) -> list[str]:
    issues = []
    if df.empty:
        issues.append(f"{bench}/{run}: empty output")
        return issues
    # MuSiQue Hybrid: some question types should be flagged as under-decomposed.
    # Memory signal was ~85 (earlier data); current data gives ~172 under-decomp rows.
    # Assert at least one qtype has is_underdecomp_flag=True.
    if bench == "musique" and run == "r1":
        hybrid_summary = df[
            (df["system"] == "hybrid") & (df["sub_q_count"] == -1)
        ]
        any_underdecomp = hybrid_summary["is_underdecomp_flag"].any()
        if not any_underdecomp:
            issues.append(
                "musique/r1 Hybrid: no under-decomp flag set (expected some qtypes flagged)"
            )
    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dim 2: decomposition statistics")
    p.add_argument(
        "--benchmark",
        choices=["2wiki", "hotpotqa", "musique", "all"],
        default="all",
    )
    p.add_argument("--run", choices=["r1", "r2", "all"], default="all")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    benches = ["2wiki", "hotpotqa", "musique"] if args.benchmark == "all" else [args.benchmark]
    runs = ["r1", "r2"] if args.run == "all" else [args.run]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_issues: list[str] = []
    for bench in benches:
        for run in runs:
            h_key = (bench, "hybrid", run)
            l_key = (bench, "logicrag", run)

            if h_key not in FILE_INVENTORY and l_key not in FILE_INVENTORY:
                print(f"SKIP: no inventory entry for {bench}/{run}")
                continue

            print(f"Computing dim2 {bench}/{run} ...", end=" ", flush=True)
            records: list[dict] = []

            if h_key in FILE_INVENTORY:
                try:
                    h_rows = load_judged(bench, "hybrid", run)
                    records.extend(_hybrid_records(h_rows, bench, run))
                except Exception as exc:
                    print(f"\n  WARN hybrid load failed: {exc}")
                    all_issues.append(f"hybrid {bench}/{run}: {exc}")

            if l_key in FILE_INVENTORY:
                try:
                    l_rows = load_judged(bench, "logicrag", run)
                    records.extend(_logicrag_records(l_rows, bench, run))
                except Exception as exc:
                    print(f"\n  WARN logicrag load failed: {exc}")
                    all_issues.append(f"logicrag {bench}/{run}: {exc}")

            if not records:
                print("no records, skipping")
                continue

            df = _aggregate(records)
            issues = sanity_check(df, bench, run)
            for issue in issues:
                print(f"\n  WARN: {issue}")
            all_issues.extend(issues)

            out = _OUT_DIR / f"dim2_{bench}_{run}.csv"
            df.to_csv(out, index=False, float_format="%.6f")
            print(f"→ {out.name} ({len(df)} rows)")

    if all_issues:
        print(f"\n{len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"  ! {issue}")
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
