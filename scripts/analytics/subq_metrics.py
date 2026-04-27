"""
Dimension 8: Sub-question level metrics for Hybrid rows.

Iterates sub_question_log entries across all Hybrid rows. Per sub-Q records:
  method (sparql/text/unknown), answer_nonempty (bool), parent judge_correct.

Computes:
  - Sub-Q method distribution (sparql vs text counts + rates)
  - Sub-Q success rate (answer non-empty)
  - Correlation: all-sub-Qs-succeed ↔ final judge_correct
  - Chain depth from depends_on field (transitive ancestors); max depth per benchmark

Output CSV: dim8_{bench}_{run}.csv
Columns: benchmark, run, metric, question_type, value, n
"""

import argparse
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
# Chain depth (transitive ancestors via depends_on)
# ---------------------------------------------------------------------------
def _compute_depth(idx: int, sql: list[dict], memo: dict[int, int]) -> int:
    """Transitive depth of sub-question at idx via depends_on indices."""
    if idx in memo:
        return memo[idx]
    deps = sql[idx].get("depends_on") or []
    if not deps or not isinstance(deps, list):
        memo[idx] = 0
        return 0
    # guard against out-of-range indices
    valid_deps = [d for d in deps if isinstance(d, int) and 0 <= d < len(sql) and d != idx]
    if not valid_deps:
        memo[idx] = 0
        return 0
    depth = 1 + max(_compute_depth(d, sql, memo) for d in valid_deps)
    memo[idx] = depth
    return depth


def _max_chain_depth(sql: list[dict]) -> int:
    """Max transitive depth across all sub-questions in a single row."""
    if not sql:
        return 0
    memo: dict[int, int] = {}
    return max(_compute_depth(i, sql, memo) for i in range(len(sql)))


# ---------------------------------------------------------------------------
# Core record extraction
# ---------------------------------------------------------------------------
def _extract_subq_records(rows: dict, bench: str, run: str) -> list[dict]:
    """Flatten sub_question_log entries into per-sub-Q records."""
    records = []
    for row in rows.values():
        if row.get("method") != "logicrag_decomposed":
            continue
        sql = row.get("sub_question_log") or []
        if not sql:
            continue
        qtype = row.get("question_type") or "unknown"
        parent_correct = int(bool(row.get("judge_correct")))
        all_succeed = all(bool((sq.get("answer") or "").strip()) for sq in sql)

        for sq in sql:
            m = sq.get("method") or "unknown"
            answer = sq.get("answer") or ""
            answer_nonempty = bool(answer.strip())
            records.append(
                {
                    "benchmark": bench,
                    "run": run,
                    "question_type": qtype,
                    "sub_method": m,
                    "answer_nonempty": int(answer_nonempty),
                    "parent_judge_correct": parent_correct,
                    "all_subq_succeed": int(all_succeed),
                }
            )
    return records


def _extract_row_level(rows: dict, bench: str, run: str) -> list[dict]:
    """Row-level summary: all-sub-Qs-succeed ↔ judge_correct + max chain depth."""
    records = []
    for row in rows.values():
        if row.get("method") != "logicrag_decomposed":
            continue
        sql = row.get("sub_question_log") or []
        if not sql:
            continue
        qtype = row.get("question_type") or "unknown"
        correct = int(bool(row.get("judge_correct")))
        all_succeed = int(all(bool((sq.get("answer") or "").strip()) for sq in sql))
        depth = _max_chain_depth(sql)
        records.append(
            {
                "benchmark": bench,
                "run": run,
                "question_type": qtype,
                "correct": correct,
                "all_subq_succeed": all_succeed,
                "max_depth": depth,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Aggregate into long-format metric table
# ---------------------------------------------------------------------------
def _build_output(
    subq_records: list[dict],
    row_records: list[dict],
    bench: str,
    run: str,
) -> pd.DataFrame:
    if not subq_records:
        return pd.DataFrame()

    sq_df = pd.DataFrame(subq_records)
    row_df = pd.DataFrame(row_records) if row_records else pd.DataFrame()

    metric_rows: list[dict] = []

    # 1. Sub-Q method distribution per question_type
    for (qtype, method), grp in sq_df.groupby(["question_type", "sub_method"]):
        metric_rows.append(
            {
                "benchmark": bench,
                "run": run,
                "metric": f"subq_method_{method}_count",
                "question_type": qtype,
                "value": len(grp),
                "n": len(sq_df[sq_df["question_type"] == qtype]),
            }
        )

    # 2. Sub-Q success rate per (question_type, method)
    for (qtype, method), grp in sq_df.groupby(["question_type", "sub_method"]):
        success = grp["answer_nonempty"].sum()
        total = len(grp)
        metric_rows.append(
            {
                "benchmark": bench,
                "run": run,
                "metric": f"subq_success_rate_{method}",
                "question_type": qtype,
                "value": success / total if total else float("nan"),
                "n": total,
            }
        )

    # 3. Overall sub-Q success rate per question_type
    for qtype, grp in sq_df.groupby("question_type"):
        total = len(grp)
        success = grp["answer_nonempty"].sum()
        metric_rows.append(
            {
                "benchmark": bench,
                "run": run,
                "metric": "subq_success_rate_all",
                "question_type": qtype,
                "value": success / total if total else float("nan"),
                "n": total,
            }
        )

    if not row_df.empty:
        # 4. Correlation: all-sub-Qs-succeed ↔ judge_correct per question_type
        for qtype, grp in row_df.groupby("question_type"):
            n = len(grp)
            # Point-biserial correlation (Pearson between bool vars)
            if n > 1 and grp["all_subq_succeed"].std() > 0 and grp["correct"].std() > 0:
                corr = grp["all_subq_succeed"].corr(grp["correct"])
            else:
                corr = float("nan")
            metric_rows.append(
                {
                    "benchmark": bench,
                    "run": run,
                    "metric": "all_subq_succeed_x_correct_corr",
                    "question_type": qtype,
                    "value": corr,
                    "n": n,
                }
            )

        # 5. Rate: all-sub-Qs-succeed → judge_correct (conditional)
        for (qtype, all_suc), grp in row_df.groupby(["question_type", "all_subq_succeed"]):
            label = "all_succeed" if all_suc else "some_fail"
            metric_rows.append(
                {
                    "benchmark": bench,
                    "run": run,
                    "metric": f"correct_given_{label}",
                    "question_type": qtype,
                    "value": grp["correct"].mean(),
                    "n": len(grp),
                }
            )

        # 6. Max chain depth per benchmark (summary row, question_type="ALL")
        max_depth_overall = int(row_df["max_depth"].max()) if not row_df.empty else 0
        metric_rows.append(
            {
                "benchmark": bench,
                "run": run,
                "metric": "max_chain_depth",
                "question_type": "ALL",
                "value": float(max_depth_overall),
                "n": len(row_df),
            }
        )

        # 7. Acc by chain depth
        for depth, grp in row_df.groupby("max_depth"):
            metric_rows.append(
                {
                    "benchmark": bench,
                    "run": run,
                    "metric": "acc_by_max_depth",
                    "question_type": f"depth_{depth}",
                    "value": grp["correct"].mean(),
                    "n": len(grp),
                }
            )

    df = pd.DataFrame(metric_rows)
    cols = ["benchmark", "run", "metric", "question_type", "value", "n"]
    return df[cols].sort_values(["metric", "question_type"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dim 8: sub-question metrics")
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

    for bench in benches:
        for run in runs:
            key = (bench, "hybrid", run)
            if key not in FILE_INVENTORY:
                print(f"SKIP: no inventory entry for {key}")
                continue
            print(f"Computing dim8 {bench}/{run} ...", end=" ", flush=True)
            try:
                h_rows = load_judged(bench, "hybrid", run)
            except Exception as exc:
                print(f"ERROR loading: {exc}")
                continue

            subq_records = _extract_subq_records(h_rows, bench, run)
            row_records = _extract_row_level(h_rows, bench, run)

            if not subq_records:
                print("no sub-Q records (all rows lack sub_question_log)")
                continue

            df = _build_output(subq_records, row_records, bench, run)
            out = _OUT_DIR / f"dim8_{bench}_{run}.csv"
            df.to_csv(out, index=False, float_format="%.6f")
            print(f"→ {out.name} ({len(df)} rows, {len(subq_records)} sub-Q records)")


if __name__ == "__main__":
    main()
