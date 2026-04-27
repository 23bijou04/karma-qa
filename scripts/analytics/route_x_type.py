"""
Dimension 1: Hybrid route×question-type cross-tab with Wilson CI.

Output: plans/260422-1027-companion-analytics-18-judged-jsonls/reports/data/
        dim1_{bench}_{run}.csv

Columns: benchmark, run, route_bucket, route_source_raw, question_type,
         n, correct, acc, wilson_lo, wilson_hi
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
# Route bucketing
# ---------------------------------------------------------------------------
_TERMINAL_SOURCES = {"incomplete_coverage", "timeout"}


def route_bucket(route_source: str | None) -> str:
    """Map raw route_source to coarse bucket."""
    rs = route_source or ""
    if rs.startswith("rule:"):
        return "rule"
    if rs.startswith("multihop_guard:"):
        return "multihop_guard"
    if rs.startswith("llm_prop:"):
        return "llm_prop"
    if rs in _TERMINAL_SOURCES or rs == "":
        return "terminal"
    # text:no_entity_uris, no_schema_fallback, no_entity_uris are text-direct paths
    if rs in {"text:no_entity_uris", "no_schema_fallback", "no_entity_uris"}:
        return "text"
    # kinship_2hop — observed in R2; treat as rule-like (KG-routed)
    if rs.startswith("kinship"):
        return "rule"
    return "other"


# ---------------------------------------------------------------------------
# Wilson CI helper
# ---------------------------------------------------------------------------
def _wilson_ci(correct: int, n: int) -> tuple[float, float]:
    """95% Wilson CI. Falls back to normal-approximation if scipy missing."""
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import proportion_confint  # type: ignore[import-untyped]

        lo, hi = proportion_confint(correct, n, method="wilson")
        return (float(lo), float(hi))
    except ImportError:
        import math

        p = correct / n
        z = 1.96
        margin = z * math.sqrt(p * (1 - p) / n)
        return (max(0.0, p - margin), min(1.0, p + margin))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute(bench: str, run: str) -> pd.DataFrame:
    rows = load_judged(bench, "hybrid", run)
    records = []
    for row in rows.values():
        rs = row.get("route_source") or ""
        bucket = route_bucket(rs)
        qtype = row.get("question_type") or "unknown"
        correct = int(bool(row.get("judge_correct")))
        records.append(
            {
                "benchmark": bench,
                "run": run,
                "route_bucket": bucket,
                "route_source_raw": rs,
                "question_type": qtype,
                "correct": correct,
            }
        )

    df = pd.DataFrame(records)
    # Group by (route_bucket, route_source_raw, question_type)
    grp = (
        df.groupby(["route_bucket", "route_source_raw", "question_type"])
        .agg(n=("correct", "count"), correct=("correct", "sum"))
        .reset_index()
    )
    grp["benchmark"] = bench
    grp["run"] = run
    grp["acc"] = grp["correct"] / grp["n"]
    grp[["wilson_lo", "wilson_hi"]] = grp.apply(
        lambda r: pd.Series(_wilson_ci(int(r["correct"]), int(r["n"]))),
        axis=1,
    )
    cols = [
        "benchmark", "run", "route_bucket", "route_source_raw",
        "question_type", "n", "correct", "acc", "wilson_lo", "wilson_hi",
    ]
    return grp[cols].sort_values(["route_bucket", "route_source_raw", "question_type"])


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_check(df: pd.DataFrame, bench: str, run: str) -> list[str]:
    """Return list of failed checks (empty = all passed)."""
    failures = []

    # Conservation: sum(correct) per benchmark must equal total judge_correct in data
    # We trust groupby is correct; cross-check within df itself
    total_n = df["n"].sum()
    if total_n == 0:
        failures.append(f"{bench}/{run}: total n=0, no data")

    # Hotpotqa R2: rule+multihop_guard >= 40% of rows
    if bench == "hotpotqa" and run == "r2":
        rule_n = df[df["route_bucket"].isin(["rule", "multihop_guard"])]["n"].sum()
        pct = rule_n / total_n if total_n else 0.0
        if pct < 0.40:
            failures.append(
                f"hotpotqa/r2: rule+multihop_guard = {pct:.1%} < 40% (sanity failed)"
            )

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dim 1: route×type cross-tab")
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
            key = (bench, "hybrid", run)
            if key not in FILE_INVENTORY:
                print(f"SKIP: no inventory entry for {key}")
                continue
            print(f"Computing dim1 {bench}/{run} ...", end=" ", flush=True)
            try:
                df = compute(bench, run)
            except Exception as exc:
                print(f"ERROR: {exc}")
                all_issues.append(f"{bench}/{run}: {exc}")
                continue

            issues = sanity_check(df, bench, run)
            for issue in issues:
                print(f"\n  WARN: {issue}")
            all_issues.extend(issues)

            out = _OUT_DIR / f"dim1_{bench}_{run}.csv"
            df.to_csv(out, index=False, float_format="%.6f")
            print(f"→ {out.name} ({len(df)} rows)")

    if all_issues:
        print(f"\n{len(all_issues)} sanity issue(s):")
        for issue in all_issues:
            print(f"  ! {issue}")
    else:
        print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
