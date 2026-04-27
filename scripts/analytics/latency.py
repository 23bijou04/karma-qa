"""
Latency percentiles for Hybrid runs — Dimension 7.

Hybrid only: group by (run, method, question_type) and emit mean, median,
p95, count from the latency_seconds field.

LinearRAG and LogicRAG lack latency instrumentation — a single summary row
per (bench, system, run) is emitted with stats as "N/A — not instrumented".

Warns to stderr (and emits a warning row) if >5% of rows have
latency_seconds missing or <= 0.

Output CSV columns:
  benchmark, system, run, method, question_type,
  n, mean_s, median_s, p95_s, note
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from scripts.analytics.load_judged import FILE_INVENTORY, load_judged  # noqa: E402

_NA_NOTE = "N/A — not instrumented"
_WARN_THRESHOLD = 0.05  # warn if >5% rows have bad latency


def _compute_latency_stats(values: list[float]) -> dict:
    """Return mean, median, p95 for a list of positive latency values."""
    arr = np.array(values, dtype=float)
    return {
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
    }


def compute_latency_hybrid(bench: str, run_: str) -> list[dict]:
    """Return per-(method, question_type) latency stats for one Hybrid run."""
    rows = load_judged(bench, "hybrid", run_)
    total = len(rows)

    # Collect values per group; track bad latency count
    groups: dict[tuple[str, str], list[float]] = {}
    bad_count = 0

    for row in rows.values():
        lat = row.get("latency_seconds")
        if lat is None or lat <= 0:
            bad_count += 1
            continue
        method = (row.get("method") or "").strip()
        qtype = (row.get("question_type") or "").strip()
        key = (method, qtype)
        groups.setdefault(key, []).append(float(lat))

    bad_rate = bad_count / total if total else 0.0
    if bad_rate > _WARN_THRESHOLD:
        print(
            f"WARN: ({bench}, hybrid, {run_}) has {bad_count}/{total} "
            f"({bad_rate:.1%}) rows with missing or non-positive latency_seconds",
            file=sys.stderr,
        )

    records = []
    for (method, qtype), values in sorted(groups.items()):
        stats = _compute_latency_stats(values)
        records.append(
            {
                "benchmark": bench,
                "system": "hybrid",
                "run": run_,
                "method": method,
                "question_type": qtype,
                "n": len(values),
                **stats,
                "note": f"bad_latency_count={bad_count}" if bad_count else "",
            }
        )
    return records


def _na_row(bench: str, sys_: str, run_: str) -> dict:
    """Emit a single N/A summary row for non-instrumented systems."""
    return {
        "benchmark": bench,
        "system": sys_,
        "run": run_,
        "method": "",
        "question_type": "",
        "n": 0,
        "mean_s": float("nan"),
        "median_s": float("nan"),
        "p95_s": float("nan"),
        "note": _NA_NOTE,
    }


def run(bench_filter=None, sys_filter=None, run_filter=None) -> pd.DataFrame:
    all_records: list[dict] = []

    for (bench, sys_, run_) in FILE_INVENTORY:
        if bench_filter and bench != bench_filter:
            continue
        if sys_filter and sys_ != sys_filter:
            continue
        if run_filter and run_ != run_filter:
            continue

        if sys_ == "hybrid":
            try:
                records = compute_latency_hybrid(bench, run_)
                all_records.extend(records)
            except Exception as exc:
                print(
                    f"WARN: ({bench}, hybrid, {run_}) skipped: {exc}", file=sys.stderr
                )
        else:
            # linearrag / logicrag — not instrumented
            all_records.append(_na_row(bench, sys_, run_))

    return pd.DataFrame(all_records)


def validate_p95_sanity(df: pd.DataFrame) -> None:
    """Warn if Hybrid R1 HotPotQA overall p95 falls outside expected [5s, 60s] range.

    The stored CSV contains per-(method, qtype) p95 values; to compute the true
    overall p95 we reload the raw latency values from the source file.
    """
    try:
        rows = load_judged("hotpotqa", "hybrid", "r1")
    except Exception as exc:
        print(f"WARN: could not reload hotpotqa/hybrid/r1 for sanity check: {exc}", file=sys.stderr)
        return
    lats = [
        float(r["latency_seconds"])
        for r in rows.values()
        if r.get("latency_seconds") and r["latency_seconds"] > 0
    ]
    if not lats:
        print("WARN: No valid latency_seconds for HotPotQA R1 Hybrid sanity check.", file=sys.stderr)
        return
    overall_p95 = float(np.percentile(lats, 95))
    if not (5.0 <= overall_p95 <= 60.0):
        print(
            f"WARN: HotPotQA R1 Hybrid overall-p95={overall_p95:.2f}s outside expected [5s, 60s]",
            file=sys.stderr,
        )
    else:
        print(f"Sanity check: HotPotQA R1 Hybrid overall-p95={overall_p95:.2f}s — OK")


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Written: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency percentiles — Hybrid only (Dim 7)")
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    df = run(bench_filter=args.benchmark, sys_filter=args.system, run_filter=args.run)
    if df.empty:
        print("No data — check filters or file availability.", file=sys.stderr)
        sys.exit(1)

    out_path = (
        _HERE
        / "plans/260422-1027-companion-analytics-18-judged-jsonls/reports/data"
        / "dim7_latency.csv"
    )
    write_csv(df, out_path)
    validate_p95_sanity(df)


if __name__ == "__main__":
    main()
