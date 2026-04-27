"""
Non-judge EM/F1/containment + hedging/extraction rates — Dimension 5.

For each (bench, sys, run), iterate rows and compute per-row metrics via
evaluation.metrics. Aggregate by (question_type, method).

LinearRAG/LogicRAG may lack a meaningful method field — their method column
holds a system-level constant (e.g. "linearrag", "logicrag_original"), not a
routing tag. We use it as-is for aggregation.

Output CSV columns:
  benchmark, system, run, question_type, method,
  n, em_mean, f1_mean, containment_mean, hedge_rate, extraction_rate
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from evaluation.metrics import compute_containment, compute_em, compute_f1  # noqa: E402
from scripts.analytics.load_judged import FILE_INVENTORY, load_judged  # noqa: E402

# Same regex as failure_modes.py — single source is here; imported there via
# the package boundary. Keep in sync manually if updated.
HEDGING_RE = re.compile(
    r"\b(i don'?t know|unable to (determine|answer)|cannot determine"
    r"|insufficient (info|information)|no (answer|information) (found|available))\b",
    re.IGNORECASE,
)


def _is_hedging(text: str) -> bool:
    """True if text is empty/whitespace or matches hedging pattern."""
    stripped = text.strip()
    return (not stripped) or bool(HEDGING_RE.search(stripped))


def _best_em_f1_containment(
    pred: str, gold: str
) -> tuple[float, float, float]:
    """Compute EM/F1/containment for a single prediction vs single gold string."""
    em = compute_em(pred, gold)
    f1 = compute_f1(pred, gold)
    con = compute_containment(pred, gold)
    return em, f1, con


def _aggregate_group(rows: list[dict]) -> dict:
    """Aggregate metric lists into means."""
    n = len(rows)
    if n == 0:
        return {}
    em_sum = f1_sum = con_sum = hedge_n = extract_n = 0.0
    for r in rows:
        em_sum += r["em"]
        f1_sum += r["f1"]
        con_sum += r["containment"]
        hedge_n += r["hedging"]
        extract_n += r["extracted"]
    return {
        "n": n,
        "em_mean": em_sum / n,
        "f1_mean": f1_sum / n,
        "containment_mean": con_sum / n,
        "hedge_rate": hedge_n / n,
        "extraction_rate": extract_n / n,
    }


def compute_nonjudge(bench: str, sys_: str, run: str) -> list[dict]:
    """Return aggregated per-(question_type, method) rows for one (bench, sys, run)."""
    rows = load_judged(bench, sys_, run)

    # Group: (qtype, method) -> list of per-row metric dicts
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in rows.values():
        pred = (row.get("predicted_answer") or "").strip()
        gold = (row.get("gold_answer") or "").strip()
        raw = (row.get("raw_answer") or "").strip()

        em, f1, con = _best_em_f1_containment(pred, gold)
        hedging = _is_hedging(raw or pred)
        extracted = len(pred) > 0

        method = (row.get("method") or "").strip()
        qtype = (row.get("question_type") or "").strip()

        groups[(qtype, method)].append(
            {
                "em": em,
                "f1": f1,
                "containment": con,
                "hedging": float(hedging),
                "extracted": float(extracted),
            }
        )

    records = []
    for (qtype, method), metric_rows in sorted(groups.items()):
        agg = _aggregate_group(metric_rows)
        records.append(
            {
                "benchmark": bench,
                "system": sys_,
                "run": run,
                "question_type": qtype,
                "method": method,
                **agg,
            }
        )
    return records


def run(bench_filter=None, sys_filter=None, run_filter=None) -> pd.DataFrame:
    all_records: list[dict] = []
    for (bench, sys_, run_) in FILE_INVENTORY:
        if bench_filter and bench != bench_filter:
            continue
        if sys_filter and sys_ != sys_filter:
            continue
        if run_filter and run_ != run_filter:
            continue
        try:
            records = compute_nonjudge(bench, sys_, run_)
            all_records.extend(records)
        except Exception as exc:
            print(f"WARN: ({bench},{sys_},{run_}) skipped: {exc}", file=sys.stderr)

    return pd.DataFrame(all_records)


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Written: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-judge EM/F1/containment metrics (Dim 5)")
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
        / "dim5_nonjudge_metrics.csv"
    )
    write_csv(df, out_path)


if __name__ == "__main__":
    main()
