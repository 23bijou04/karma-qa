"""
Failure mode classifier — Dimension 3. FAILURE_RULES_VERSION=v1.

Bins (first-match-wins, judge_correct==False rows only):
  hedging, no_entity_linked, sparql_error, subq_fusion,
  empty_answer, extraction_fail, timeout, wrong_answer.

Output: benchmark, system, run, bin, failures_in_bin, pct_of_failures, pct_of_total.
Top CSV comment line: # FAILURE_RULES_VERSION=v1
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from scripts.analytics.load_judged import FILE_INVENTORY, load_judged  # noqa: E402

FAILURE_RULES_VERSION = "v1"

# Regex: literal | is alternation — markdown table uses \| for rendering only
HEDGING_RE = re.compile(
    r"\b(i don'?t know|unable to (determine|answer)|cannot determine"
    r"|insufficient (info|information)|no (answer|information) (found|available))\b",
    re.IGNORECASE,
)

SPARQL_ERROR_TOKENS = {"SPARQL", "syntax", "Virtuoso", "timeout"}

_HOP_RE = re.compile(r"^(\d+)hop")


def expected_hops(qtype: str) -> int:
    """Return expected sub-question count for Nhop question types."""
    m = _HOP_RE.match(qtype)
    return int(m.group(1)) if m else 0


def _is_hedging(row: dict) -> bool:
    """Check if answer is an explicit hedge phrase (non-empty text matching HEDGING_RE).

    Empty text is NOT hedging — it falls through to the `empty_answer` bin so
    missing-output failures are distinguishable from explicit refusals.
    """
    text = (row.get("raw_answer") or row.get("predicted_answer") or "").strip()
    return bool(text) and bool(HEDGING_RE.search(text))


def _has_sparql_error(row: dict) -> bool:
    """Check for SPARQL error tokens in top-level error or sub_question_log entries."""
    top_err = row.get("error") or ""
    if any(tok in top_err for tok in SPARQL_ERROR_TOKENS):
        return True
    for sq in row.get("sub_question_log") or []:
        sm = sq.get("solver_meta") or {}
        sq_err = sm.get("error") or ""
        if any(tok in sq_err for tok in SPARQL_ERROR_TOKENS):
            return True
    return False


def classify_failure(row: dict, system: str) -> str:
    """Assign a failure bin to a judge_correct==False row. First match wins."""
    # 1. hedging (all systems)
    if _is_hedging(row):
        return "hedging"

    if system == "hybrid":
        # 2. no_entity_linked
        di = row.get("debug_info") or {}
        el = di.get("entities_linked")
        if (isinstance(el, list) and len(el) == 0) or "no_entity_uris" in (
            row.get("route_source") or ""
        ):
            return "no_entity_linked"

        # 3. sparql_error
        if row.get("method") == "sparql" and _has_sparql_error(row):
            return "sparql_error"

        # 4. subq_fusion — only for Nhop question types
        qtype = row.get("question_type") or ""
        if row.get("method") == "logicrag_decomposed" and _HOP_RE.match(qtype):
            sql = row.get("sub_question_log") or []
            exp = expected_hops(qtype)
            if exp > 0 and len(sql) < exp:
                return "subq_fusion"

    # 5. empty_answer
    pred = (row.get("predicted_answer") or "").strip()
    if not pred:
        return "empty_answer"

    # 6. extraction_fail
    raw = (row.get("raw_answer") or "").strip()
    if raw and not pred:
        return "extraction_fail"

    # 7. timeout
    if row.get("method") == "timeout":
        return "timeout"

    # 8. default
    return "wrong_answer"


def compute_failure_bins(
    bench: str, sys_: str, run: str
) -> list[dict]:
    """Return bin counts for one (bench, sys, run) triple."""
    rows = load_judged(bench, sys_, run)
    total = len(rows)
    failures = [r for r in rows.values() if not r.get("judge_correct")]
    n_failures = len(failures)

    bin_counts: dict[str, int] = {}
    for row in failures:
        b = classify_failure(row, sys_)
        bin_counts[b] = bin_counts.get(b, 0) + 1

    records = []
    for bin_name, count in bin_counts.items():
        records.append(
            {
                "benchmark": bench,
                "system": sys_,
                "run": run,
                "bin": bin_name,
                "failures_in_bin": count,
                "pct_of_failures": count / n_failures if n_failures else 0.0,
                "pct_of_total": count / total if total else 0.0,
            }
        )
    # Sort for determinism
    records.sort(key=lambda r: (r["bin"],))
    return records


def run(bench_filter=None, sys_filter=None, run_filter=None) -> pd.DataFrame:
    """Compute failure bins for all matching (bench, sys, run) triples."""
    all_records: list[dict] = []
    for (bench, sys_, run_) in FILE_INVENTORY:
        if bench_filter and bench != bench_filter:
            continue
        if sys_filter and sys_ != sys_filter:
            continue
        if run_filter and run_ != run_filter:
            continue
        try:
            records = compute_failure_bins(bench, sys_, run_)
            all_records.extend(records)
        except Exception as exc:
            print(f"WARN: ({bench},{sys_},{run_}) skipped: {exc}", file=sys.stderr)

    return pd.DataFrame(all_records)


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comment = f"# FAILURE_RULES_VERSION={FAILURE_RULES_VERSION}\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(comment)
        df.to_csv(fh, index=False, float_format="%.6f")
    print(f"Written: {out_path}")


def validate_conservation(df: pd.DataFrame) -> bool:
    """Check that bin counts sum to total failures per (bench, sys, run).

    We can't easily verify total failures here without reloading; instead we
    verify internal consistency: pct_of_failures sums to ~1.0 per group.
    """
    ok = True
    grouped = df.groupby(["benchmark", "system", "run"])["pct_of_failures"].sum()
    for key, total in grouped.items():
        if abs(total - 1.0) > 0.01:
            print(f"WARN: pct_of_failures sum={total:.4f} (expected 1.0) for {key}", file=sys.stderr)
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure mode bins (Dim 3)")
    parser.add_argument("--all", action="store_true", default=True, help="Process all files")
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
        / "plans/260422-1027-companion-analytics-18-judged-jsonls/reports/data/dim3_failure_modes.csv"
    )
    write_csv(df, out_path)

    if not validate_conservation(df):
        print("Conservation check failed — see warnings above.", file=sys.stderr)
    else:
        print("Conservation check: PASSED (pct_of_failures sums to 1.0 per group)")


if __name__ == "__main__":
    main()
