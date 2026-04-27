"""Routing breakdown for Table 4 (paper).

Classifies each Hybrid prediction into one of four execution modes, then
aggregates judge accuracy + substring containment per mode per benchmark.

Modes:
  1. Direct SPARQL        -> top-level method == "sparql"
  2. Decomp: all SPARQL   -> logicrag_decomposed, all sub_question methods sparql
  3. Decomp: SPARQL+text  -> logicrag_decomposed, at least one sparql AND one text sub
  4. Text only            -> top-level method == "text" OR
                             logicrag_decomposed with all sub methods == text

Containment is recomputed from predicted_answer vs gold_answer via
evaluation.metrics.compute_containment to avoid relying on stored (stale) fields.

Usage:
    python scripts/compute-routing-breakdown.py
Output:
    <thirag-paper>/figures/data/routing-breakdown.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evaluation.metrics import compute_containment  # noqa: E402

_D = REPO / "data"
FIG_DATA = Path("~/Documents/2026/2026_02/thirag-paper/figures/data").expanduser()

# Hybrid-only, run1 only (matches paper Table 4 convention).
BENCHMARKS = {
    "2WikiMHQA": _D / "eval_1000q/hybrid/hybrid_eval_1000_judged.jsonl",
    "HotpotQA":  _D / "cross_benchmark/hotpotqa/hybrid_hotpotqa_1000q_judged.jsonl",
    "MuSiQue":   _D / "cross_benchmark/musique/hybrid_musique_1000q_judged.jsonl",
}

MODE_ORDER = [
    "Direct SPARQL",
    "Decomposed: all SPARQL",
    "Decomposed: SPARQL+text",
    "Text only",
]

# Paper Table 5 any-SPARQL % audit targets (within drift).
# Values sourced from authoritative method-based classifier (same classifier
# used in `classify()` above). Keep in sync when regenerating the paper table.
AUDIT_ANY_SPARQL = {"2WikiMHQA": 70.0, "HotpotQA": 17.4, "MuSiQue": 42.6}
AUDIT_TOL_PP = 2.0


def classify(record: dict) -> str:
    """Return one of MODE_ORDER for a Hybrid prediction record."""
    top = record.get("method")
    if top == "sparql":
        return "Direct SPARQL"
    if top == "text":
        return "Text only"
    if top == "logicrag_decomposed":
        subs = record.get("sub_question_log") or []
        sub_methods = {s.get("method") for s in subs}
        # Defensive: if no subs logged, treat like top "text".
        if not sub_methods:
            return "Text only"
        has_sparql = "sparql" in sub_methods
        has_text = "text" in sub_methods
        if has_sparql and not has_text:
            return "Decomposed: all SPARQL"
        if has_sparql and has_text:
            return "Decomposed: SPARQL+text"
        return "Text only"
    raise ValueError(f"unknown method: {top!r}")


def _cont(record: dict) -> float:
    gold = record.get("gold_answer") or ""
    pred = record.get("predicted_answer") or ""
    if not gold or not pred:
        return 0.0
    return compute_containment(pred, gold)


def aggregate(path: Path) -> dict[str, dict]:
    """Return {mode: {n, acc, cont}} plus {"All": {...}}."""
    buckets: dict[str, list[tuple[bool, float]]] = {m: [] for m in MODE_ORDER}
    all_outcomes: list[tuple[bool, float]] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            mode = classify(r)
            judge = bool(r.get("judge_correct"))
            cont = _cont(r)
            buckets[mode].append((judge, cont))
            all_outcomes.append((judge, cont))

    def summarize(items: list[tuple[bool, float]]) -> dict:
        n = len(items)
        if n == 0:
            return {"n": 0, "acc": 0.0, "cont": 0.0, "pct": 0.0}
        acc = sum(1 for j, _ in items if j) / n * 100
        cont = sum(c for _, c in items) / n * 100
        return {"n": n, "acc": round(acc, 1), "cont": round(cont, 1)}

    total = len(all_outcomes)
    out: dict[str, dict] = {}
    for m in MODE_ORDER:
        s = summarize(buckets[m])
        s["pct"] = round(s["n"] / total * 100, 1) if total else 0.0
        out[m] = s
    all_s = summarize(all_outcomes)
    all_s["pct"] = 100.0
    out["All"] = all_s
    return out


def audit(bench: str, agg: dict) -> None:
    """Check Any-SPARQL% against paper Table 5 within tolerance."""
    any_sparql_pct = (
        agg["Direct SPARQL"]["pct"]
        + agg["Decomposed: all SPARQL"]["pct"]
        + agg["Decomposed: SPARQL+text"]["pct"]
    )
    expected = AUDIT_ANY_SPARQL[bench]
    delta = abs(any_sparql_pct - expected)
    status = "PASS" if delta <= AUDIT_TOL_PP else "FAIL"
    print(f"{status} {bench}: any-SPARQL {any_sparql_pct:.1f}% vs paper {expected}% "
          f"(delta {delta:.1f}pp, tol {AUDIT_TOL_PP}pp)")
    if status == "FAIL":
        sys.exit(1)


def write_csv(results: dict[str, dict[str, dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "mode", "n", "pct_questions", "acc_pct", "cont_pct"])
        for bench, agg in results.items():
            for mode in MODE_ORDER + ["All"]:
                row = agg[mode]
                w.writerow([bench, mode, row["n"], row["pct"], row["acc"], row["cont"]])
    print(f"wrote {out_path}")


def print_table(results: dict[str, dict[str, dict]]) -> None:
    """Pretty-print a preview so we can paste numbers into LaTeX."""
    for bench, agg in results.items():
        print(f"\n=== {bench} ===")
        print(f"{'Mode':<26} {'NB':>5} {'%':>6} {'Acc':>6} {'Cont':>6}")
        for mode in MODE_ORDER + ["All"]:
            r = agg[mode]
            print(f"{mode:<26} {r['n']:>5} {r['pct']:>6} {r['acc']:>6} {r['cont']:>6}")


def main() -> None:
    results = {}
    for bench, path in BENCHMARKS.items():
        results[bench] = aggregate(path)
        audit(bench, results[bench])
    print_table(results)
    write_csv(results, FIG_DATA / "routing-breakdown.csv")


if __name__ == "__main__":
    main()
