"""
Dimension 4 — R1↔R2 flip-rate per (benchmark, system, question_type).

# Note: flip-rate p-values raw (single paired test per cell). Between-system
# significance (dim 9) corrected with Bonferroni family=9 per run.

Engine: abs(b-c)<25 or b==0 or c==0 → statsmodels exact binomial (if available)
        else fallback to repo chi2 with engine="chi2_smallcell_biased" + stderr WARNING.
        Otherwise: repo chi2 with engine="chi2".
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO / "plans" / "260422-1027-companion-analytics-18-judged-jsonls" / "reports" / "data"

# -- load_judged (Phase 01) --
_lj = importlib.util.spec_from_file_location("load_judged", _REPO / "scripts/analytics/load_judged.py")
_lj_mod = importlib.util.module_from_spec(_lj)
_lj.loader.exec_module(_lj_mod)
load_judged = _lj_mod.load_judged

# -- mcnemar from repo --
_ces = importlib.util.spec_from_file_location(
    "compare_evals_statistical", _REPO / "scripts/compare_evals_statistical.py"
)
_ces_mod = importlib.util.module_from_spec(_ces)
_ces.loader.exec_module(_ces_mod)
mcnemar = _ces_mod.mcnemar

# -- optional deps (import-guarded) --
try:
    from scipy.stats import binomtest as _binomtest  # type: ignore[import]
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy absent — Wilson CI uses normal approximation", stacklevel=1)

try:
    from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar  # type: ignore[import]
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False

BENCHMARKS = ["2wiki", "hotpotqa", "musique"]
SYSTEMS = ["hybrid", "linearrag", "logicrag"]
_COLS = [
    "benchmark", "system", "question_type", "n",
    "r1_acc", "r1_lo", "r1_hi", "r2_acc", "r2_lo", "r2_hi",
    "flips", "flip_pct", "b", "c", "chi2", "p_value", "engine",
    "bonferroni_family_size", "family_scope",
]
_FLOAT_COLS = {"r1_acc", "r1_lo", "r1_hi", "r2_acc", "r2_lo", "r2_hi", "flip_pct", "chi2", "p_value"}


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """95% Wilson CI; normal-approx fallback if scipy absent."""
    if n == 0:
        return (0.0, 0.0)
    if _SCIPY:
        ci = _binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        return (float(ci.low), float(ci.high))
    import math
    p_hat = k / n
    m = 1.95996 * math.sqrt(p_hat * (1 - p_hat) / n)
    return (max(0.0, p_hat - m), min(1.0, p_hat + m))


def _engine(b: int, c: int, r1d: dict, r2d: dict) -> tuple[float, float, str]:
    """Select McNemar engine; return (chi2, p_value, engine_name)."""
    use_exact = abs(b - c) < 25 or b == 0 or c == 0
    if use_exact:
        if _STATSMODELS:
            import numpy as np  # type: ignore[import]
            result = _sm_mcnemar(np.array([[0, b], [c, 0]]), exact=True)
            stat = float(result.statistic) if result.statistic is not None else 0.0
            return (stat, float(result.pvalue), "exact_binomial")
        warnings.warn(
            f"statsmodels absent; b+c={b + c} small-cell — using chi2 (biased). "
            "Install statsmodels for exact binomial.",
            stacklevel=4,
        )
        res = mcnemar(r1d, r2d, round_values=False)
        return (float(res["chi2"]), float(res["p_value"]), "chi2_smallcell_biased")
    res = mcnemar(r1d, r2d, round_values=False)
    return (float(res["chi2"]), float(res["p_value"]), "chi2")


def compute_flip_rate(bench: str, system: str) -> list[dict]:
    """Flip-rate rows for one (benchmark, system) across all question_types."""
    r1 = load_judged(bench, system, "r1")
    r2 = load_judged(bench, system, "r2")
    shared = set(r1) & set(r2)
    if len(shared) < 995:
        warnings.warn(f"[{bench}/{system}] shared={len(shared)} < 995 — data gaps", stacklevel=2)

    qtype_qids: dict[str, list[str]] = {}
    for qid in shared:
        qtype_qids.setdefault(r1[qid].get("question_type", "unknown"), []).append(qid)

    rows = []
    for qtype, qids in sorted(qtype_qids.items()):
        n = len(qids)
        r1_qt = {q: r1[q] for q in qids}
        r2_qt = {q: r2[q] for q in qids}
        r1_k = sum(1 for d in r1_qt.values() if d.get("judge_correct"))
        r2_k = sum(1 for d in r2_qt.values() if d.get("judge_correct"))
        flips = sum(1 for q in qids if bool(r1[q].get("judge_correct")) != bool(r2[q].get("judge_correct")))
        res = mcnemar(r1_qt, r2_qt, round_values=False)
        b, c = res["regressed"], res["improved"]
        chi2_val, p_val, engine = _engine(b, c, r1_qt, r2_qt)
        rows.append({
            "benchmark": bench, "system": system, "question_type": qtype, "n": n,
            "r1_acc": r1_k / n if n else 0.0, "r1_lo": wilson_ci(r1_k, n)[0], "r1_hi": wilson_ci(r1_k, n)[1],
            "r2_acc": r2_k / n if n else 0.0, "r2_lo": wilson_ci(r2_k, n)[0], "r2_hi": wilson_ci(r2_k, n)[1],
            "flips": flips, "flip_pct": flips / n if n else 0.0,
            "b": b, "c": c, "chi2": chi2_val, "p_value": p_val, "engine": engine,
            "bonferroni_family_size": 1, "family_scope": "flip-rate-raw",
        })
    return rows


def run_all(benchmarks: list[str], systems: list[str]) -> None:
    import csv
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for bench in benchmarks:
        for sys_ in systems:
            all_rows.extend(compute_flip_rate(bench, sys_))
    for bench in benchmarks:
        out = _OUT_DIR / f"dim4_{bench}.csv"
        bench_rows = [r for r in all_rows if r["benchmark"] == bench]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_COLS)
            w.writeheader()
            for row in bench_rows:
                w.writerow({c: f"{float(row[c]):.6f}" if c in _FLOAT_COLS else row[c] for c in _COLS})
        print(f"  Wrote {out} ({len(bench_rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dim 4: R1↔R2 flip-rate per (bench, sys, qtype)")
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--run", choices=["r1", "r2"], help="(unused; R1↔R2 always paired)")
    args = parser.parse_args()
    benchmarks = [args.benchmark] if args.benchmark else BENCHMARKS
    run_all(benchmarks, SYSTEMS)
    print("Done — dim4 CSVs written to:", _OUT_DIR)


if __name__ == "__main__":
    main()
