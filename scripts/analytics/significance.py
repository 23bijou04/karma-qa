"""
Dimension 9 — Between-system McNemar significance per (benchmark, run, pair).

Pairs: (hybrid, linearrag), (hybrid, logicrag), (linearrag, logicrag).
Bonferroni family = 9 per run (3 pairs × 3 benchmarks). R1 and R2 are
independent families — NEVER pooled to 18.
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

# -- optional scipy (import-guarded) --
try:
    from scipy.stats import binomtest as _binomtest  # type: ignore[import]
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy absent — Wilson CI uses normal approximation", stacklevel=1)

BENCHMARKS = ["2wiki", "hotpotqa", "musique"]
RUNS = ["r1", "r2"]
SYSTEMS = ["hybrid", "linearrag", "logicrag"]
PAIRS = [("hybrid", "linearrag"), ("hybrid", "logicrag"), ("linearrag", "logicrag")]
BONFERRONI_FAMILY = 9  # 3 pairs × 3 benchmarks, per run (R1 and R2 independent)

_COLS = [
    "benchmark", "run", "sys_a", "sys_b", "n",
    "acc_a", "acc_a_lo", "acc_a_hi",
    "acc_b", "acc_b_lo", "acc_b_hi",
    "b", "c", "chi2", "p_value", "p_bonferroni",
    "significant_05", "bonferroni_family_size", "family_scope",
]
_FLOAT_COLS = {
    "acc_a", "acc_a_lo", "acc_a_hi", "acc_b", "acc_b_lo", "acc_b_hi",
    "chi2", "p_value", "p_bonferroni",
}


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


def compute_significance(bench: str, run: str) -> list[dict]:
    """Between-system rows for one (benchmark, run)."""
    sys_data = {s: load_judged(bench, s, run) for s in SYSTEMS}
    rows = []
    for sys_a, sys_b in PAIRS:
        da, db = sys_data[sys_a], sys_data[sys_b]
        shared = set(da) & set(db)
        n = len(shared)
        if n < 995:
            warnings.warn(f"[{bench}/{run}] {sys_a} vs {sys_b}: shared={n} < 995", stacklevel=2)
        ai = {q: da[q] for q in shared}
        bi = {q: db[q] for q in shared}
        ak = sum(1 for d in ai.values() if d.get("judge_correct"))
        bk = sum(1 for d in bi.values() if d.get("judge_correct"))
        acc_a, acc_b = ak / n if n else 0.0, bk / n if n else 0.0
        res = mcnemar(ai, bi, round_values=False)  # raw p preserved for Bonferroni
        p_raw = float(res["p_value"])
        p_bonf = min(1.0, p_raw * BONFERRONI_FAMILY)
        rows.append({
            "benchmark": bench, "run": run, "sys_a": sys_a, "sys_b": sys_b, "n": n,
            "acc_a": acc_a, "acc_a_lo": wilson_ci(ak, n)[0], "acc_a_hi": wilson_ci(ak, n)[1],
            "acc_b": acc_b, "acc_b_lo": wilson_ci(bk, n)[0], "acc_b_hi": wilson_ci(bk, n)[1],
            "b": res["regressed"], "c": res["improved"],
            "chi2": float(res["chi2"]), "p_value": p_raw,
            "p_bonferroni": p_bonf, "significant_05": p_bonf < 0.05,
            "bonferroni_family_size": BONFERRONI_FAMILY, "family_scope": "run-local",
        })
    return rows


def run_all(benchmarks: list[str], runs: list[str]) -> None:
    import csv
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for bench in benchmarks:
        for run in runs:
            all_rows.extend(compute_significance(bench, run))
    for bench in benchmarks:
        for run in runs:
            out = _OUT_DIR / f"dim9_{bench}_{run}.csv"
            subset = [r for r in all_rows if r["benchmark"] == bench and r["run"] == run]
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=_COLS)
                w.writeheader()
                for row in subset:
                    w.writerow({c: f"{float(row[c]):.6f}" if c in _FLOAT_COLS else row[c] for c in _COLS})
            print(f"  Wrote {out} ({len(subset)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dim 9: between-system McNemar with Bonferroni per (bench, run)")
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--run", choices=RUNS)
    args = parser.parse_args()
    run_all([args.benchmark] if args.benchmark else BENCHMARKS, [args.run] if args.run else RUNS)
    print("Done — dim9 CSVs written to:", _OUT_DIR)


if __name__ == "__main__":
    main()
