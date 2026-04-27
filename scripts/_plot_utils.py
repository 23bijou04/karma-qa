"""Shared helpers for paper figure scripts.

Private module (underscore prefix) imported by plot-*.py siblings via
``importlib`` (since the kebab-case script filenames are not importable).

Contains:
- Wilson 95% CI (hand-rolled, no statsmodels dep)
- McNemar test with paired contingency (hand-rolled binomial)
- Okabe-Ito palette (colorblind-safe) + system-to-color mapping
- matplotlib rcParams for LNCS single-column figures
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Okabe-Ito palette (colorblind-safe, grayscale-distinguishable).
# Hybrid is the darkest (#0072B2) so it dominates grayscale prints.
OKABE_ITO = {
    "Hybrid":    "#0072B2",   # blue
    "LogicRAG":  "#E69F00",   # orange
    "LinearRAG": "#009E73",   # green
}
MARKERS = {
    "Hybrid":    "o",
    "LogicRAG":  "s",
    "LinearRAG": "^",
}

# LNCS single-column is ~4.8 in wide; figsize tuned for that.
LNCS_RCPARAMS = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n at confidence 1-alpha.

    Returns (lo, hi) on the [0, 1] scale. Hand-rolled to avoid statsmodels dep.
    See https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class McNemarResult:
    """McNemar paired-test outcome.

    ``b``: system-A-correct & system-B-wrong. ``c``: A-wrong & B-correct.
    ``p_value``: exact binomial two-sided p-value on the b/c split.
    """
    b: int
    c: int
    p_value: float


def mcnemar_exact(system_a_correct: list[bool], system_b_correct: list[bool]) -> McNemarResult:
    """Exact McNemar test on paired binary outcomes (same questions, two systems).

    Uses the exact binomial distribution on discordant pairs (b, c), avoiding
    the chi-square approximation that fails for small samples. Hand-rolled.
    """
    if len(system_a_correct) != len(system_b_correct):
        raise ValueError(f"Paired lengths differ: {len(system_a_correct)} vs {len(system_b_correct)}")
    b = sum(1 for a, x in zip(system_a_correct, system_b_correct) if a and not x)
    c = sum(1 for a, x in zip(system_a_correct, system_b_correct) if not a and x)
    n = b + c
    if n == 0:
        return McNemarResult(b=0, c=0, p_value=1.0)
    # Two-sided exact binomial on min(b, c) under H0: P(success) = 0.5.
    k = min(b, c)
    # P(X <= k) + P(X >= n-k) where X ~ Binomial(n, 0.5).
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_two = min(1.0, 2 * tail)  # symmetric distribution at p=0.5
    return McNemarResult(b=b, c=c, p_value=p_two)


def significance_stars(p: float, alpha_levels: tuple[float, ...] = (0.001, 0.01, 0.05)) -> str:
    """Conventional star notation: *** p<0.001, ** p<0.01, * p<0.05."""
    a3, a2, a1 = alpha_levels
    if p < a3:
        return "***"
    if p < a2:
        return "**"
    if p < a1:
        return "*"
    return ""
