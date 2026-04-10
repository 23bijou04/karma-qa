"""Stratified sampling and statistical utilities for ablation study."""

import json
import math
import random
from typing import Optional

# Standard evidence chain lengths per question type
STANDARD_EVIDENCE_LENGTHS = {
    "comparison": 2,
    "compositional": 2,
    "inference": 2,
    "bridge_comparison": 4,
}


def create_stratified_sample(
    data_file: str,
    n_per_type: int = 50,
    seed: int = 42,
    output_file: Optional[str] = None,
) -> list[dict]:
    """Create a stratified random sample from the dev set.

    Filters to questions with standard-length evidence chains,
    then samples n_per_type from each of the 4 question types.
    """
    with open(data_file, "r") as f:
        data = json.load(f)

    # Group by type, filtering for valid evidence lengths
    by_type: dict[str, list[dict]] = {}
    for q in data:
        qtype = q.get("type", "")
        evidences = q.get("evidences", [])
        expected_len = STANDARD_EVIDENCE_LENGTHS.get(qtype)

        if not evidences or expected_len is None:
            continue
        if len(evidences) != expected_len:
            continue

        by_type.setdefault(qtype, []).append(q)

    # Sample from each type
    rng = random.Random(seed)
    sample = []
    for qtype in sorted(by_type.keys()):
        pool = by_type[qtype]
        n = min(n_per_type, len(pool))
        sample.extend(rng.sample(pool, n))

    if output_file:
        with open(output_file, "w") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)

    return sample


def wilson_confidence_interval(
    successes: int, total: int, confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)

    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator

    return (max(0.0, center - spread), min(1.0, center + spread))


def mcnemar_test(results_a: list[bool], results_b: list[bool]) -> tuple[float, float]:
    """McNemar's test for paired binary outcomes.

    Returns (chi2_statistic, p_value).
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result lists must have equal length")

    # Count discordant pairs
    b = sum(1 for a, bb in zip(results_a, results_b) if not a and bb)  # A wrong, B right
    c = sum(1 for a, bb in zip(results_a, results_b) if a and not bb)  # A right, B wrong

    if b + c == 0:
        return (0.0, 1.0)

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value from chi-squared distribution (1 df)
    # Using survival function approximation
    p_value = _chi2_sf(chi2, df=1)

    return (chi2, p_value)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Approximate chi-squared survival function (p-value).

    Uses the Wilson-Hilferty approximation for simplicity.
    """
    if x <= 0:
        return 1.0
    # For df=1, use the normal approximation
    z = math.sqrt(x)
    # Standard normal survival function approximation
    # Chi-squared(df=1) survival: P(X>x) = erfc(sqrt(x) / sqrt(2))
    return math.erfc(z / math.sqrt(2))


def bootstrap_ci(
    scores: list[float], n_bootstrap: int = 1000,
    confidence: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a mean score (e.g., F1)."""
    if not scores:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(scores)
    means = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    low_idx = int(alpha / 2 * n_bootstrap)
    high_idx = int((1 - alpha / 2) * n_bootstrap)

    return (means[low_idx], means[min(high_idx, n_bootstrap - 1)])
