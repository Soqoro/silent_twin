"""Small deterministic bootstrap helpers for matched Tier-1 cohorts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import random
from typing import TypeVar


T = TypeVar("T")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    fn = statistic or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(float(fn(sample)))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return _quantile(estimates, alpha), _quantile(estimates, 1.0 - alpha)


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """CI for mean(left - right), resampling matched pairs together."""

    if len(left) != len(right):
        raise ValueError("paired bootstrap cohorts have different sizes")
    if not left:
        raise ValueError("paired bootstrap requires at least one matched pair")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    return bootstrap_ci(
        differences, confidence=confidence, resamples=resamples, seed=seed
    )

