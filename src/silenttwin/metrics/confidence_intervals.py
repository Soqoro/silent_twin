"""Small deterministic bootstrap helpers for matched Tier-1 cohorts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from itertools import product
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


def _validate_clusters(
    observations: Sequence[T], cluster_ids: Sequence[Hashable]
) -> dict[Hashable, list[T]]:
    if len(observations) != len(cluster_ids):
        raise ValueError("observations and cluster_ids have different sizes")
    if not observations:
        raise ValueError("clustered inference requires at least one observation")
    clusters: dict[Hashable, list[T]] = defaultdict(list)
    for observation, cluster_id in zip(observations, cluster_ids):
        try:
            hash(cluster_id)
        except TypeError as exc:
            raise TypeError("cluster IDs must be hashable") from exc
        clusters[cluster_id].append(observation)
    return dict(clusters)


def cluster_bootstrap_statistic_ci(
    observations: Sequence[T],
    cluster_ids: Sequence[Hashable],
    statistic: Callable[[Sequence[T]], float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a statistic by resampling complete task clusters.

    Repeated decoding seeds or rows for one public task therefore move
    together.  The statistic is evaluated on the reconstructed row sample,
    which supports nonlinear metrics such as AUC as well as means.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    clusters = _validate_clusters(observations, cluster_ids)
    identifiers = tuple(clusters)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sample: list[T] = []
        for _ in identifiers:
            selected = identifiers[rng.randrange(len(identifiers))]
            sample.extend(clusters[selected])
        estimate = float(statistic(sample))
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        raise ValueError("statistic was undefined for every cluster bootstrap resample")
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return _quantile(estimates, alpha), _quantile(estimates, 1.0 - alpha)


def task_cluster_bootstrap_ci(
    values: Sequence[float],
    task_ids: Sequence[Hashable],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """CI for a task-weighted mean, averaging repeated rows within each task."""

    clusters = _validate_clusters(values, task_ids)
    task_means = [sum(float(item) for item in rows) / len(rows) for rows in clusters.values()]
    return bootstrap_ci(
        task_means, confidence=confidence, resamples=resamples, seed=seed
    )


def paired_task_cluster_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    task_ids: Sequence[Hashable],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired CI with the public task, rather than each row, as the unit."""

    if len(left) != len(right):
        raise ValueError("paired cohorts have different sizes")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    return task_cluster_bootstrap_ci(
        differences,
        task_ids,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def paired_cluster_permutation_p_value(
    left: Sequence[float],
    right: Sequence[float],
    task_ids: Sequence[Hashable],
    *,
    permutations: int = 10000,
    seed: int = 0,
    exact_cluster_limit: int = 18,
) -> float:
    """Two-sided paired sign-flip test on task-level mean differences."""

    if len(left) != len(right):
        raise ValueError("paired cohorts have different sizes")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    clusters = _validate_clusters(differences, task_ids)
    task_differences = tuple(
        sum(rows) / len(rows) for rows in clusters.values()
    )
    observed = abs(sum(task_differences) / len(task_differences))

    if len(task_differences) <= exact_cluster_limit:
        sign_vectors = product((-1.0, 1.0), repeat=len(task_differences))
        extreme = 0
        total = 0
        for signs in sign_vectors:
            estimate = abs(
                sum(sign * value for sign, value in zip(signs, task_differences))
                / len(task_differences)
            )
            extreme += estimate >= observed - 1e-15
            total += 1
        return extreme / total

    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        estimate = abs(
            sum(
                (-value if rng.randrange(2) else value)
                for value in task_differences
            )
            / len(task_differences)
        )
        extreme += estimate >= observed - 1e-15
    # Add-one correction avoids a spuriously exact zero Monte Carlo p-value.
    return (extreme + 1) / (permutations + 1)


__all__ = [
    "bootstrap_ci",
    "cluster_bootstrap_statistic_ci",
    "paired_bootstrap_ci",
    "paired_cluster_permutation_p_value",
    "paired_task_cluster_bootstrap_ci",
    "task_cluster_bootstrap_ci",
]
