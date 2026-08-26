"""Posterior uncertainty, calibration, and proper scoring rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math


def _validate_labels_probabilities(
    labels: Sequence[int | bool], probabilities: Sequence[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities have different sizes")
    if not labels:
        raise ValueError("metric requires at least one observation")
    normalized_labels: list[int] = []
    normalized_probabilities: list[float] = []
    for label, probability in zip(labels, probabilities):
        normalized = int(label)
        if normalized not in {0, 1}:
            raise ValueError(f"binary label must be zero or one, got {label!r}")
        score = float(probability)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"probability must be finite and in [0, 1], got {probability!r}")
        normalized_labels.append(normalized)
        normalized_probabilities.append(score)
    return tuple(normalized_labels), tuple(normalized_probabilities)


def binary_entropy(probability: float, *, base: float = 2.0) -> float:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    if base <= 0.0 or base == 1.0:
        raise ValueError("entropy base must be positive and not one")
    if value in {0.0, 1.0}:
        return 0.0
    return -(value * math.log(value, base) + (1.0 - value) * math.log(1.0 - value, base))


def posterior_entropy(
    posterior: Mapping[str, float], *, states: Sequence[str] = ("theta0", "theta1"), base: float = 2.0
) -> float:
    if set(posterior) != set(states):
        raise ValueError(f"posterior must contain exactly {list(states)}")
    probabilities = [float(posterior[state]) for state in states]
    if any(not math.isfinite(item) or item < 0.0 for item in probabilities):
        raise ValueError("posterior probabilities must be finite and non-negative")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("posterior probabilities must sum to one")
    if base <= 0.0 or base == 1.0:
        raise ValueError("entropy base must be positive and not one")
    return -sum(item * math.log(item, base) for item in probabilities if item > 0.0)


def mean_entropy_reduction(
    prior: Mapping[str, float], posteriors: Iterable[Mapping[str, float]]
) -> float:
    materialized = tuple(posteriors)
    if not materialized:
        raise ValueError("entropy reduction requires at least one posterior")
    prior_entropy = posterior_entropy(prior)
    return prior_entropy - sum(posterior_entropy(item) for item in materialized) / len(materialized)


def binary_brier_score(
    labels: Sequence[int | bool], probabilities: Sequence[float]
) -> float:
    actual, predicted = _validate_labels_probabilities(labels, probabilities)
    return sum((score - label) ** 2 for label, score in zip(actual, predicted)) / len(actual)


def binary_log_loss(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    epsilon: float = 1e-15,
) -> float:
    actual, predicted = _validate_labels_probabilities(labels, probabilities)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between zero and one half")
    loss = 0.0
    for label, score in zip(actual, predicted):
        bounded = min(1.0 - epsilon, max(epsilon, score))
        loss -= label * math.log(bounded) + (1 - label) * math.log(1.0 - bounded)
    return loss / len(actual)


def expected_calibration_error(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    actual, predicted = _validate_labels_probabilities(labels, probabilities)
    if bins <= 0:
        raise ValueError("bins must be positive")
    grouped: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for label, score in zip(actual, predicted):
        index = min(bins - 1, int(score * bins))
        grouped[index].append((label, score))
    total = len(actual)
    error = 0.0
    for group in grouped:
        if not group:
            continue
        empirical = sum(label for label, _ in group) / len(group)
        confidence = sum(score for _, score in group) / len(group)
        error += len(group) / total * abs(empirical - confidence)
    return error


def invalid_output_rate(indicators: Iterable[int | bool]) -> float:
    values = tuple(int(item) for item in indicators)
    if not values:
        raise ValueError("invalid-output rate requires at least one observation")
    if any(item not in {0, 1} for item in values):
        raise ValueError("invalid-output indicators must be binary")
    return sum(values) / len(values)


__all__ = [
    "binary_brier_score",
    "binary_entropy",
    "binary_log_loss",
    "expected_calibration_error",
    "invalid_output_rate",
    "mean_entropy_reduction",
    "posterior_entropy",
]
