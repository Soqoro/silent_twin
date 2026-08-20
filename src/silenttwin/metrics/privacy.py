"""Privacy and distinguishability metrics."""

from __future__ import annotations

from collections.abc import Sequence
import math


def _same_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError(f"length mismatch: {len(left)} != {len(right)}")
    if not left:
        raise ValueError("metric requires at least one observation")


def accuracy(labels: Sequence[int | bool], predictions: Sequence[int | bool]) -> float:
    _same_length(labels, predictions)
    return sum(int(int(label) == int(prediction)) for label, prediction in zip(labels, predictions)) / len(labels)


def accuracy_above_prior(
    labels: Sequence[int | bool], predictions: Sequence[int | bool]
) -> float:
    _same_length(labels, predictions)
    ones = sum(int(label) for label in labels)
    prior = max(ones, len(labels) - ones) / len(labels)
    return accuracy(labels, predictions) - prior


def binary_auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Compute ROC AUC with pairwise tie handling and no third-party package.

    A constant distinguisher has AUC 0.5.  A single-class cohort has no
    defined AUC and returns NaN so callers cannot mistake it for evidence.
    """

    _same_length(labels, scores)
    positives = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return math.nan
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable += 1.0
            elif positive == negative:
                favorable += 0.5
    return favorable / (len(positives) * len(negatives))


def transcript_exact_distance(left: bytes | str, right: bytes | str) -> int:
    """Tier-1 exact distance: zero iff canonical transcript bytes match."""

    left_bytes = left.encode("utf-8") if isinstance(left, str) else left
    right_bytes = right.encode("utf-8") if isinstance(right, str) else right
    shared = sum(a != b for a, b in zip(left_bytes, right_bytes))
    return shared + abs(len(left_bytes) - len(right_bytes))

