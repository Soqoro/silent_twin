"""Useful-work and commit-integrity metrics."""

from __future__ import annotations

from collections.abc import Iterable


def salvage_rate(numerator: int, denominator: int) -> float:
    if denominator < 0 or numerator < 0:
        raise ValueError("salvage counts cannot be negative")
    if numerator > denominator:
        raise ValueError("salvage numerator exceeds denominator")
    # A workflow with no eligible effects has vacuous, rather than failed,
    # preservation.  The raw counts remain in every result record.
    return 1.0 if denominator == 0 else numerator / denominator


def indicator_rate(values: Iterable[int | bool]) -> float:
    materialized = [int(value) for value in values]
    if not materialized:
        raise ValueError("metric requires at least one observation")
    return sum(materialized) / len(materialized)

