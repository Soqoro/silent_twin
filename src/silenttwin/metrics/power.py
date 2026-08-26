"""Deterministic simulation power analysis for paired binary outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Mapping, Sequence

from silenttwin.schemas import stable_digest


POWER_ANALYSIS_SCHEMA = "silenttwin.paired_power.v1"
SAMPLE_SIZE_FREEZE_SCHEMA = "silenttwin.sample_size_freeze.v2"


def paired_discordance_rate(
    target: Sequence[int | bool], reference: Sequence[int | bool]
) -> float:
    if len(target) != len(reference):
        raise ValueError("paired outcomes have different sizes")
    if not target:
        raise ValueError("paired discordance requires at least one pair")
    normalized = [(int(left), int(right)) for left, right in zip(target, reference)]
    if any(left not in {0, 1} or right not in {0, 1} for left, right in normalized):
        raise ValueError("paired outcomes must be binary")
    return sum(left != right for left, right in normalized) / len(normalized)


def exact_mcnemar_p_value(target_only: int, reference_only: int) -> float:
    """Two-sided exact conditional binomial p-value for paired discordance."""

    if target_only < 0 or reference_only < 0:
        raise ValueError("discordant counts cannot be negative")
    discordant = target_only + reference_only
    if discordant == 0:
        return 1.0
    tail = min(target_only, reference_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


@dataclass(frozen=True)
class PowerEstimate:
    sample_size: int
    effect: float
    discordance: float
    alpha: float
    simulations: int
    seed: int
    power: float

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": POWER_ANALYSIS_SCHEMA, **asdict(self)}


def simulate_paired_binary_power(
    sample_size: int,
    *,
    effect: float,
    discordance: float,
    alpha: float = 0.05,
    simulations: int = 10000,
    seed: int = 0,
) -> PowerEstimate:
    """Estimate power under p10-p01=`effect`, p10+p01=`discordance`."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not -1.0 <= effect <= 1.0:
        raise ValueError("effect must lie in [-1, 1]")
    if not 0.0 <= discordance <= 1.0:
        raise ValueError("discordance must lie in [0, 1]")
    if abs(effect) > discordance + 1e-15:
        raise ValueError("absolute effect cannot exceed discordance")
    target_only_probability = (discordance + effect) / 2.0
    reference_only_probability = (discordance - effect) / 2.0
    rng = random.Random(seed)
    rejected = 0
    for _ in range(simulations):
        target_only = 0
        reference_only = 0
        for _ in range(sample_size):
            draw = rng.random()
            if draw < target_only_probability:
                target_only += 1
            elif draw < target_only_probability + reference_only_probability:
                reference_only += 1
        rejected += exact_mcnemar_p_value(target_only, reference_only) <= alpha
    return PowerEstimate(
        sample_size=sample_size,
        effect=float(effect),
        discordance=float(discordance),
        alpha=float(alpha),
        simulations=simulations,
        seed=seed,
        power=rejected / simulations,
    )


@dataclass(frozen=True)
class RequiredSampleSize:
    target_power: float
    selected_sample_size: int | None
    estimates: tuple[PowerEstimate, ...]

    @property
    def achieved(self) -> bool:
        return self.selected_sample_size is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POWER_ANALYSIS_SCHEMA,
            "target_power": self.target_power,
            "selected_sample_size": self.selected_sample_size,
            "achieved": self.achieved,
            "estimates": [item.to_dict() for item in self.estimates],
        }


def find_required_sample_size(
    candidates: Sequence[int],
    *,
    effect: float,
    discordance: float,
    target_power: float = 0.8,
    alpha: float = 0.05,
    simulations: int = 10000,
    seed: int = 0,
) -> RequiredSampleSize:
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must lie strictly between zero and one")
    ordered = tuple(sorted(set(candidates)))
    if not ordered or any(item <= 0 for item in ordered):
        raise ValueError("candidates must contain positive sample sizes")
    estimates: list[PowerEstimate] = []
    selected: int | None = None
    for index, sample_size in enumerate(ordered):
        estimate = simulate_paired_binary_power(
            sample_size,
            effect=effect,
            discordance=discordance,
            alpha=alpha,
            simulations=simulations,
            seed=seed + 104729 * index,
        )
        estimates.append(estimate)
        if selected is None and estimate.power >= target_power:
            selected = sample_size
    return RequiredSampleSize(target_power, selected, tuple(estimates))


def make_sample_size_freeze(
    *,
    experiment_id: str,
    dataset_revision: str,
    development_manifest_hash: str,
    contrast_id: str,
    sample_size: int,
    power_estimate: PowerEstimate,
    frozen_before_test: bool = True,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size != power_estimate.sample_size:
        raise ValueError("sample_size does not match the selected power estimate")
    if experiment_id not in {"e1", "e2"}:
        raise ValueError("sample-size freeze experiment_id must be e1 or e2")
    if not contrast_id.startswith(f"{experiment_id}_"):
        raise ValueError("sample-size freeze contrast belongs to another experiment")
    if not dataset_revision or not development_manifest_hash or not contrast_id:
        raise ValueError("freeze identifiers must be non-empty")
    if not frozen_before_test:
        raise ValueError("a valid freeze must be created before test inspection")
    payload = {
        "schema_version": SAMPLE_SIZE_FREEZE_SCHEMA,
        "status": "frozen",
        "experiment_id": experiment_id,
        "dataset_revision": dataset_revision,
        "development_manifest_hash": development_manifest_hash,
        "contrast_id": contrast_id,
        "sample_size": sample_size,
        "power_estimate": power_estimate.to_dict(),
        "frozen_before_test": True,
        "test_results_inspected": False,
    }
    return {**payload, "freeze_hash": stable_digest(payload)}


def validate_sample_size_freeze(
    freeze: Mapping[str, Any],
    *,
    experiment_id: str,
    dataset_revision: str,
    contrast_id: str,
    development_manifest_hash: str | None = None,
) -> int:
    payload = dict(freeze)
    recorded_hash = payload.pop("freeze_hash", None)
    if recorded_hash != stable_digest(payload):
        raise ValueError("sample-size freeze hash is missing or invalid")
    if payload.get("schema_version") != SAMPLE_SIZE_FREEZE_SCHEMA:
        raise ValueError("unsupported sample-size freeze schema")
    if payload.get("status") != "frozen":
        raise ValueError("held-out sample size is not frozen")
    if payload.get("experiment_id") != experiment_id:
        raise ValueError("sample-size freeze belongs to a different experiment")
    if payload.get("dataset_revision") != dataset_revision:
        raise ValueError("sample-size freeze uses a different dataset revision")
    if payload.get("contrast_id") != contrast_id:
        raise ValueError("sample-size freeze uses a different primary contrast")
    if development_manifest_hash is not None and payload.get(
        "development_manifest_hash"
    ) != development_manifest_hash:
        raise ValueError("sample-size freeze uses different development evidence")
    if payload.get("frozen_before_test") is not True or payload.get(
        "test_results_inspected"
    ) is not False:
        raise ValueError("sample-size freeze was not made before held-out inspection")
    sample_size = payload.get("sample_size")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0:
        raise ValueError("sample-size freeze has no positive integer sample size")
    return sample_size


__all__ = [
    "POWER_ANALYSIS_SCHEMA",
    "SAMPLE_SIZE_FREEZE_SCHEMA",
    "PowerEstimate",
    "RequiredSampleSize",
    "exact_mcnemar_p_value",
    "find_required_sample_size",
    "make_sample_size_freeze",
    "paired_discordance_rate",
    "simulate_paired_binary_power",
    "validate_sample_size_freeze",
]
