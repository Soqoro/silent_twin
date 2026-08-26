"""Held-out transcript classifier with explicit training provenance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from silenttwin.attackers.base import (
    Attacker,
    AttackerTelemetry,
    Prediction,
    PublicTrialContext,
    TranscriptSnapshot,
)
from silenttwin.schemas import AgentVisibleTranscript, canonical_json, stable_digest


def _snapshot(value: TranscriptSnapshot | AgentVisibleTranscript) -> TranscriptSnapshot:
    if isinstance(value, TranscriptSnapshot):
        return value
    if isinstance(value, AgentVisibleTranscript):
        return TranscriptSnapshot.from_transcript(value)
    raise TypeError("classifier transcripts must be immutable snapshots or visible transcripts")


def transcript_features(
    transcript: TranscriptSnapshot | AgentVisibleTranscript,
) -> tuple[str, ...]:
    """Public-only categorical event features."""

    snapshot = _snapshot(transcript)
    return tuple(
        canonical_json(
            {
                "event_type": event.get("event_type", event.get("event")),
                "action_id": event.get("action_id"),
                "status": event.get("status"),
                "error_code": (
                    event.get("error", {}).get("code")
                    if isinstance(event.get("error"), Mapping)
                    else None
                ),
                "timing_bucket": event.get("timing_bucket", event.get("latency_bucket")),
                "termination": event.get("termination"),
                "callback": event.get("callback"),
                "visible_memory": event.get("visible_memory"),
            }
        )
        for event in snapshot
    )


@dataclass(frozen=True, slots=True)
class TrainingExample:
    transcript: TranscriptSnapshot
    label: str
    template_id: str
    runtime_configuration_hash: str

    def __post_init__(self) -> None:
        if self.label not in {"theta0", "theta1"}:
            raise ValueError("training labels must be theta0 or theta1")
        if not self.template_id or not self.runtime_configuration_hash:
            raise ValueError("training example provenance must be non-empty")


@dataclass(frozen=True, slots=True)
class TrainingProvenance:
    dataset_revision: str
    split: str
    template_ids: tuple[str, ...]
    runtime_configuration_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("black-box classifiers may only fit the training split")
        if not self.dataset_revision:
            raise ValueError("dataset revision must be recorded")
        object.__setattr__(self, "template_ids", tuple(sorted(set(self.template_ids))))
        object.__setattr__(
            self,
            "runtime_configuration_hashes",
            tuple(sorted(set(self.runtime_configuration_hashes))),
        )

    @property
    def provenance_hash(self) -> str:
        return stable_digest(
            {
                "dataset_revision": self.dataset_revision,
                "split": self.split,
                "template_ids": self.template_ids,
                "runtime_configuration_hashes": self.runtime_configuration_hashes,
            }
        )


class BlackBoxAttacker(Attacker):
    """Nearest-trajectory classifier fitted only on declared training data."""

    name = "black_box"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._examples: tuple[tuple[tuple[str, ...], str], ...] = ()
        self.training_provenance: TrainingProvenance | None = None

    def fit(
        self,
        examples: Iterable[TrainingExample],
        *,
        provenance: TrainingProvenance,
    ) -> "BlackBoxAttacker":
        materialized = tuple(examples)
        if not materialized:
            raise ValueError("black-box training requires at least one example")
        template_ids = {example.template_id for example in materialized}
        runtime_hashes = {example.runtime_configuration_hash for example in materialized}
        if template_ids != set(provenance.template_ids):
            raise ValueError("training template IDs do not match declared provenance")
        if runtime_hashes != set(provenance.runtime_configuration_hashes):
            raise ValueError("training runtime configurations do not match declared provenance")
        self._examples = tuple(
            (transcript_features(example.transcript), example.label) for example in materialized
        )
        self.training_provenance = provenance
        return self

    train = fit

    @staticmethod
    def _distance(left: Sequence[str], right: Sequence[str]) -> int:
        shared = min(len(left), len(right))
        return sum(left[index] != right[index] for index in range(shared)) + abs(len(left) - len(right))

    def predict_hidden_state(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction:
        if not self._examples or self.training_provenance is None:
            return Prediction.invalid("black_box_not_fitted", prior=self.initial_prior)
        if context.dataset_revision != self.training_provenance.dataset_revision:
            return Prediction.invalid("black_box_dataset_revision_mismatch", prior=self.initial_prior)
        if context.template_id in set(self.training_provenance.template_ids):
            return Prediction.invalid("black_box_template_split_leakage", prior=self.initial_prior)
        if context.dataset_split == "train":
            return Prediction.invalid("black_box_evaluation_on_training_split", prior=self.initial_prior)

        features = transcript_features(transcript)
        distances = [(self._distance(features, known), label) for known, label in self._examples]
        best_distance = min(distance for distance, _ in distances)
        counts = Counter(label for distance, label in distances if distance == best_distance)
        theta0_count = counts.get("theta0", 0)
        theta1_count = counts.get("theta1", 0)
        total = theta0_count + theta1_count
        posterior = {
            "theta0": theta0_count / total,
            "theta1": theta1_count / total,
        }
        prediction = "theta1" if posterior["theta1"] > posterior["theta0"] else "theta0"
        telemetry = AttackerTelemetry(
            metadata={
                "classifier": "nearest_public_trajectory",
                "training_provenance_hash": self.training_provenance.provenance_hash,
                "nearest_distance": best_distance,
                "nearest_count": total,
            }
        )
        return Prediction(prediction=prediction, posterior=posterior, telemetry=telemetry)

    online_prediction = predict_hidden_state


TranscriptClassifierAttacker = BlackBoxAttacker


__all__ = [
    "BlackBoxAttacker",
    "TrainingExample",
    "TrainingProvenance",
    "TranscriptClassifierAttacker",
    "transcript_features",
]
