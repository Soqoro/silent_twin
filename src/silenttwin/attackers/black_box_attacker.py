"""Simple transcript-only learned attacker for Tier-1 experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from silenttwin.attackers.base import Attacker
from silenttwin.schemas import AgentVisibleTranscript


def transcript_features(transcript: AgentVisibleTranscript) -> tuple[str, ...]:
    return tuple(
        f"{event.event_type}|{event.action_id}|{event.status}|"
        f"{event.error.get('code') if event.error else ''}|{event.timing_bucket}|{event.termination or ''}"
        for event in transcript
    )


class BlackBoxAttacker(Attacker):
    name = "black_box"

    def __init__(self, *, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        self._examples: list[tuple[tuple[str, ...], str]] = []

    def fit(
        self,
        examples: Iterable[tuple[AgentVisibleTranscript, str] | dict[str, Any]],
    ) -> "BlackBoxAttacker":
        self._examples.clear()
        for example in examples:
            if isinstance(example, dict):
                transcript = example["transcript"]
                label = example.get("label", example.get("private_state"))
            else:
                transcript, label = example
            if not isinstance(transcript, AgentVisibleTranscript):
                raise TypeError("black-box training transcripts must be AgentVisibleTranscript instances")
            if label not in {"theta0", "theta1"}:
                raise ValueError("black-box labels must be theta0 or theta1")
            self._examples.append((transcript_features(transcript), label))
        return self

    train = fit

    @staticmethod
    def _distance(left: Sequence[str], right: Sequence[str]) -> int:
        shared = min(len(left), len(right))
        return sum(left[index] != right[index] for index in range(shared)) + abs(len(left) - len(right))

    def predict_state(
        self,
        transcript: AgentVisibleTranscript,
        *,
        world_pair=None,
        runtime=None,
        actions: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        features = transcript_features(transcript)
        if self._examples:
            distances = [(self._distance(features, known), label) for known, label in self._examples]
            best_distance = min(distance for distance, _ in distances)
            counts = Counter(label for distance, label in distances if distance == best_distance)
            theta0_count = counts.get("theta0", 0)
            theta1_count = counts.get("theta1", 0)
            prediction = "theta1" if theta1_count > theta0_count else "theta0"
            total = theta0_count + theta1_count
            self.posterior = {
                "theta0": theta0_count / total,
                "theta1": theta1_count / total,
            }
            return prediction

        # Deterministic cold-start proxy learned by convention from held-out
        # synthetic training data: cloud-link success indicates theta0.
        cloud_events = [event for event in transcript if event.action_id == "probe_cloud_link"]
        if cloud_events:
            last = cloud_events[-1]
            allowed_like = last.status in {"ok", "tracked"} and last.error is None
            self.posterior = (
                {"theta0": 1.0, "theta1": 0.0}
                if allowed_like
                else {"theta0": 0.0, "theta1": 1.0}
            )
        else:
            self.posterior = {"theta0": 0.5, "theta1": 0.5}
        return self._posterior_argmax()


TranscriptClassifierAttacker = BlackBoxAttacker

__all__ = ["BlackBoxAttacker", "TranscriptClassifierAttacker", "transcript_features"]
