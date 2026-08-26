"""Condition-independent RNG schedule for AgentDojo experiments."""

from __future__ import annotations

from dataclasses import dataclass

from silenttwin.schemas import stable_digest


def _seed(*parts: object) -> int:
    return int(stable_digest(list(parts))[:16], 16)


@dataclass(frozen=True, slots=True)
class SeedSchedule:
    scenario_id: str
    immutable_model_revision: str
    replicate: int

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not self.immutable_model_revision:
            raise ValueError("immutable_model_revision must be non-empty")
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")

    def probe(self, index: int) -> int:
        if index < 0:
            raise ValueError("probe index must be non-negative")
        return _seed(
            self.scenario_id,
            self.immutable_model_revision,
            self.replicate,
            "probe",
            index,
        )

    def prediction(self) -> int:
        return _seed(
            self.scenario_id,
            self.immutable_model_revision,
            self.replicate,
            "prediction",
        )

    def final(self) -> int:
        return _seed(
            self.scenario_id,
            self.immutable_model_revision,
            self.replicate,
            "final",
        )

    def postselection(self) -> int:
        """Condition/Q/theta-independent retained-callback decoding seed."""

        return _seed(
            self.scenario_id,
            self.immutable_model_revision,
            self.replicate,
            "postselection",
        )

    def ecological_tool_filter(self) -> int:
        """Frozen local-transport seed for the adapted released tool filter."""

        return _seed(
            self.scenario_id,
            self.immutable_model_revision,
            self.replicate,
            "ecological_tool_filter",
        )

    def random_final(self) -> int:
        # Deliberately omits the model revision: this is the frozen,
        # target-independent random-choice control rather than model decoding.
        return _seed(self.scenario_id, self.replicate, "random_final")

    def environment(self) -> int:
        return _seed(self.scenario_id, self.replicate, "environment")

    def monitor(self, index: int = 0) -> int:
        if index < 0:
            raise ValueError("monitor index must be non-negative")
        return _seed(self.scenario_id, self.replicate, "monitor", index)

    def final_monitor(self) -> int:
        """Condition/Q/model-independent action-monitor settlement seed."""

        return _seed(self.scenario_id, self.replicate, "monitor", "final")

    def ecological_turn(self, index: int) -> int:
        if index < 0:
            raise ValueError("turn index must be non-negative")
        return _seed(
            self.scenario_id,
            self.immutable_model_revision,
            self.replicate,
            "ecological_tool_turn",
            index,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": "sha256-first-u64-v1",
            "scenario_id": self.scenario_id,
            "immutable_model_revision": self.immutable_model_revision,
            "replicate": self.replicate,
            "forbidden_factors": [
                "query_budget",
                "runtime",
                "condition",
                "theta",
                "donor_state",
                "monitor_verdict",
            ],
        }


__all__ = ["SeedSchedule"]
