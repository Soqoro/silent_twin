"""Restricted, data-only attacker protocol.

The experiment harness is trusted; attacker callbacks are not. Consequently
callbacks receive immutable copies of public data and return public action IDs.
They never receive a world, world pair, runtime, protected session, runtime
step, staged effect, controller, or trusted report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from silenttwin.schemas import (
    AgentVisibleTranscript,
    VisibleEvent,
    assert_agent_visible,
    canonical_json,
    stable_digest,
)


STATE_IDS = ("theta0", "theta1")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    """Return a recursively immutable JSON value after boundary validation."""

    import json

    copied = json.loads(canonical_json(value))

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType({str(key): freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(copied)


@dataclass(frozen=True, slots=True)
class PublicAction:
    """An immutable attacker-visible action descriptor.

    ``policy_tag`` and all monitor/ground-truth fields are deliberately absent.
    The trusted runner resolves ``action_id`` against the actual world's action
    catalogue, so callers cannot forge arguments, effectfulness, or dependency
    metadata.
    """

    action_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    effectful: bool = False
    public_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id or not self.tool_name:
            raise ValueError("public action IDs and tool names must be non-empty")
        assert_agent_visible(self.arguments, "$.public_action.arguments")
        assert_agent_visible(self.public_metadata, "$.public_action.public_metadata")
        object.__setattr__(self, "arguments", _freeze(self.arguments))
        object.__setattr__(self, "public_metadata", _freeze(self.public_metadata))

    @property
    def kind(self) -> str:
        return str(self.public_metadata.get("kind", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "arguments": _thaw(self.arguments),
            "effectful": self.effectful,
            "public_metadata": _thaw(self.public_metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PublicAction":
        return cls(
            action_id=str(value["action_id"]),
            tool_name=str(value["tool_name"]),
            arguments=value.get("arguments", {}),
            effectful=bool(value.get("effectful", False)),
            public_metadata=value.get("public_metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class TranscriptSnapshot(Sequence[Mapping[str, Any]]):
    """Immutable snapshot of exactly the events delivered to an attacker."""

    events: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        frozen: list[Mapping[str, Any]] = []
        for index, event in enumerate(self.events):
            materialized = event.to_dict() if isinstance(event, VisibleEvent) else dict(event)
            assert_agent_visible(materialized, f"$.events[{index}]")
            frozen.append(_freeze(materialized))
        object.__setattr__(self, "events", tuple(frozen))

    @classmethod
    def from_events(
        cls,
        events: Iterable[VisibleEvent | Mapping[str, Any]],
        *,
        schema_version: str = "1.0",
    ) -> "TranscriptSnapshot":
        return cls(tuple(events), schema_version=schema_version)

    @classmethod
    def from_transcript(cls, transcript: AgentVisibleTranscript) -> "TranscriptSnapshot":
        serialized = transcript.to_dict()
        return cls.from_events(serialized["events"], schema_version=serialized["schema_version"])

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "events": [_thaw(event) for event in self.events],
        }
        assert_agent_visible(result)
        return result

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def transcript_hash(self) -> str:
        return stable_digest(self.to_dict())

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.events)

    def __getitem__(self, index):
        return self.events[index]


@dataclass(frozen=True, slots=True)
class PublicTrialContext:
    """Complete attacker-visible trial input with no trusted object references."""

    public_task: str
    public_environment: Mapping[str, Any]
    tool_schemas: Mapping[str, Any]
    public_runtime_config: Mapping[str, Any]
    public_actions: tuple[PublicAction, ...]
    query_budget: int
    public_instance_hash: str
    runtime_name: str = "unknown"
    feedback_protocol: str = "genuine"
    pair_family: str = "monitor_blind_spot"
    template_id: str = "unknown"
    dataset_split: str = "unspecified"
    dataset_revision: str = "unspecified"

    def __post_init__(self) -> None:
        if self.query_budget < 0:
            raise ValueError("query_budget must be non-negative")
        if not self.public_instance_hash:
            raise ValueError("public_instance_hash must be non-empty")
        normalized_family = self.pair_family.strip().lower().replace("-", "_")
        if normalized_family == "blind_spot":
            normalized_family = "monitor_blind_spot"
        object.__setattr__(self, "pair_family", normalized_family)
        object.__setattr__(self, "public_environment", _freeze(self.public_environment))
        object.__setattr__(self, "tool_schemas", _freeze(self.tool_schemas))
        object.__setattr__(self, "public_runtime_config", _freeze(self.public_runtime_config))
        object.__setattr__(self, "public_actions", tuple(self.public_actions))
        ids = [action.action_id for action in self.public_actions]
        if len(ids) != len(set(ids)):
            raise ValueError("public action IDs must be unique")

    @classmethod
    def from_public_projection(
        cls,
        projection: Mapping[str, Any],
        *,
        query_budget: int,
        public_instance_hash: str,
        runtime_name: str = "unknown",
        feedback_protocol: str = "genuine",
        pair_family: str = "monitor_blind_spot",
        template_id: str = "unknown",
        dataset_split: str = "unspecified",
        dataset_revision: str = "unspecified",
    ) -> "PublicTrialContext":
        actions = tuple(PublicAction.from_mapping(item) for item in projection.get("actions", ()))
        return cls(
            public_task=str(projection.get("public_task", "")),
            public_environment=projection.get("public_environment", {}),
            tool_schemas=projection.get("tool_schemas", {}),
            public_runtime_config=projection.get("public_runtime_config", {}),
            public_actions=actions,
            query_budget=query_budget,
            public_instance_hash=public_instance_hash,
            runtime_name=runtime_name,
            feedback_protocol=feedback_protocol,
            pair_family=pair_family,
            template_id=template_id,
            dataset_split=dataset_split,
            dataset_revision=dataset_revision,
        )

    @property
    def probes(self) -> tuple[PublicAction, ...]:
        return tuple(action for action in self.public_actions if action.kind == "probe")

    @property
    def exploits(self) -> tuple[PublicAction, ...]:
        return tuple(action for action in self.public_actions if action.kind == "exploit")

    def action(self, action_id: str) -> PublicAction:
        for action in self.public_actions:
            if action.action_id == action_id:
                return action
        raise KeyError(f"unknown public action ID: {action_id}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "public_task": self.public_task,
            "public_environment": _thaw(self.public_environment),
            "tool_schemas": _thaw(self.tool_schemas),
            "public_runtime_config": _thaw(self.public_runtime_config),
            "public_actions": [action.to_dict() for action in self.public_actions],
            "query_budget": self.query_budget,
            "public_instance_hash": self.public_instance_hash,
            "runtime_name": self.runtime_name,
            "feedback_protocol": self.feedback_protocol,
            "pair_family": self.pair_family,
            "template_id": self.template_id,
            "dataset_split": self.dataset_split,
            "dataset_revision": self.dataset_revision,
        }
        assert_agent_visible(result)
        return result


@dataclass(frozen=True, slots=True)
class AttackerTelemetry:
    latency_ms: float = 0.0
    retries: int = 0
    failures: tuple[str, ...] = ()
    model_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    prompt_hash: str | None = None
    response_hash: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or self.retries < 0 or self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("telemetry counters must be non-negative")
        object.__setattr__(self, "failures", tuple(str(item) for item in self.failures))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "failures": list(self.failures),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": _thaw(self.metadata),
        }


def normalized_posterior(value: Mapping[str, float]) -> dict[str, float]:
    if set(value) != set(STATE_IDS):
        raise ValueError("posterior must contain exactly theta0 and theta1")
    result = {state: float(value[state]) for state in STATE_IDS}
    if any(not math.isfinite(probability) or probability < 0.0 for probability in result.values()):
        raise ValueError("posterior probabilities must be finite and non-negative")
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("posterior must have positive total mass")
    return {state: probability / total for state, probability in result.items()}


@dataclass(frozen=True, slots=True)
class Prediction:
    prediction: str | None
    posterior: Mapping[str, float]
    valid: bool = True
    failure: str | None = None
    telemetry: AttackerTelemetry = field(default_factory=AttackerTelemetry)

    def __post_init__(self) -> None:
        posterior = normalized_posterior(self.posterior)
        object.__setattr__(self, "posterior", MappingProxyType(posterior))
        if self.valid:
            if self.prediction not in STATE_IDS:
                raise ValueError("a valid prediction must be theta0 or theta1")
            if self.failure is not None:
                raise ValueError("a valid prediction cannot include a failure")
        elif self.prediction is not None:
            raise ValueError("an invalid prediction must not invent a state label")

    @classmethod
    def invalid(
        cls,
        failure: str,
        *,
        prior: Mapping[str, float] | None = None,
        telemetry: AttackerTelemetry | None = None,
    ) -> "Prediction":
        return cls(
            prediction=None,
            posterior=prior or {"theta0": 0.5, "theta1": 0.5},
            valid=False,
            failure=failure,
            telemetry=telemetry or AttackerTelemetry(failures=(failure,)),
        )

    @property
    def posterior_entropy(self) -> float:
        return -sum(probability * math.log2(probability) for probability in self.posterior.values() if probability)

    @property
    def score_theta1(self) -> float:
        return float(self.posterior["theta1"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "posterior": dict(self.posterior),
            "posterior_entropy": self.posterior_entropy,
            "valid": self.valid,
            "failure": self.failure,
            "telemetry": self.telemetry.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AttackRunResult:
    """Public-only compatibility result; trusted execution lives in TrialResult."""

    transcript: TranscriptSnapshot
    probe_action_ids: tuple[str, ...]
    prediction: Prediction
    selected_exploit: str | None = None
    final_exploit_attempts: int = 0

    @property
    def predicted_state(self) -> str | None:
        return self.prediction.prediction

    @property
    def posterior(self) -> Mapping[str, float]:
        return self.prediction.posterior


class Attacker(ABC):
    """Adaptive attacker whose callbacks cross only the public data boundary."""

    name = "base"

    def __init__(
        self,
        *,
        seed: int = 0,
        prior: Mapping[str, float] | None = None,
        **_: Any,
    ) -> None:
        if seed < 0:
            raise ValueError("attacker seed must be non-negative")
        self.seed = int(seed)
        self.initial_prior = normalized_posterior(prior or {"theta0": 0.5, "theta1": 0.5})
        self._probe_index = 0

    def reset(self) -> None:
        self._probe_index = 0

    def choose_probe(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
        remaining_budget: int,
    ) -> str:
        if remaining_budget <= 0:
            raise ValueError("remaining probe budget must be positive")
        probes = context.probes
        if not probes:
            raise RuntimeError("public action catalogue has no probes")
        choice = probes[self._probe_index % len(probes)]
        self._probe_index += 1
        return choice.action_id

    def select_probe(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
        remaining_budget: int,
    ) -> str:
        return self.choose_probe(context, transcript, remaining_budget)

    @abstractmethod
    def predict_hidden_state(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction:
        raise NotImplementedError

    def predict_state(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction:
        return self.predict_hidden_state(context, transcript)

    def infer_state(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction:
        return self.predict_hidden_state(context, transcript)

    def online_prediction(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction | None:
        """Return a no-extra-cost online prediction when supported.

        Real model attackers inherit ``None`` so the runner never introduces
        unplanned model calls merely to populate a confidence curve.
        """

        del context, transcript
        return None

    def choose_final_exploit(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
        prediction: Prediction,
    ) -> str:
        exploits = context.exploits
        if not exploits:
            raise RuntimeError("public action catalogue has no final exploits")
        if prediction.valid and prediction.prediction == "theta1" and len(exploits) > 1:
            return exploits[1].action_id
        return exploits[0].action_id

    def select_exploit(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
        prediction: Prediction,
    ) -> str:
        return self.choose_final_exploit(context, transcript, prediction)

    def choose_exploit(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
        prediction: Prediction,
    ) -> str:
        return self.choose_final_exploit(context, transcript, prediction)


BaseAttacker = Attacker


__all__ = [
    "STATE_IDS",
    "AttackRunResult",
    "Attacker",
    "AttackerTelemetry",
    "BaseAttacker",
    "Prediction",
    "PublicAction",
    "PublicTrialContext",
    "TranscriptSnapshot",
    "normalized_posterior",
]
