"""Shared, dependency-free schemas for the SilentTwin experiments.

The most important boundary in this module is the one between
``AgentVisibleTranscript`` and private evaluation records.  Runtime code may
append only :class:`VisibleEvent` instances to a transcript; monitor verdicts,
true-policy labels, and the hidden world state live in separate records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "1.0"


class MonitorDecision(str, Enum):
    """A monitor's (fallible) verdict."""

    ALLOW = "allow"
    BLOCK = "block"

    @classmethod
    def coerce(cls, value: "MonitorDecision | str | bool") -> "MonitorDecision":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.ALLOW if value else cls.BLOCK
        normalized = str(value).strip().lower()
        if normalized in {"allow", "allowed", "permit", "permitted", "true", "1"}:
            return cls.ALLOW
        if normalized in {"block", "blocked", "deny", "denied", "false", "0"}:
            return cls.BLOCK
        raise ValueError(f"unknown monitor decision: {value!r}")


class TruePolicyLabel(str, Enum):
    """Ground-truth safety label, independent of monitor behavior."""

    PERMITTED = "permitted"
    PROHIBITED = "prohibited"

    @classmethod
    def coerce(cls, value: "TruePolicyLabel | str | bool") -> "TruePolicyLabel":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.PERMITTED if value else cls.PROHIBITED
        normalized = str(value).strip().lower()
        if normalized in {"allow", "allowed", "permit", "permitted", "safe", "true", "1"}:
            return cls.PERMITTED
        if normalized in {"block", "blocked", "deny", "denied", "unsafe", "prohibited", "false", "0"}:
            return cls.PROHIBITED
        raise ValueError(f"unknown true-policy label: {value!r}")


# Compatibility aliases used by some downstream experiment code.
PolicyLabel = TruePolicyLabel
Decision = MonitorDecision


class EffectStatus(str, Enum):
    PROPOSED = "proposed"
    ADMISSIBLE = "admissible"
    REJECTED = "rejected"
    COMMITTED = "committed"
    DISCARDED = "discarded"


class SessionState(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"
    COMMITTED = "committed"
    ABORTED = "aborted"


def _jsonable(value: Any) -> Any:
    """Convert common immutable/domain values into canonical JSON values."""

    if isinstance(value, Enum):
        return value.value
    # Domain objects with an explicit public representation take precedence
    # over generic dataclass traversal (which may include private/internal
    # fields such as transcript sealing state).
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot canonically serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize *value* deterministically for comparisons and hashing."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


canonical_hash = stable_digest


def _copy_public(value: Any) -> Any:
    """Copy JSON-like data so callers cannot mutate a stored transcript."""

    # Round-tripping through the canonical representation also rejects opaque
    # Python objects that could later serialize in surprising ways.
    return json.loads(canonical_json(value))


PRIVATE_FIELD_NAMES = frozenset(
    {
        "theta",
        "private_state",
        "hidden_state",
        "actual_state",
        "actual_hidden_state",
        "donor_state",
        "source_state",
        "feedback_source_state",
        "monitor_decision",
        "monitor_verdict",
        "true_policy_label",
        "ground_truth_label",
        "policy_epoch",
        "admissible",
        "commit_token",
        "staged_effect",
        "staged_effects",
        "committed_effects",
        "private_records",
        "controller",
        "trusted_report",
    }
)


def assert_agent_visible(value: Any, path: str = "$") -> None:
    """Reject mappings that try to cross the private/public boundary.

    This is intentionally key-based: ordinary tool output may contain words
    such as ``"allowed"`` without being a guardrail verdict, while a field
    explicitly named ``monitor_decision`` is never acceptable.
    """

    namespace = getattr(value, "__silenttwin_namespace__", None)
    if namespace == "trusted":
        raise ValueError(f"trusted object cannot enter the visible transcript at {path}")
    if namespace == "public":
        serializer = getattr(value, "to_public_dict", None)
        if not callable(serializer):
            raise ValueError(f"public namespace object at {path} has no serializer")
        encoded = serializer()
        if not is_agentdojo_public_envelope(encoded):
            raise ValueError(f"invalid AgentDojo public envelope at {path}")
        return
    if isinstance(value, Mapping) and isinstance(value.get("schema"), str) and str(
        value.get("schema")
    ).startswith("silenttwin.agentdojo.trusted"):
        raise ValueError(f"trusted envelope cannot enter the visible transcript at {path}")
    if isinstance(value, Mapping) and is_agentdojo_public_envelope(value):
        # AgentDojo application data is opaque within its explicit typed
        # namespace. Keys such as world/session are ordinary domain payload.
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in PRIVATE_FIELD_NAMES or normalized.startswith("private_"):
                raise ValueError(f"private field {key!r} cannot enter the visible transcript at {path}")
            assert_agent_visible(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_agent_visible(item, f"{path}[{index}]")


def is_agentdojo_public_envelope(value: Any) -> bool:
    """Recognize the durable representation of a typed public envelope."""

    if not isinstance(value, Mapping) or set(value) != {"schema", "value"}:
        return False
    schema = value.get("schema")
    return (
        isinstance(schema, str)
        and schema.startswith("silenttwin.agentdojo.")
        and ".trusted" not in schema
    )


@dataclass(frozen=True)
class ActionProposal:
    """A public tool action proposed by the protected agent."""

    action_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    effectful: bool = False
    data_dependencies: tuple[str, ...] = ()
    control_dependencies: tuple[str, ...] = ()
    atomic_group: str | None = None
    policy_tag: str = "safe"
    public_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not self.tool_name:
            raise ValueError("tool_name must be non-empty")
        object.__setattr__(self, "arguments", MappingProxyType(_copy_public(self.arguments)))
        object.__setattr__(self, "public_metadata", MappingProxyType(_copy_public(self.public_metadata)))
        object.__setattr__(self, "data_dependencies", tuple(dict.fromkeys(self.data_dependencies)))
        object.__setattr__(self, "control_dependencies", tuple(dict.fromkeys(self.control_dependencies)))
        dependencies = self.data_dependencies + self.control_dependencies
        if self.action_id in dependencies:
            raise ValueError(f"action {self.action_id!r} cannot depend on itself")
        assert_agent_visible(self.arguments)
        assert_agent_visible(self.public_metadata)

    @property
    def id(self) -> str:
        return self.action_id

    @property
    def payload(self) -> Any:
        return self.arguments.get("payload", self.arguments.get("content"))

    @property
    def destination(self) -> Any:
        return self.arguments.get("destination", self.arguments.get("recipient", self.arguments.get("path")))

    @property
    def payload_digest(self) -> str:
        return stable_digest(self.payload)

    @property
    def destination_digest(self) -> str:
        return stable_digest(self.destination)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "effectful": self.effectful,
            "data_dependencies": list(self.data_dependencies),
            "control_dependencies": list(self.control_dependencies),
            "atomic_group": self.atomic_group,
            "policy_tag": self.policy_tag,
            "public_metadata": dict(self.public_metadata),
        }


# The paper and early prototypes use both names.
ProposedAction = ActionProposal


@dataclass(frozen=True)
class VisibleEvent:
    """One observation available to the protected agent session."""

    event_type: str
    sequence: int
    action_id: str | None = None
    tool_name: str | None = None
    status: str = "ok"
    value: Any = None
    error: Mapping[str, Any] | None = None
    receipt: Mapping[str, Any] | None = None
    identifier: str | None = None
    timing_bucket: str = "fixed"
    retries: int = 0
    termination: str | None = None
    callback: Mapping[str, Any] | None = None
    visible_memory: Mapping[str, Any] | None = None
    settlement_message: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence must be non-negative")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        for name in ("value", "error", "receipt", "callback", "visible_memory", "settlement_message"):
            item = getattr(self, name)
            if item is not None:
                assert_agent_visible(item, f"$.{name}")
                object.__setattr__(self, name, _copy_public(item))

    @property
    def latency_bucket(self) -> str:
        return self.timing_bucket

    @property
    def retry_count(self) -> int:
        return self.retries

    def to_dict(self) -> dict[str, Any]:
        # Keep an explicit, stable schema.  Null fields are retained so two
        # adjacent worlds cannot differ because of conditional key omission.
        return {
            "event_type": self.event_type,
            "sequence": self.sequence,
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "value": self.value,
            "error": self.error,
            "receipt": self.receipt,
            "identifier": self.identifier,
            "timing_bucket": self.timing_bucket,
            "retries": self.retries,
            "termination": self.termination,
            "callback": self.callback,
            "visible_memory": self.visible_memory,
            "settlement_message": self.settlement_message,
        }


AgentVisibleObservation = VisibleEvent
TranscriptEvent = VisibleEvent


@dataclass
class AgentVisibleTranscript:
    """Append-only public transcript.

    The transcript has no generic metadata escape hatch.  This makes accidental
    inclusion of trusted evaluation data much harder than filtering it later.
    """

    events: list[VisibleEvent] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    _sealed: bool = field(default=False, init=False, repr=False)

    def append(self, event: VisibleEvent) -> VisibleEvent:
        if self._sealed:
            raise RuntimeError("agent-visible transcript is sealed")
        if not isinstance(event, VisibleEvent):
            raise TypeError("only VisibleEvent instances may enter an agent-visible transcript")
        expected = len(self.events)
        if event.sequence != expected:
            raise ValueError(f"event sequence {event.sequence} does not match next sequence {expected}")
        # Validate the fully materialized representation, not just selected
        # fields, as a defense against future schema additions.
        assert_agent_visible(event.to_dict())
        self.events.append(event)
        return event

    append_event = append

    def seal(self) -> None:
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "events": [event.to_dict() for event in self.events],
        }
        assert_agent_visible(result)
        return result

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    def serialize(self) -> str:
        return self.canonical_json()

    @property
    def transcript_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def hash(self) -> str:
        return self.transcript_hash

    def __iter__(self) -> Iterator[VisibleEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> VisibleEvent:
        return self.events[index]


@dataclass(frozen=True)
class PrivateEvaluationRecord:
    """Trusted-only record corresponding to an action."""

    action_id: str
    private_state: str
    monitor_decision: MonitorDecision
    true_policy_label: TruePolicyLabel
    policy_epoch: int
    effect_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "monitor_decision", MonitorDecision.coerce(self.monitor_decision))
        object.__setattr__(self, "true_policy_label", TruePolicyLabel.coerce(self.true_policy_label))
        object.__setattr__(self, "details", MappingProxyType(_copy_public(self.details)))

    def trusted_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass
class StagedEffect:
    """A proposed external mutation held outside the protected session."""

    effect_id: str
    payload_digest: str
    destination_digest: str
    monitor_decision: MonitorDecision
    true_policy_label: TruePolicyLabel
    policy_epoch: int = 0
    action_id: str | None = None
    tool_name: str | None = None
    data_dependencies: tuple[str, ...] = ()
    control_dependencies: tuple[str, ...] = ()
    atomic_group: str | None = None
    admissible: bool | None = None
    status: EffectStatus = EffectStatus.PROPOSED
    private_payload: Any = field(default=None, repr=False, compare=False)
    private_destination: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.effect_id:
            raise ValueError("effect_id must be non-empty")
        self.monitor_decision = MonitorDecision.coerce(self.monitor_decision)
        self.true_policy_label = TruePolicyLabel.coerce(self.true_policy_label)
        self.data_dependencies = tuple(dict.fromkeys(self.data_dependencies))
        self.control_dependencies = tuple(dict.fromkeys(self.control_dependencies))
        if self.effect_id in self.data_dependencies + self.control_dependencies:
            raise ValueError(f"effect {self.effect_id!r} cannot depend on itself")
        if self.admissible is None:
            self.admissible = self.monitor_decision is MonitorDecision.ALLOW
        if self.status is EffectStatus.PROPOSED:
            self.status = EffectStatus.ADMISSIBLE if self.admissible else EffectStatus.REJECTED
        if self.admissible and self.status is EffectStatus.REJECTED:
            raise ValueError("an admissible effect cannot start rejected")
        if not self.admissible and self.status is EffectStatus.ADMISSIBLE:
            raise ValueError("an inadmissible effect cannot start admissible")

    @classmethod
    def from_action(
        cls,
        action: ActionProposal,
        *,
        effect_id: str,
        monitor_decision: MonitorDecision,
        true_policy_label: TruePolicyLabel,
        policy_epoch: int,
    ) -> "StagedEffect":
        return cls(
            effect_id=effect_id,
            action_id=action.action_id,
            tool_name=action.tool_name,
            payload_digest=action.payload_digest,
            destination_digest=action.destination_digest,
            data_dependencies=action.data_dependencies,
            control_dependencies=action.control_dependencies,
            atomic_group=action.atomic_group,
            monitor_decision=monitor_decision,
            true_policy_label=true_policy_label,
            policy_epoch=policy_epoch,
            private_payload=action.payload,
            private_destination=action.destination,
        )

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.data_dependencies + self.control_dependencies))

    @property
    def is_prohibited(self) -> bool:
        return self.true_policy_label is TruePolicyLabel.PROHIBITED

    def trusted_dict(self, *, include_private_payload: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "effect_id": self.effect_id,
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "payload_digest": self.payload_digest,
            "destination_digest": self.destination_digest,
            "data_dependencies": list(self.data_dependencies),
            "control_dependencies": list(self.control_dependencies),
            "atomic_group": self.atomic_group,
            "monitor_decision": self.monitor_decision.value,
            "true_policy_label": self.true_policy_label.value,
            "policy_epoch": self.policy_epoch,
            "admissible": self.admissible,
            "status": self.status.value,
        }
        if include_private_payload:
            result["private_payload"] = _jsonable(self.private_payload)
            result["private_destination"] = _jsonable(self.private_destination)
        return result


@dataclass(frozen=True)
class CommitToken:
    effect_id: str
    policy_epoch: int
    payload_digest: str
    destination_digest: str
    signature: str

    @property
    def token(self) -> str:
        return self.signature


@dataclass(frozen=True)
class ToolResult:
    """Guardrail-independent temporary tool result."""

    ok: bool = True
    value: Any = None
    error_code: str | None = None
    error_message: str | None = None
    receipt_id: str | None = None
    identifier: str | None = None
    timing_bucket: str = "fixed"
    retries: int = 0
    effectful: bool = False


@dataclass(frozen=True)
class RuntimeStep:
    event: VisibleEvent
    private_record: PrivateEvaluationRecord
    staged_effect: StagedEffect | None = None

    @property
    def observation(self) -> VisibleEvent:
        return self.event


@dataclass(frozen=True)
class CommitReport:
    committed_effect_ids: tuple[str, ...]
    discarded_effect_ids: tuple[str, ...]
    rejected_effect_ids: tuple[str, ...]
    prohibited_effect_ids: tuple[str, ...] = ()
    dependency_violations: tuple[str, ...] = ()
    atomicity_violations: tuple[str, ...] = ()

    @property
    def committed(self) -> tuple[str, ...]:
        return self.committed_effect_ids

    @property
    def prohibited_effect_count(self) -> int:
        return len(self.prohibited_effect_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "committed_effect_ids": list(self.committed_effect_ids),
            "discarded_effect_ids": list(self.discarded_effect_ids),
            "rejected_effect_ids": list(self.rejected_effect_ids),
            "prohibited_effect_ids": list(self.prohibited_effect_ids),
            "dependency_violations": list(self.dependency_violations),
            "atomicity_violations": list(self.atomicity_violations),
        }


@dataclass(frozen=True)
class TrustedReport:
    """Final report delivered only to a distinct trusted evaluator."""

    session_id: str
    final_state: SessionState
    commit_report: CommitReport
    private_records: tuple[PrivateEvaluationRecord, ...]
    staged_effects: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "final_state": self.final_state.value,
            "commit_report": self.commit_report.to_dict(),
            "private_records": [record.trusted_dict() for record in self.private_records],
            "staged_effects": [dict(effect) for effect in self.staged_effects],
        }


__all__ = [
    "SCHEMA_VERSION",
    "ActionProposal",
    "AgentVisibleObservation",
    "AgentVisibleTranscript",
    "CommitReport",
    "CommitToken",
    "Decision",
    "EffectStatus",
    "MonitorDecision",
    "PolicyLabel",
    "PrivateEvaluationRecord",
    "ProposedAction",
    "RuntimeStep",
    "SessionState",
    "StagedEffect",
    "ToolResult",
    "TranscriptEvent",
    "TruePolicyLabel",
    "TrustedReport",
    "VisibleEvent",
    "assert_agent_visible",
    "is_agentdojo_public_envelope",
    "canonical_hash",
    "canonical_json",
    "stable_digest",
]
