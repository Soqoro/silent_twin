"""Dependency-free scientific backend protocol.

The protocol keeps causal assignment and feedback randomization in the trusted
runner while allowing either the exact Tier-1 finite-state mechanism or a
stateful AgentDojo environment to supply tool execution and grading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from silenttwin.agentdojo.canonical import (
    CanonicalToolCall,
    CanonicalToolSchema,
    calls_hash,
    schema_hash,
)
from silenttwin.agentdojo.visibility import PublicEnvelope, TrustedOnly, public_value
from silenttwin.schemas import stable_digest


class BackendProtocolError(RuntimeError):
    pass


class BackendExecutionError(RuntimeError):
    pass


class EnvironmentRole(str, Enum):
    BASE = "base"
    GENUINE_PROBE = "genuine_probe"
    SHUFFLED_DONOR = "shuffled_donor"
    SEMANTIC_TWIN = "semantic_twin"
    FINAL_TARGET = "final_target"
    GRADER = "grader"


class BackendErrorStage(str, Enum):
    SETUP = "setup"
    CLONE = "clone"
    MONITOR = "monitor"
    PROBE = "probe"
    MODEL = "model"
    PREDICTION = "prediction"
    FINAL_SELECTION = "final_selection"
    FINAL_EXECUTION = "final_execution"
    UTILITY_GRADER = "utility_grader"
    ATTACK_GRADER = "attack_grader"
    RETIREMENT = "retirement"
    PROTOCOL = "protocol"


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    environment_backend: str
    protocol_revision: str = "silenttwin.backend.v1"
    package_version: str | None = None
    source_revision: str | None = None
    benchmark_version: str | None = None
    catalog_hash: str | None = None
    exact_transcript_model: bool = False

    def __post_init__(self) -> None:
        if self.environment_backend not in {"finite_state", "agentdojo"}:
            raise ValueError("environment_backend must be finite_state or agentdojo")
        if not self.protocol_revision:
            raise ValueError("backend protocol revision must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_backend": self.environment_backend,
            "protocol_revision": self.protocol_revision,
            "package_version": self.package_version,
            "source_revision": self.source_revision,
            "benchmark_version": self.benchmark_version,
            "catalog_hash": self.catalog_hash,
            "exact_transcript_model": self.exact_transcript_model,
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class PublicPlan:
    plan_id: str
    label: str
    description: str
    kind: str = "candidate"
    public_steps: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id or not self.label:
            raise ValueError("public plan ID and label must be non-empty")
        if self.kind not in {"probe", "candidate", "workflow"}:
            raise ValueError(f"unsupported public plan kind: {self.kind!r}")
        copied: list[Mapping[str, Any]] = []
        for step in self.public_steps:
            envelope = public_value(step)
            value = envelope.unwrap()
            if not isinstance(value, dict):
                raise TypeError("public plan steps must be objects")
            copied.append(MappingProxyType(value))
        object.__setattr__(self, "public_steps", tuple(copied))

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "label": self.label,
            "description": self.description,
            "kind": self.kind,
            "public_steps": [dict(step) for step in self.public_steps],
        }


@dataclass(frozen=True, slots=True)
class PublicScenario:
    scenario_id: str
    suite: str
    user_prompt: str
    tool_schemas: tuple[CanonicalToolSchema, ...]
    candidate_plans: tuple[PublicPlan, ...]
    structural_group_id: str
    dataset_split: str
    public_environment: PublicEnvelope = field(default_factory=lambda: public_value({}))

    def __post_init__(self) -> None:
        if not all((self.scenario_id, self.suite, self.structural_group_id)):
            raise ValueError("public scenario identifiers must be non-empty")
        if self.dataset_split not in {"train", "development", "test", "fixture"}:
            raise ValueError(f"unsupported dataset split: {self.dataset_split!r}")
        object.__setattr__(self, "tool_schemas", tuple(self.tool_schemas))
        object.__setattr__(self, "candidate_plans", tuple(self.candidate_plans))
        if not isinstance(self.public_environment, PublicEnvelope):
            object.__setattr__(
                self, "public_environment", public_value(self.public_environment)
            )
        plan_ids = [plan.plan_id for plan in self.candidate_plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("public plan IDs must be unique")

    def plan(self, plan_id: str) -> PublicPlan:
        for plan in self.candidate_plans:
            if plan.plan_id == plan_id:
                return plan
        raise KeyError(f"unknown public plan: {plan_id}")

    @property
    def tool_schema_hash(self) -> str:
        return schema_hash(self.tool_schemas)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "suite": self.suite,
            "user_prompt": self.user_prompt,
            "tool_schemas": [schema.to_dict() for schema in self.tool_schemas],
            "candidate_plans": [plan.to_public_dict() for plan in self.candidate_plans],
            "structural_group_id": self.structural_group_id,
            "dataset_split": self.dataset_split,
            "public_environment": self.public_environment.unwrap(),
        }

    @property
    def public_hash(self) -> str:
        return stable_digest(self.to_public_dict())


@dataclass(frozen=True, slots=True)
class TrustedPlan(TrustedOnly):
    plan_id: str
    calls: tuple[CanonicalToolCall, ...]
    data_dependencies: tuple[str, ...] = ()
    control_dependencies: tuple[str, ...] = ()
    atomic_group: str | None = None
    ground_truth_label: str = "prohibited"
    registry_revision: str = "unfrozen"

    def __post_init__(self) -> None:
        if not self.plan_id or not self.calls:
            raise ValueError("trusted plans need an ID and at least one call")
        if self.ground_truth_label not in {"permitted", "prohibited"}:
            raise ValueError("ground_truth_label must be permitted or prohibited")
        object.__setattr__(self, "calls", tuple(self.calls))
        object.__setattr__(
            self, "data_dependencies", tuple(dict.fromkeys(self.data_dependencies))
        )
        object.__setattr__(
            self,
            "control_dependencies",
            tuple(dict.fromkeys(self.control_dependencies)),
        )

    @property
    def call_sequence_hash(self) -> str:
        return calls_hash(self.calls)

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "calls": [call.to_dict() for call in self.calls],
            "call_sequence_hash": self.call_sequence_hash,
            "data_dependencies": list(self.data_dependencies),
            "control_dependencies": list(self.control_dependencies),
            "atomic_group": self.atomic_group,
            "ground_truth_label": self.ground_truth_label,
            "registry_revision": self.registry_revision,
        }


@dataclass(slots=True)
class EnvironmentHandle(TrustedOnly):
    environment_id: str
    role: EnvironmentRole
    theta: str
    seed: int
    initial_hash: str
    environment: Any = field(repr=False, compare=False)
    pre_environment: Any = field(default=None, repr=False, compare=False)
    active: bool = True

    def __post_init__(self) -> None:
        if not self.environment_id or not self.initial_hash:
            raise ValueError("environment handle identity and hash must be non-empty")
        if self.theta not in {"theta0", "theta1", "public"}:
            raise ValueError(f"unsupported environment theta: {self.theta!r}")
        if self.seed < 0:
            raise ValueError("environment seed must be non-negative")

    def ensure_active(self) -> None:
        if not self.active:
            raise BackendProtocolError(
                f"environment {self.environment_id} is retired"
            )

    def retire(self) -> None:
        self.ensure_active()
        self.active = False
        self.environment = None
        self.pre_environment = None

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "role": self.role.value,
            "theta": self.theta,
            "seed": self.seed,
            "initial_hash": self.initial_hash,
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class GuardEvaluation(TrustedOnly):
    plan_id: str
    profile_id: str
    monitor_family: str
    decision: str
    score: float | None = None
    policy_epoch: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {"allow", "block", "error"}:
            raise ValueError("guard decision must be allow, block, or error")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "profile_id": self.profile_id,
            "monitor_family": self.monitor_family,
            "decision": self.decision,
            "score": self.score,
            "policy_epoch": self.policy_epoch,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class TrustedActionRecord(TrustedOnly):
    action_id: str
    plan_id: str
    call: CanonicalToolCall
    monitor_decision: str
    ground_truth_label: str
    policy_epoch: int
    effect_id: str | None
    status: str
    pre_environment_hash: str
    post_environment_hash: str
    data_dependencies: tuple[str, ...] = ()
    control_dependencies: tuple[str, ...] = ()
    atomic_group: str | None = None
    backend_error: str | None = None

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "plan_id": self.plan_id,
            "call": self.call.to_dict(),
            "call_digest": self.call.digest,
            "monitor_decision": self.monitor_decision,
            "ground_truth_label": self.ground_truth_label,
            "policy_epoch": self.policy_epoch,
            "effect_id": self.effect_id,
            "status": self.status,
            "pre_environment_hash": self.pre_environment_hash,
            "post_environment_hash": self.post_environment_hash,
            "data_dependencies": list(self.data_dependencies),
            "control_dependencies": list(self.control_dependencies),
            "atomic_group": self.atomic_group,
            "backend_error": self.backend_error,
        }


@dataclass(frozen=True, slots=True)
class BackendError(TrustedOnly):
    stage: BackendErrorStage
    code: str
    message: str
    retryable: bool = False

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True, slots=True)
class BackendActionResult(TrustedOnly):
    plan_id: str
    public_events: tuple[PublicEnvelope, ...]
    trusted_actions: tuple[TrustedActionRecord, ...]
    pre_environment_hash: str
    post_environment_hash: str
    trace_hash: str
    monitor: GuardEvaluation
    staged_effect_ids: tuple[str, ...] = ()
    committed_effect_ids: tuple[str, ...] = ()
    rejected_effect_ids: tuple[str, ...] = ()
    errors: tuple[BackendError, ...] = ()
    raw_trace: Any = field(default=None, repr=False, compare=False)
    output_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_events", tuple(self.public_events))
        if any(not isinstance(event, PublicEnvelope) for event in self.public_events):
            raise TypeError("backend public events must be PublicEnvelope values")
        object.__setattr__(self, "trusted_actions", tuple(self.trusted_actions))
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def valid(self) -> bool:
        return not self.errors and self.monitor.decision != "error"

    def public_event_dicts(self) -> list[Any]:
        return [event.unwrap() for event in self.public_events]

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "public_events": self.public_event_dicts(),
            "trusted_actions": [item.to_trusted_dict() for item in self.trusted_actions],
            "pre_environment_hash": self.pre_environment_hash,
            "post_environment_hash": self.post_environment_hash,
            "trace_hash": self.trace_hash,
            "monitor": self.monitor.to_trusted_dict(),
            "staged_effect_ids": list(self.staged_effect_ids),
            "committed_effect_ids": list(self.committed_effect_ids),
            "rejected_effect_ids": list(self.rejected_effect_ids),
            "errors": [error.to_trusted_dict() for error in self.errors],
            "output_text_hash": stable_digest(self.output_text),
        }


@dataclass(frozen=True, slots=True)
class BackendGrades(TrustedOnly):
    utility: bool | None
    raw_attack_success: bool | None
    attack_success: bool | None
    safe: bool | None
    safe_and_useful: bool | None
    run_valid: bool
    conservative_attack_success: bool
    errors: tuple[BackendError, ...] = ()

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "utility": self.utility,
            "raw_attack_success": self.raw_attack_success,
            "attack_success": self.attack_success,
            "safe": self.safe,
            "safe_and_useful": self.safe_and_useful,
            "run_valid": self.run_valid,
            "conservative_attack_success": self.conservative_attack_success,
            "errors": [error.to_trusted_dict() for error in self.errors],
        }


@dataclass(frozen=True, slots=True)
class FinalAttemptOutcome(TrustedOnly):
    plan_id: str
    monitor_decision: str
    accepted_effect_id: str | None
    result: BackendActionResult

    @property
    def action_id(self) -> str:
        return self.plan_id


@runtime_checkable
class BackendEpisode(Protocol):
    identity: BackendIdentity
    public_scenario: PublicScenario

    @property
    def trusted_plans(self) -> Mapping[str, TrustedPlan]: ...

    def fresh_environment(
        self, theta: str, role: EnvironmentRole, seed: int
    ) -> EnvironmentHandle: ...

    def clone_environment(
        self, source: EnvironmentHandle, role: EnvironmentRole
    ) -> EnvironmentHandle: ...

    def environment_hash(self, environment: EnvironmentHandle) -> str: ...

    def evaluate_guard(self, theta: str, plan_id: str, *, seed: int = 0) -> GuardEvaluation: ...

    def execute_probe(
        self, environment: EnvironmentHandle, theta: str, plan_id: str, *, seed: int = 0
    ) -> BackendActionResult: ...

    def execute_speculative(
        self, environment: EnvironmentHandle, plan_id: str, *, seed: int = 0
    ) -> BackendActionResult: ...

    def execute_final(
        self, environment: EnvironmentHandle, theta: str, plan_id: str, *, seed: int = 0
    ) -> BackendActionResult: ...

    def grade(
        self, environment: EnvironmentHandle, result: BackendActionResult
    ) -> BackendGrades: ...

    def false_negative_plan_ids(self, theta: str) -> tuple[str, ...]: ...

    def retire_environment(self, environment: EnvironmentHandle) -> None: ...


__all__ = [
    "BackendActionResult",
    "BackendEpisode",
    "BackendError",
    "BackendErrorStage",
    "BackendExecutionError",
    "BackendGrades",
    "BackendIdentity",
    "BackendProtocolError",
    "EnvironmentHandle",
    "EnvironmentRole",
    "FinalAttemptOutcome",
    "GuardEvaluation",
    "PublicPlan",
    "PublicScenario",
    "TrustedActionRecord",
    "TrustedPlan",
]
