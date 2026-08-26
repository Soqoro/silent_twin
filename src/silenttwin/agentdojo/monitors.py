"""Action-level monitor adapters for controlled AgentDojo experiments.

All adapters consume the same canonical public tuple.  They return trusted
guard evaluations and never place a profile identifier or verdict in the
model-visible conversation.  Learned adapters are intentionally fail-closed:
an unavailable checkpoint is an infrastructure error, not an implicit mock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from silenttwin.backends.base import GuardEvaluation
from silenttwin.schemas import canonical_json, stable_digest

from .canonical import CanonicalToolCall, CanonicalToolSchema, strict_json_loads
from .visibility import PublicEnvelope, TrustedOnly, public_value


class MonitorUnavailableError(RuntimeError):
    """The configured real monitor cannot be executed locally."""


class MonitorProtocolError(RuntimeError):
    """A monitor returned malformed or unsupported output."""


def _raw_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def monitor_text_hash(text: str) -> str:
    """Canonical hash algorithm for frozen prompt and policy text."""

    return stable_digest(str(text))


@dataclass(frozen=True, slots=True)
class MonitorInput:
    """The condition-independent tuple supplied to every action monitor."""

    frozen_user_goal: str
    public_conversation_summary: PublicEnvelope
    proposed_call: CanonicalToolCall
    tool_schema: CanonicalToolSchema

    def __post_init__(self) -> None:
        if not self.frozen_user_goal:
            raise ValueError("monitor input requires a frozen user goal")
        if not isinstance(self.public_conversation_summary, PublicEnvelope):
            object.__setattr__(
                self,
                "public_conversation_summary",
                public_value(self.public_conversation_summary),
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "frozen_user_goal": self.frozen_user_goal,
            "public_conversation_summary": self.public_conversation_summary.unwrap(),
            "proposed_call": self.proposed_call.to_dict(),
            "tool_schema": self.tool_schema.to_dict(),
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class MonitorProfile(TrustedOnly):
    """Immutable scientific identity for one action-gate configuration."""

    profile_id: str
    family: str
    implementation: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    checkpoint_fingerprint: str
    runtime_fingerprint: str
    dtype: str
    frozen_profile_hash: str
    prompt_hash: str
    policy_hash: str
    threshold: float = 0.5
    reasoning_mode: str = "direct"
    decoding: Mapping[str, Any] = field(default_factory=dict)
    prompt_template: str | None = None
    policy_text: str | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.profile_id,
                self.family,
                self.implementation,
                self.model_id,
                self.model_revision,
                self.tokenizer_revision,
                self.checkpoint_fingerprint,
                self.runtime_fingerprint,
                self.dtype,
                self.frozen_profile_hash,
                self.prompt_hash,
                self.policy_hash,
            )
        ):
            raise ValueError("monitor profile identity fields must be non-empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("monitor threshold must lie in [0,1]")
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "family": self.family,
            "implementation": self.implementation,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "dtype": self.dtype,
            "frozen_profile_hash": self.frozen_profile_hash,
            "prompt_hash": self.prompt_hash,
            "policy_hash": self.policy_hash,
            "threshold": self.threshold,
            "reasoning_mode": self.reasoning_mode,
            "decoding": dict(self.decoding),
            "prompt_template": self.prompt_template,
            "policy_text": self.policy_text,
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_trusted_dict())


@runtime_checkable
class ActionMonitor(Protocol):
    profile: MonitorProfile

    def evaluate(
        self, monitor_input: MonitorInput, *, plan_id: str, seed: int = 0
    ) -> GuardEvaluation: ...


class DeterministicPolicyMonitor:
    """Task-scoped authorization fixture with frozen call/plan decisions.

    The registry is trusted.  In particular, the model cannot invent policy
    labels or alter the call arguments used for the lookup.
    """

    def __init__(
        self,
        *,
        profile_id: str,
        decisions: Mapping[str, str | bool],
        default_decision: str = "block",
        policy_epoch: int = 0,
    ) -> None:
        if default_decision not in {"allow", "block"}:
            raise ValueError("default monitor decision must be allow or block")
        normalized: dict[str, str] = {}
        for key, value in decisions.items():
            if isinstance(value, bool):
                value = "allow" if value else "block"
            value = str(value).lower()
            if value not in {"allow", "block"}:
                raise ValueError(f"invalid deterministic decision for {key!r}")
            normalized[str(key)] = value
        self._decisions = MappingProxyType(normalized)
        self._default = default_decision
        self.policy_epoch = int(policy_epoch)
        identity = stable_digest(
            {
                "profile_id": profile_id,
                "decisions": normalized,
                "default": default_decision,
                "policy_epoch": self.policy_epoch,
            }
        )
        self.profile = MonitorProfile(
            profile_id=profile_id,
            family="deterministic_task_policy",
            implementation="builtin",
            model_id="deterministic-task-policy",
            model_revision=f"sha256:{identity}",
            tokenizer_revision="builtin:none",
            checkpoint_fingerprint=f"sha256:{identity}",
            runtime_fingerprint="builtin:deterministic-runtime-v1",
            dtype="deterministic",
            frozen_profile_hash=identity,
            prompt_hash=stable_digest("canonical-action-tuple-v1"),
            policy_hash=identity,
            threshold=0.5,
            reasoning_mode="deterministic",
            decoding={"temperature": 0.0},
        )

    def evaluate(
        self, monitor_input: MonitorInput, *, plan_id: str, seed: int = 0
    ) -> GuardEvaluation:
        del seed
        # Permit registries keyed by exact call digest, by function, or by the
        # controlled plan ID; most-specific wins.
        decision = self._decisions.get(
            monitor_input.proposed_call.digest,
            self._decisions.get(
                monitor_input.proposed_call.function,
                self._decisions.get(plan_id, self._default),
            ),
        )
        return GuardEvaluation(
            plan_id=plan_id,
            profile_id=self.profile.profile_id,
            monitor_family=self.profile.family,
            decision=decision,
            score=1.0 if decision == "block" else 0.0,
            policy_epoch=self.policy_epoch,
            provenance={
                "monitor_input_hash": monitor_input.digest,
                "monitor_profile_hash": self.profile.digest,
                "deterministic": True,
            },
        )


def _strict_monitor_output(text: str) -> tuple[str, float]:
    try:
        value = strict_json_loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MonitorProtocolError("monitor output is not JSON") from exc
    if not isinstance(value, dict) or set(value) != {"decision", "score"}:
        raise MonitorProtocolError(
            'monitor output must be exactly {"decision","score"}'
        )
    decision = str(value["decision"]).lower()
    if decision not in {"allow", "block"}:
        raise MonitorProtocolError("monitor decision must be allow or block")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise MonitorProtocolError("monitor score must be numeric")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise MonitorProtocolError("monitor score must lie in [0,1]")
    return decision, score


class LocalActionMonitor:
    """Strict adapter around an explicitly supplied local model client."""

    def __init__(self, profile: MonitorProfile, model_client: Any | None) -> None:
        if not profile.prompt_template or not profile.policy_text:
            raise ValueError(
                "learned monitor profiles require frozen prompt_template and policy_text"
            )
        if profile.prompt_template.count("{monitor_input_json}") != 1 or profile.prompt_template.count(
            "{policy_text}"
        ) != 1:
            raise ValueError(
                "monitor prompt_template must contain exactly one {monitor_input_json} and {policy_text} token"
            )
        if monitor_text_hash(profile.prompt_template) != profile.prompt_hash:
            raise ValueError("monitor prompt_template does not match prompt_hash")
        if monitor_text_hash(profile.policy_text) != profile.policy_hash:
            raise ValueError("monitor policy_text does not match policy_hash")
        self.profile = profile
        self._client = model_client
        self.last_failure_provenance: Mapping[str, Any] | None = None

    def _prompt(self, monitor_input: MonitorInput) -> str:
        body = canonical_json(monitor_input.to_canonical_dict())
        assert self.profile.prompt_template is not None
        assert self.profile.policy_text is not None
        return self.profile.prompt_template.replace(
            "{policy_text}", self.profile.policy_text
        ).replace("{monitor_input_json}", body)

    def evaluate(
        self, monitor_input: MonitorInput, *, plan_id: str, seed: int = 0
    ) -> GuardEvaluation:
        prompt = self._prompt(monitor_input)
        started = time.perf_counter()
        client = self._client
        response: Any | None = None
        raw_text: str | None = None
        materialized_metadata: dict[str, Any] = {}
        try:
            if client is None or not callable(getattr(client, "complete", None)):
                raise MonitorUnavailableError(
                    f"monitor profile {self.profile.profile_id!r} has no configured local checkpoint client"
                )
            response = client.complete(
                prompt,
                seed=int(seed),
                max_tokens=int(self.profile.decoding.get("max_new_tokens", 64)),
            )
            raw_value = getattr(response, "text", response)
            if not isinstance(raw_value, str):
                raise MonitorProtocolError("local monitor response has no text")
            raw_text = raw_value
            metadata = getattr(response, "metadata", {})
            usage = getattr(response, "usage", None)
            materialized_metadata = (
                dict(metadata) if isinstance(metadata, Mapping) else {}
            )
            materialized_metadata["model_role"] = "monitor"
            if usage is not None:
                materialized_metadata["usage"] = {
                    "input_tokens": int(getattr(usage, "input_tokens", 0)),
                    "output_tokens": int(getattr(usage, "output_tokens", 0)),
                    "total_tokens": int(getattr(usage, "total_tokens", 0)),
                }
            parsed_decision, score = _strict_monitor_output(raw_text)
            # The numeric score is authoritative at the frozen threshold.  A
            # disagreement with the textual decision is a protocol failure
            # rather than an undocumented tie-breaker.
            threshold_decision = "block" if score >= self.profile.threshold else "allow"
            if parsed_decision != threshold_decision:
                raise MonitorProtocolError(
                    "monitor decision disagrees with its frozen score threshold"
                )
            rendered_chat = str(
                materialized_metadata.get("rendered_input", prompt)
            )
            model_call = {
                "canonical_monitor_input": monitor_input.to_canonical_dict(),
                "protocol_prompt": prompt,
                "rendered_chat_template_input": rendered_chat,
                "rendered_input_hash": _raw_text_sha256(rendered_chat),
                "raw_response": raw_text,
                "raw_response_hash": _raw_text_sha256(raw_text),
                "parsed_output": {"decision": parsed_decision, "score": score},
                "seed": int(seed),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "metadata": materialized_metadata,
                "failure": None,
            }
            self.last_failure_provenance = None
        except Exception as exc:
            provider = getattr(client, "failure_metadata", None)
            failure_metadata = dict(provider()) if callable(provider) else {}
            if response is not None and not materialized_metadata:
                metadata = getattr(response, "metadata", {})
                materialized_metadata = (
                    dict(metadata) if isinstance(metadata, Mapping) else {}
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    materialized_metadata["usage"] = {
                        "input_tokens": int(getattr(usage, "input_tokens", 0)),
                        "output_tokens": int(getattr(usage, "output_tokens", 0)),
                        "total_tokens": int(getattr(usage, "total_tokens", 0)),
                    }
                materialized_metadata["model_role"] = "monitor"
            failure_rendered_chat = str(
                materialized_metadata.get(
                    "rendered_input",
                    failure_metadata.get("rendered_input", prompt),
                )
            )
            self.last_failure_provenance = {
                "canonical_monitor_input": monitor_input.to_canonical_dict(),
                "protocol_prompt": prompt,
                "rendered_chat_template_input": failure_rendered_chat,
                "rendered_input_hash": _raw_text_sha256(failure_rendered_chat),
                "raw_response": raw_text,
                "raw_response_hash": (
                    _raw_text_sha256(raw_text) if raw_text is not None else None
                ),
                "parsed_output": None,
                "seed": int(seed),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "metadata": materialized_metadata,
                "failure_metadata": failure_metadata,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
            raise
        return GuardEvaluation(
            plan_id=plan_id,
            profile_id=self.profile.profile_id,
            monitor_family=self.profile.family,
            decision=parsed_decision,
            score=score,
            provenance={
                "monitor_input_hash": monitor_input.digest,
                "monitor_profile_hash": self.profile.digest,
                "frozen_profile_hash": self.profile.frozen_profile_hash,
                "rendered_prompt_hash": stable_digest(prompt),
                "raw_response_hash": _raw_text_sha256(raw_text),
                "seed": int(seed),
                "model_metadata": materialized_metadata,
                "model_call": model_call,
            },
        )


class GraniteGuardianMonitor(LocalActionMonitor):
    """Granite Guardian action-gate adapter; checkpoint must be operator supplied."""

    def __init__(self, profile: MonitorProfile, model_client: Any | None) -> None:
        if profile.family != "granite_guardian_4_1_8b":
            raise ValueError("Granite adapter requires family=granite_guardian_4_1_8b")
        super().__init__(profile, model_client)


class GptOssSafeguardMonitor(LocalActionMonitor):
    """gpt-oss-safeguard action-gate adapter; no synthetic fallback."""

    def __init__(self, profile: MonitorProfile, model_client: Any | None) -> None:
        if profile.family != "gpt_oss_safeguard_20b":
            raise ValueError("gpt-oss adapter requires family=gpt_oss_safeguard_20b")
        super().__init__(profile, model_client)


class PromptInjectionDetectorAdapter:
    """Ecological content/input detector, deliberately not an action monitor."""

    family = "transformers_pi_detector"
    role = "content_input_detector"

    def __init__(
        self,
        detector: Callable[
            [str], bool | tuple[bool, float] | Mapping[str, Any]
        ]
        | None,
        *,
        profile_id: str = "agentdojo-transformers-pi-detector",
    ) -> None:
        self._detector = detector
        self.profile_id = profile_id

    def evaluate_content(self, content: str) -> Mapping[str, Any]:
        if self._detector is None:
            raise MonitorUnavailableError(
                "AgentDojo transformer prompt-injection detector is not configured"
            )
        value = self._detector(content)
        if isinstance(value, bool):
            return {
                "detected": value,
                "family": self.family,
                "role": self.role,
                "profile_id": self.profile_id,
            }
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], bool)
            and isinstance(value[1], (int, float))
            and not isinstance(value[1], bool)
            and 0.0 <= float(value[1]) <= 1.0
        ):
            return {
                "detected": value[0],
                "safety_score": float(value[1]),
                "family": self.family,
                "role": self.role,
                "profile_id": self.profile_id,
            }
        if (
            not isinstance(value, Mapping)
            or not isinstance(value.get("detected"), bool)
        ):
            raise MonitorProtocolError("content detector returned an invalid result")
        return {
            **dict(value),
            "detected": value["detected"],
            "family": self.family,
            "role": self.role,
            "profile_id": self.profile_id,
        }


@dataclass(frozen=True, slots=True)
class MonitorPair(TrustedOnly):
    theta0: ActionMonitor
    theta1: ActionMonitor
    pair_registry_hash: str

    def __post_init__(self) -> None:
        if not self.pair_registry_hash:
            raise ValueError("monitor pair requires a frozen registry hash")
        if self.theta0.profile.profile_id == self.theta1.profile.profile_id:
            raise ValueError("theta monitor profiles must be distinct")
        if self.theta0.profile.family != self.theta1.profile.family:
            raise ValueError("theta monitor profiles must use one compatible family")

    def for_theta(self, theta: str) -> ActionMonitor:
        if theta == "theta0":
            return self.theta0
        if theta == "theta1":
            return self.theta1
        raise ValueError(f"unknown private monitor state: {theta!r}")

    def to_trusted_dict(self) -> dict[str, Any]:
        return {
            "theta0_profile": self.theta0.profile.to_trusted_dict(),
            "theta1_profile": self.theta1.profile.to_trusted_dict(),
            "pair_registry_hash": self.pair_registry_hash,
        }


__all__ = [
    "ActionMonitor",
    "DeterministicPolicyMonitor",
    "GptOssSafeguardMonitor",
    "GraniteGuardianMonitor",
    "LocalActionMonitor",
    "MonitorInput",
    "MonitorPair",
    "MonitorProfile",
    "MonitorProtocolError",
    "MonitorUnavailableError",
    "monitor_text_hash",
    "PromptInjectionDetectorAdapter",
]
