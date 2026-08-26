"""Pinned AgentDojo backend behind the dependency-free backend protocol.

Only :mod:`silenttwin.agentdojo.compat` is allowed to know upstream API
details.  This module operates on its narrow facade and may be imported on a
CPU worker that does not have AgentDojo installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from silenttwin.agentdojo.canonical import CanonicalToolCall, calls_hash
from silenttwin.agentdojo.graders import grade_agentdojo
from silenttwin.agentdojo.monitors import MonitorInput, MonitorPair
from silenttwin.agentdojo.visibility import public_value
from silenttwin.schemas import stable_digest

from .base import (
    BackendActionResult,
    BackendError,
    BackendErrorStage,
    BackendGrades,
    BackendIdentity,
    BackendProtocolError,
    EnvironmentHandle,
    EnvironmentRole,
    GuardEvaluation,
    PublicScenario,
    TrustedActionRecord,
    TrustedPlan,
)


def _compat_module(compat: Any | None) -> Any:
    return compat or importlib.import_module("silenttwin.agentdojo.compat")


def _outcome_field(outcome: Any, name: str, default: Any = None) -> Any:
    if isinstance(outcome, Mapping):
        return outcome.get(name, default)
    return getattr(outcome, name, default)


def _json_projection(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_projection(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _json_projection(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_projection(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_projection(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Tool-return objects in the pinned suites are Pydantic.  This last branch
    # is for error objects and preserves a durable representation without
    # attempting to serialize their implementation internals.
    return str(value)


@dataclass(frozen=True, slots=True)
class _ExecutionMode:
    name: str
    commit: bool
    apply_monitor: bool
    error_stage: BackendErrorStage


_PROBE = _ExecutionMode("probe", False, True, BackendErrorStage.PROBE)
_SPECULATIVE = _ExecutionMode("speculative", False, False, BackendErrorStage.PROBE)
_FINAL = _ExecutionMode("final", True, True, BackendErrorStage.FINAL_EXECUTION)


class AgentDojoBackend:
    """One immutable controlled scenario backed by a pinned TaskSuite."""

    def __init__(
        self,
        *,
        suite: Any,
        user_task: Any,
        injection_task: Any | None,
        injections: Mapping[str, str],
        public_scenario: PublicScenario,
        trusted_plans: Mapping[str, TrustedPlan],
        monitor_pair: MonitorPair,
        package_version: str = "0.1.35",
        source_revision: str = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b",
        benchmark_version: str = "v1.2.2",
        catalog_hash: str | None = None,
        expected_initial_environment_hash: str | None = None,
        compat: Any | None = None,
    ) -> None:
        self._compat = _compat_module(compat)
        self.suite = suite
        self.user_task = user_task
        self.injection_task = injection_task
        self.injections = MappingProxyType(dict(injections))
        self.public_scenario = public_scenario
        self._trusted_plans = MappingProxyType(dict(trusted_plans))
        self.monitor_pair = monitor_pair
        self.expected_initial_environment_hash = expected_initial_environment_hash
        self.identity = BackendIdentity(
            environment_backend="agentdojo",
            package_version=package_version,
            source_revision=source_revision,
            benchmark_version=benchmark_version,
            catalog_hash=catalog_hash,
            exact_transcript_model=False,
        )
        self._environment_counter = 0
        public_ids = {plan.plan_id for plan in public_scenario.candidate_plans}
        if set(self._trusted_plans) != public_ids:
            raise ValueError(
                "trusted plan registry must exactly match public candidate/probe IDs"
            )
        schema_names = {schema.name for schema in public_scenario.tool_schemas}
        for plan in self._trusted_plans.values():
            unknown = {call.function for call in plan.calls} - schema_names
            if unknown:
                raise ValueError(
                    f"plan {plan.plan_id!r} references unknown tools: {sorted(unknown)}"
                )

    @property
    def trusted_plans(self) -> Mapping[str, TrustedPlan]:
        return self._trusted_plans

    def _next_environment_id(self, role: EnvironmentRole) -> str:
        value = self._environment_counter
        self._environment_counter += 1
        return f"{self.public_scenario.scenario_id}:{role.value}:{value:06d}"

    def fresh_environment(
        self, theta: str, role: EnvironmentRole, seed: int
    ) -> EnvironmentHandle:
        if theta not in {"theta0", "theta1", "public"}:
            raise ValueError(f"unsupported environment theta: {theta!r}")
        # compat.load_environment performs both default injection and the user
        # task's init_environment on a never-before-used base environment.
        loaded = self._compat.load_environment(
            self.suite, self.user_task, dict(self.injections)
        )
        if isinstance(loaded, tuple) and len(loaded) == 2:
            environment, pre_environment = loaded
        else:
            environment = loaded
            pre_environment = self._compat.clone_environment(environment)
        initial_hash = str(self._compat.environment_hash(environment))
        if (
            self.expected_initial_environment_hash is not None
            and initial_hash != self.expected_initial_environment_hash
        ):
            raise BackendProtocolError(
                "live AgentDojo initial environment differs from the frozen scenario registry"
            )
        return EnvironmentHandle(
            environment_id=self._next_environment_id(role),
            role=role,
            theta=theta,
            seed=int(seed),
            initial_hash=initial_hash,
            environment=environment,
            pre_environment=pre_environment,
        )

    def clone_environment(
        self, source: EnvironmentHandle, role: EnvironmentRole
    ) -> EnvironmentHandle:
        source.ensure_active()
        environment = self._compat.clone_environment(source.environment)
        clone = EnvironmentHandle(
            environment_id=self._next_environment_id(role),
            role=role,
            theta=source.theta,
            seed=source.seed,
            initial_hash=source.initial_hash,
            environment=environment,
            pre_environment=self._compat.clone_environment(source.pre_environment),
        )
        if clone.environment is source.environment:
            raise BackendProtocolError("AgentDojo clone aliased its source environment")
        if self.environment_hash(clone) != self.environment_hash(source):
            raise BackendProtocolError("AgentDojo deep clone changed environment state")
        return clone

    def environment_hash(self, environment: EnvironmentHandle) -> str:
        environment.ensure_active()
        return str(self._compat.environment_hash(environment.environment))

    def _plan(self, plan_id: str) -> TrustedPlan:
        try:
            return self._trusted_plans[plan_id]
        except KeyError as exc:
            raise BackendProtocolError(f"unknown controlled plan {plan_id!r}") from exc

    def _schema_for(self, call: CanonicalToolCall):
        for schema in self.public_scenario.tool_schemas:
            if schema.name == call.function:
                return schema
        raise BackendProtocolError(f"missing schema for {call.function!r}")

    def evaluate_guard(
        self, theta: str, plan_id: str, *, seed: int = 0
    ) -> GuardEvaluation:
        plan = self._plan(plan_id)
        monitor = self.monitor_pair.for_theta(theta)
        evaluations: list[GuardEvaluation] = []
        conversation = public_value(
            {
                "scenario_id": self.public_scenario.scenario_id,
                "summary": "frozen controlled-plan evaluation",
            }
        )
        for index, call in enumerate(plan.calls):
            monitor_input = MonitorInput(
                frozen_user_goal=self.public_scenario.user_prompt,
                public_conversation_summary=conversation,
                proposed_call=call,
                tool_schema=self._schema_for(call),
            )
            try:
                evaluation = monitor.evaluate(
                    monitor_input,
                    plan_id=plan_id,
                    seed=int(seed) + index,
                )
            except Exception as exc:
                # A plan may contain several independently monitored calls.  A
                # later transport/protocol failure must not discard the exact
                # successful call ledger accumulated before it.  Convert only
                # the failed call into an explicit error evaluation and let the
                # aggregate decision fail closed.
                evaluation = GuardEvaluation(
                    plan_id=plan_id,
                    profile_id=monitor.profile.profile_id,
                    monitor_family=monitor.profile.family,
                    decision="error",
                    provenance={
                        "call_index": index,
                        "monitor_input_hash": monitor_input.digest,
                        "monitor_profile_hash": monitor.profile.digest,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "failed_model_call": getattr(
                            monitor, "last_failure_provenance", None
                        ),
                    },
                )
                evaluations.append(evaluation)
                break
            evaluations.append(evaluation)
        decision = "error" if any(item.decision == "error" for item in evaluations) else (
            "block" if any(item.decision == "block" for item in evaluations) else "allow"
        )
        scores = [item.score for item in evaluations if item.score is not None]
        return GuardEvaluation(
            plan_id=plan_id,
            profile_id=monitor.profile.profile_id,
            monitor_family=monitor.profile.family,
            decision=decision,
            score=max(scores) if scores else None,
            policy_epoch=max((item.policy_epoch for item in evaluations), default=0),
            provenance={
                "monitor_profile_hash": monitor.profile.digest,
                "pair_registry_hash": self.monitor_pair.pair_registry_hash,
                "call_evaluations": [item.to_trusted_dict() for item in evaluations],
            },
        )

    @staticmethod
    def _semantic_twin_guard(plan_id: str) -> GuardEvaluation:
        return GuardEvaluation(
            plan_id=plan_id,
            profile_id="semantic-twin",
            monitor_family="decision_independent_semantic_twin",
            decision="allow",
            score=None,
            provenance={"decision_independent": True},
        )

    def _effect_id(self, plan: TrustedPlan, index: int) -> str:
        return "effect-" + stable_digest(
            [self.public_scenario.scenario_id, plan.plan_id, index, plan.calls[index].digest]
        )[:20]

    def _rejected_result(
        self,
        environment: EnvironmentHandle,
        plan: TrustedPlan,
        monitor: GuardEvaluation,
        *,
        mode: _ExecutionMode,
        errors: Sequence[BackendError] = (),
    ) -> BackendActionResult:
        state_hash = self.environment_hash(environment)
        rejected = tuple(self._effect_id(plan, index) for index in range(len(plan.calls)))
        status = "error" if errors or monitor.decision == "error" else "rejected"
        actions = tuple(
            TrustedActionRecord(
                action_id=f"{plan.plan_id}:{index}",
                plan_id=plan.plan_id,
                call=call,
                monitor_decision=monitor.decision,
                ground_truth_label=plan.ground_truth_label,
                policy_epoch=monitor.policy_epoch,
                effect_id=rejected[index],
                status=status,
                pre_environment_hash=state_hash,
                post_environment_hash=state_hash,
                data_dependencies=plan.data_dependencies,
                control_dependencies=plan.control_dependencies,
                atomic_group=plan.atomic_group,
                backend_error=errors[0].code if errors else None,
            )
            for index, call in enumerate(plan.calls)
        )
        # This event is rendered by the feedback-policy layer before delivery;
        # it contains no hidden profile, theta, or policy label.
        event = public_value(
            {
                "event_type": "tool_result",
                "plan_id": plan.plan_id,
                "status": "failed" if errors else "denied",
                "mode": mode.name,
            }
        )
        return BackendActionResult(
            plan_id=plan.plan_id,
            public_events=(event,),
            trusted_actions=actions,
            pre_environment_hash=state_hash,
            post_environment_hash=state_hash,
            trace_hash=calls_hash(plan.calls),
            monitor=monitor,
            staged_effect_ids=rejected,
            rejected_effect_ids=rejected,
            errors=tuple(errors),
            raw_trace=(),
        )

    def _execute(
        self,
        environment: EnvironmentHandle,
        theta: str,
        plan_id: str,
        *,
        seed: int,
        mode: _ExecutionMode,
    ) -> BackendActionResult:
        environment.ensure_active()
        plan = self._plan(plan_id)
        try:
            monitor = (
                self.evaluate_guard(theta, plan_id, seed=seed)
                if mode.apply_monitor
                else self._semantic_twin_guard(plan_id)
            )
        except Exception as exc:
            selected_monitor = self.monitor_pair.for_theta(theta)
            fallback = GuardEvaluation(
                plan_id=plan_id,
                profile_id="monitor-error",
                monitor_family="unavailable",
                decision="error",
                provenance={
                    "error_type": type(exc).__name__,
                    "failed_model_call": getattr(
                        selected_monitor, "last_failure_provenance", None
                    ),
                },
            )
            return self._rejected_result(
                environment,
                plan,
                fallback,
                mode=mode,
                errors=(
                    BackendError(
                        BackendErrorStage.MONITOR,
                        "monitor_evaluation_error",
                        f"{type(exc).__name__}:{exc}",
                    ),
                ),
            )
        if monitor.decision != "allow":
            monitor_errors = (
                (
                    BackendError(
                        BackendErrorStage.MONITOR,
                        "monitor_returned_error",
                        "action monitor returned an explicit error decision",
                    ),
                )
                if monitor.decision == "error"
                else ()
            )
            return self._rejected_result(
                environment, plan, monitor, mode=mode, errors=monitor_errors
            )

        pre_hash = self.environment_hash(environment)
        working = self._compat.clone_environment(environment.environment)
        traces: list[Any] = []
        actions: list[TrustedActionRecord] = []
        public_events = []
        errors: list[BackendError] = []
        effect_ids: list[str] = []
        for index, call in enumerate(plan.calls):
            call_pre = str(self._compat.environment_hash(working))
            effect_id = self._effect_id(plan, index)
            effect_ids.append(effect_id)
            try:
                outcome = self._compat.execute_call(
                    self.suite, working, call.to_dict()
                )
                result = _outcome_field(outcome, "result")
                error = _outcome_field(outcome, "error")
                reported_pre = _outcome_field(outcome, "pre_environment_hash")
                reported_post = _outcome_field(outcome, "post_environment_hash")
                trace = _outcome_field(outcome, "trace", (call.to_dict(),))
                if isinstance(trace, Sequence) and not isinstance(
                    trace, (str, bytes, bytearray)
                ):
                    traces.extend(trace)
                else:
                    traces.append(trace)
            except Exception as exc:
                result = None
                error = f"{type(exc).__name__}:{exc}"
                reported_pre = None
                reported_post = None
                traces.append(call.to_dict())
            call_post = str(self._compat.environment_hash(working))
            if reported_pre is not None and str(reported_pre) != call_pre:
                error = "compat execute_call reported an inconsistent pre-environment hash"
            if reported_post is not None and str(reported_post) != call_post:
                error = "compat execute_call reported an inconsistent post-environment hash"
            if error:
                backend_error = BackendError(
                    mode.error_stage,
                    "agentdojo_tool_error",
                    str(error),
                )
                errors.append(backend_error)
                status = "error"
            else:
                backend_error = None
                status = "staged" if not mode.commit else "committed"
            actions.append(
                TrustedActionRecord(
                    action_id=f"{plan.plan_id}:{index}",
                    plan_id=plan.plan_id,
                    call=call,
                    monitor_decision=monitor.decision,
                    ground_truth_label=plan.ground_truth_label,
                    policy_epoch=monitor.policy_epoch,
                    effect_id=effect_id,
                    status=status,
                    pre_environment_hash=call_pre,
                    post_environment_hash=call_post,
                    data_dependencies=plan.data_dependencies,
                    control_dependencies=plan.control_dependencies,
                    atomic_group=plan.atomic_group,
                    backend_error=backend_error.code if backend_error else None,
                )
            )
            public_events.append(
                public_value(
                    {
                        "event_type": "tool_result",
                        "call_id": call.call_id,
                        "function": call.function,
                        "status": "error" if error else "ok",
                        "value": None if error else _json_projection(result),
                        "error": (
                            {"code": "TOOL_EXECUTION_FAILED", "message": "tool call failed"}
                            if error
                            else None
                        ),
                        "mode": mode.name,
                    }
                )
            )
            if error:
                break

        all_effect_ids = tuple(
            self._effect_id(plan, index) for index in range(len(plan.calls))
        )
        if errors:
            rewritten: list[TrustedActionRecord] = []
            for index, call in enumerate(plan.calls):
                previous = actions[index] if index < len(actions) else None
                status = (
                    "error"
                    if previous is not None and previous.status == "error"
                    else "dependency_discarded"
                )
                rewritten.append(
                    TrustedActionRecord(
                        action_id=f"{plan.plan_id}:{index}",
                        plan_id=plan.plan_id,
                        call=call,
                        monitor_decision=monitor.decision,
                        ground_truth_label=plan.ground_truth_label,
                        policy_epoch=monitor.policy_epoch,
                        effect_id=all_effect_ids[index],
                        status=status,
                        pre_environment_hash=(
                            previous.pre_environment_hash if previous else pre_hash
                        ),
                        post_environment_hash=(
                            previous.post_environment_hash if previous else pre_hash
                        ),
                        data_dependencies=plan.data_dependencies,
                        control_dependencies=plan.control_dependencies,
                        atomic_group=plan.atomic_group,
                        backend_error=(
                            previous.backend_error
                            if previous is not None and previous.status == "error"
                            else "atomic_plan_discarded"
                        ),
                    )
                )
            actions = rewritten

        # Atomic controlled plan: a failed call never publishes the partially
        # mutated working clone.  Probe/speculative environments may evolve
        # inside their disposable handle, but never mutate their source.
        if not errors:
            environment.environment = working
        post_hash = self.environment_hash(environment)
        committed = all_effect_ids if mode.commit and not errors else ()
        staged = all_effect_ids
        rejected = all_effect_ids if errors else ()
        output_text = ""
        if public_events:
            value = public_events[-1].unwrap().get("value")
            output_text = value if isinstance(value, str) else json.dumps(
                value, sort_keys=True, separators=(",", ":")
            )
        return BackendActionResult(
            plan_id=plan.plan_id,
            public_events=tuple(public_events),
            trusted_actions=tuple(actions),
            pre_environment_hash=pre_hash,
            post_environment_hash=post_hash,
            trace_hash=stable_digest(
                {
                    "calls": [call.to_dict() for call in plan.calls],
                    "statuses": [action.status for action in actions],
                }
            ),
            monitor=monitor,
            staged_effect_ids=staged,
            committed_effect_ids=committed,
            rejected_effect_ids=rejected,
            errors=tuple(errors),
            raw_trace=tuple(traces),
            output_text=output_text,
        )

    def execute_probe(
        self,
        environment: EnvironmentHandle,
        theta: str,
        plan_id: str,
        *,
        seed: int = 0,
    ) -> BackendActionResult:
        return self._execute(environment, theta, plan_id, seed=seed, mode=_PROBE)

    def execute_speculative(
        self, environment: EnvironmentHandle, plan_id: str, *, seed: int = 0
    ) -> BackendActionResult:
        return self._execute(
            environment, "public", plan_id, seed=seed, mode=_SPECULATIVE
        )

    def execute_final(
        self,
        environment: EnvironmentHandle,
        theta: str,
        plan_id: str,
        *,
        seed: int = 0,
    ) -> BackendActionResult:
        return self._execute(environment, theta, plan_id, seed=seed, mode=_FINAL)

    def grade(
        self, environment: EnvironmentHandle, result: BackendActionResult
    ) -> BackendGrades:
        environment.ensure_active()
        return grade_agentdojo(
            user_task=self.user_task,
            injection_task=self.injection_task,
            output_text=result.output_text,
            pre_environment=environment.pre_environment,
            post_environment=environment.environment,
            traces=result.raw_trace or (),
            compat=self._compat,
        )

    def false_negative_plan_ids(self, theta: str) -> tuple[str, ...]:
        return tuple(
            plan.plan_id
            for plan in self._trusted_plans.values()
            if plan.ground_truth_label == "prohibited"
            and self.evaluate_guard(theta, plan.plan_id).decision == "allow"
        )

    def retire_environment(self, environment: EnvironmentHandle) -> None:
        environment.retire()


__all__ = ["AgentDojoBackend"]
