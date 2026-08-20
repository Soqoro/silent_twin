"""Common runtime interface for feedback baselines and SilentTwin."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from silenttwin.runtime.controller import DependencyAwareController
from silenttwin.runtime.retirement import ProtectedSession, TrustedReporter
from silenttwin.schemas import (
    ActionProposal,
    CommitReport,
    MonitorDecision,
    PrivateEvaluationRecord,
    RuntimeStep,
    SessionState,
    StagedEffect,
    ToolResult,
    TruePolicyLabel,
    TrustedReport,
    VisibleEvent,
    stable_digest,
)
from silenttwin.tools.temporary_environment import TemporaryEnvironment
from silenttwin.worlds.finite_state import FiniteStateWorld


@dataclass(frozen=True)
class RuntimeFinalization:
    session: ProtectedSession
    commit_report: CommitReport
    trusted_report: TrustedReport

    @property
    def transcript(self):
        return self.session.transcript

    @property
    def committed_effect_ids(self) -> tuple[str, ...]:
        return self.commit_report.committed_effect_ids

    @property
    def committed_effects(self) -> tuple[StagedEffect, ...]:
        selected = set(self.commit_report.committed_effect_ids)
        return tuple(effect for effect in self.session.staged_effects if effect.effect_id in selected)


class Runtime(ABC):
    """Synchronous runtime interface.

    Subclasses vary only the protected session's visible observations.  Policy
    and monitor evaluation, trusted records, staging, and final control remain
    structurally separate.
    """

    name = "base"

    def __init__(
        self,
        world: FiniteStateWorld | None = None,
        *,
        seed: int = 0,
        trusted_reporter: TrustedReporter | None = None,
        controller_class: type[DependencyAwareController] = DependencyAwareController,
        executor: Any = None,
        **_: Any,
    ) -> None:
        self.world = world
        self.seed = int(seed)
        self.trusted_reporter = trusted_reporter or TrustedReporter()
        self.controller_class = controller_class
        self.executor = executor
        self._session_counter = 0

    def start_session(
        self,
        world: FiniteStateWorld | None = None,
        *,
        session_id: str | None = None,
    ) -> ProtectedSession:
        selected = world or self.world
        if selected is None:
            raise ValueError("a world is required to start a runtime session")
        public_hash = stable_digest(selected.public_projection())
        if session_id is None:
            session_id = f"session-{public_hash[:12]}-{self._session_counter:04d}"
        self._session_counter += 1
        environment = TemporaryEnvironment(
            public_instance_hash=public_hash,
            public_environment=selected.public_environment,
        )
        session = ProtectedSession(
            session_id=session_id,
            world=selected,
            temporary_environment=environment,
        )
        return session

    create_session = start_session

    def _coerce_action(self, session: ProtectedSession, action: ActionProposal | str) -> ActionProposal:
        if isinstance(action, str):
            return session.world.action(action)
        if not isinstance(action, ActionProposal):
            raise TypeError("action must be an ActionProposal or public action ID")
        return action

    def _next_effect_id(self, session: ProtectedSession, action: ActionProposal) -> str:
        existing = {effect.effect_id for effect in session.staged_effects}
        if action.action_id not in existing:
            return action.action_id
        suffix = 2
        while f"{action.action_id}#{suffix}" in existing:
            suffix += 1
        return f"{action.action_id}#{suffix}"

    @staticmethod
    def _success_event(
        action: ActionProposal,
        result: ToolResult,
        sequence: int,
        *,
        status: str = "ok",
        timing_bucket: str | None = None,
        identifier: str | None = None,
        value: Any = None,
    ) -> VisibleEvent:
        visible_value = result.value if value is None else value
        receipt = {"id": result.receipt_id, "temporary": True} if result.receipt_id else None
        return VisibleEvent(
            event_type="tool_result",
            sequence=sequence,
            action_id=action.action_id,
            tool_name=action.tool_name,
            status=status,
            value=visible_value,
            receipt=receipt,
            identifier=identifier if identifier is not None else result.identifier,
            timing_bucket=timing_bucket or result.timing_bucket,
            retries=result.retries,
        )

    def visible_event(
        self,
        session: ProtectedSession,
        action: ActionProposal,
        monitor_decision: MonitorDecision,
        true_policy_label: TruePolicyLabel,
        temporary_result: ToolResult,
        sequence: int,
    ) -> VisibleEvent:
        """Default baseline: detailed decision-correlated refusal."""

        if monitor_decision is MonitorDecision.ALLOW:
            return self._success_event(action, temporary_result, sequence)
        return VisibleEvent(
            event_type="tool_result",
            sequence=sequence,
            action_id=action.action_id,
            tool_name=action.tool_name,
            status="refused",
            error={
                "code": "POLICY_BLOCKED",
                "message": f"Action category {action.policy_tag} was rejected by the guardrail.",
            },
            identifier=temporary_result.identifier,
            timing_bucket=temporary_result.timing_bucket,
        )

    def _after_visible_event(
        self,
        session: ProtectedSession,
        action: ActionProposal,
        monitor_decision: MonitorDecision,
    ) -> None:
        return None

    def execute(self, session: ProtectedSession, action: ActionProposal | str) -> RuntimeStep:
        session.ensure_active("invoke the runtime")
        proposal = self._coerce_action(session, action)
        world = session.world
        monitor_decision = MonitorDecision.coerce(world.monitor.decide(proposal))
        true_label = TruePolicyLabel.coerce(world.true_policy.evaluate(proposal))
        sequence = len(session.transcript)

        # Simulation has no access to either verdict or private state.
        temporary_result = session.temporary_environment.invoke(proposal, sequence=sequence)
        effect: StagedEffect | None = None
        if proposal.effectful:
            effect = StagedEffect.from_action(
                proposal,
                effect_id=self._next_effect_id(session, proposal),
                monitor_decision=monitor_decision,
                true_policy_label=true_label,
                policy_epoch=world.policy_epoch,
            )
            session.staged_effects.append(effect)
        private = PrivateEvaluationRecord(
            action_id=proposal.action_id,
            private_state=world.private_state,
            monitor_decision=monitor_decision,
            true_policy_label=true_label,
            policy_epoch=world.policy_epoch,
            effect_id=effect.effect_id if effect else None,
            details={"runtime": self.name},
        )
        session.private_records.append(private)
        event = self.visible_event(
            session,
            proposal,
            monitor_decision,
            true_label,
            temporary_result,
            sequence,
        )
        session.append_visible(event)
        self._after_visible_event(session, proposal, monitor_decision)
        return RuntimeStep(event=event, private_record=private, staged_effect=effect)

    execute_action = execute
    run_action = execute
    step = execute

    def run_actions(
        self,
        session: ProtectedSession,
        actions: Iterable[ActionProposal | str],
    ) -> tuple[RuntimeStep, ...]:
        results: list[RuntimeStep] = []
        for action in actions:
            if not session.active:
                break
            results.append(self.execute(session, action))
        return tuple(results)

    def run(
        self,
        actions: Iterable[ActionProposal | str] | ActionProposal | str,
        *,
        world: FiniteStateWorld | None = None,
        finalize: bool = False,
    ) -> tuple[ProtectedSession, tuple[RuntimeStep, ...], RuntimeFinalization | None]:
        session = self.start_session(world)
        if isinstance(actions, (ActionProposal, str)):
            selected: Iterable[ActionProposal | str] = (actions,)
        else:
            selected = actions
        steps = self.run_actions(session, selected)
        result = self.finalize(session) if finalize else None
        return session, steps, result

    def _before_retirement(self, session: ProtectedSession) -> None:
        return None

    def _controller_kwargs(self) -> dict[str, Any]:
        return {}

    def _terminal_transition(self, session: ProtectedSession, report: CommitReport) -> None:
        session.mark_committed()

    def finalize(self, session: ProtectedSession) -> RuntimeFinalization:
        if session.state is SessionState.ACTIVE:
            self._before_retirement(session)
            # The protected session is irreversibly retired before the final
            # controller evaluates tokens or invokes any external executor.
            session.retire()
        elif session.state is not SessionState.RETIRED:
            raise RuntimeError(f"session is already terminal: {session.state.value}")

        controller = self.controller_class(
            session.staged_effects,
            expected_policy_epoch=session.world.policy_epoch,
            executor=self.executor,
            **self._controller_kwargs(),
        )
        commit_report = controller.commit_all()
        self._terminal_transition(session, commit_report)
        trusted_report = TrustedReport(
            session_id=session.session_id,
            final_state=session.state,
            commit_report=commit_report,
            private_records=tuple(session.private_records),
            staged_effects=tuple(effect.trusted_dict() for effect in session.staged_effects),
        )
        self.trusted_reporter.deliver(trusted_report)
        return RuntimeFinalization(session, commit_report, trusted_report)

    settle = finalize


BaseRuntime = Runtime


__all__ = ["BaseRuntime", "Runtime", "RuntimeFinalization"]
