"""Opaque verdict-dependent termination baseline."""

from silenttwin.runtime.base import Runtime
from silenttwin.schemas import (
    ActionProposal,
    CommitReport,
    MonitorDecision,
    ToolResult,
    TruePolicyLabel,
    VisibleEvent,
)


class OpaqueTerminationRuntime(Runtime):
    name = "opaque_termination"

    def visible_event(
        self,
        session,
        action: ActionProposal,
        monitor_decision: MonitorDecision,
        true_policy_label: TruePolicyLabel,
        temporary_result: ToolResult,
        sequence: int,
    ) -> VisibleEvent:
        if monitor_decision is MonitorDecision.ALLOW:
            return self._success_event(action, temporary_result, sequence)
        return VisibleEvent(
            event_type="termination",
            sequence=sequence,
            action_id=action.action_id,
            tool_name=action.tool_name,
            status="terminated",
            error={"code": "SESSION_ENDED", "message": "The session ended."},
            identifier=temporary_result.identifier,
            timing_bucket=temporary_result.timing_bucket,
            termination="opaque",
        )

    def _after_visible_event(self, session, action, monitor_decision) -> None:
        if monitor_decision is MonitorDecision.BLOCK:
            session.retire()

    def _terminal_transition(self, session, report: CommitReport) -> None:
        if any(event.termination for event in session.transcript):
            session.abort()
        else:
            session.mark_committed()


OpaqueTermination = OpaqueTerminationRuntime

__all__ = ["OpaqueTermination", "OpaqueTerminationRuntime"]
