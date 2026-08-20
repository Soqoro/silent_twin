"""Verdict-correlated generic-failure baseline."""

from silenttwin.runtime.base import Runtime
from silenttwin.schemas import ActionProposal, MonitorDecision, ToolResult, TruePolicyLabel, VisibleEvent


class GenericFailureRuntime(Runtime):
    name = "generic_failure"

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
            event_type="tool_result",
            sequence=sequence,
            action_id=action.action_id,
            tool_name=action.tool_name,
            status="error",
            error={"code": "GENERIC_FAILURE", "message": "The operation failed."},
            identifier=temporary_result.identifier,
            timing_bucket=temporary_result.timing_bucket,
        )


GenericFailure = GenericFailureRuntime

__all__ = ["GenericFailure", "GenericFailureRuntime"]
