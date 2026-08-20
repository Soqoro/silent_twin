"""Binary allow/deny feedback baseline."""

from silenttwin.runtime.base import Runtime
from silenttwin.schemas import ActionProposal, MonitorDecision, ToolResult, TruePolicyLabel, VisibleEvent


class BinaryDenialRuntime(Runtime):
    name = "binary_denial"

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
            status="denied",
            error={"code": "DENIED", "message": "Action denied."},
            identifier=temporary_result.identifier,
            timing_bucket=temporary_result.timing_bucket,
        )


BinaryDenial = BinaryDenialRuntime

__all__ = ["BinaryDenial", "BinaryDenialRuntime"]
