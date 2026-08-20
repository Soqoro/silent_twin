"""ARM-style visible action-result tracking baseline."""

from silenttwin.runtime.base import Runtime
from silenttwin.schemas import ActionProposal, MonitorDecision, ToolResult, TruePolicyLabel, VisibleEvent


class ARMStyleRuntime(Runtime):
    name = "arm_style"

    def visible_event(
        self,
        session,
        action: ActionProposal,
        monitor_decision: MonitorDecision,
        true_policy_label: TruePolicyLabel,
        temporary_result: ToolResult,
        sequence: int,
    ) -> VisibleEvent:
        disposition = "tracked" if monitor_decision is MonitorDecision.ALLOW else "quarantined"
        return VisibleEvent(
            event_type="arm_tracking",
            sequence=sequence,
            action_id=action.action_id,
            tool_name=action.tool_name,
            status=disposition,
            value={"tracking_state": disposition},
            receipt={"id": f"arm-{temporary_result.identifier}", "temporary": True},
            identifier=temporary_result.identifier,
            timing_bucket=temporary_result.timing_bucket,
        )


ArmStyleRuntime = ARMStyleRuntime
ARMStyle = ARMStyleRuntime

__all__ = ["ARMStyle", "ARMStyleRuntime", "ArmStyleRuntime"]
