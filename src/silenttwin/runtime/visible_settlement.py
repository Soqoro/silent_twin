"""Transactional staging baseline whose settlement is visible."""

from silenttwin.runtime.base import Runtime


class VisibleSettlementRuntime(Runtime):
    name = "visible_settlement"

    def _before_retirement(self, session) -> None:
        # Deliberately leaky baseline.  Avoid private field names, but expose
        # accept/reject outcomes to the protected agent before retirement.
        outcomes = [
            {
                "action_id": record.action_id,
                "outcome": "accepted" if record.monitor_decision.value == "allow" else "rejected",
            }
            for record in session.private_records
        ]
        session.receive_settlement({"outcomes": outcomes, "complete": True})


VisibleSettlement = VisibleSettlementRuntime

__all__ = ["VisibleSettlement", "VisibleSettlementRuntime"]
