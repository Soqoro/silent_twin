"""Guardrail-independent deterministic temporary tool environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from silenttwin.schemas import ActionProposal, ToolResult, stable_digest


@dataclass
class TemporaryEnvironment:
    """Simulates tool behavior without producing real-world mutations.

    Its output is a pure function of public instance data, the public action,
    and sequence number.  It never receives a monitor or hidden state.
    """

    public_instance_hash: str
    public_environment: Mapping[str, Any] = field(default_factory=dict)
    fidelity: str = "exact"
    fixed_timing_bucket: str = "fixed"
    _values: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _identifier(self, action: ActionProposal, sequence: int) -> str:
        digest = stable_digest(
            {
                "public_instance_hash": self.public_instance_hash,
                "sequence": sequence,
                "action": action.to_dict(),
            }
        )
        return f"tmp-{digest[:16]}"

    def invoke(self, action: ActionProposal, *, sequence: int = 0) -> ToolResult:
        identifier = self._identifier(action, sequence)
        if self.fidelity == "degraded":
            # Deliberately state-independent inaccuracy for Experiment 5: it
            # reduces utility but does not reveal theta.
            return ToolResult(
                ok=False,
                value=None,
                error_code="TEMPORARY_UNAVAILABLE",
                error_message="temporary simulator unavailable",
                identifier=identifier,
                timing_bucket=self.fixed_timing_bucket,
                effectful=action.effectful,
            )
        value = {
            "accepted": True,
            "temporary": True,
            "tool": action.tool_name,
            "action": action.action_id,
        }
        receipt_id = f"receipt-{identifier[4:]}" if action.effectful else None
        if action.effectful:
            self._values[action.action_id] = value
        return ToolResult(
            ok=True,
            value=value,
            receipt_id=receipt_id,
            identifier=identifier,
            timing_bucket=self.fixed_timing_bucket,
            retries=0,
            effectful=action.effectful,
        )

    simulate = invoke

    def read(self, key: str) -> Any:
        if key in self._values:
            return self._values[key]
        return self.public_environment.get(key)

    def clear(self) -> None:
        self._values.clear()


TwinEnvironment = TemporaryEnvironment


__all__ = ["TemporaryEnvironment", "TwinEnvironment"]
