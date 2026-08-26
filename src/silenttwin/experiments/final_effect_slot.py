"""One-shot final-effect selection invariant for Experiment 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from silenttwin.schemas import MonitorDecision, RuntimeStep


class FinalSlotError(RuntimeError):
    pass


class FinalSlotState(str, Enum):
    OPEN = "open"
    SELECTED = "selected"
    SEALED = "sealed"


@dataclass(slots=True)
class FinalEffectSlot:
    """Locks one public exploit selection before its private verdict exists."""

    allowed_action_ids: frozenset[str]
    state: FinalSlotState = FinalSlotState.OPEN
    selected_action_id: str | None = None
    requested_action_id: str | None = None
    accepted_effect_id: str | None = None
    attempt_count: int = 0
    execution_failure: str | None = None
    _bound_step: RuntimeStep | None = field(default=None, init=False, repr=False)

    def __init__(self, allowed_action_ids: Iterable[str]) -> None:
        allowed = frozenset(str(item) for item in allowed_action_ids)
        if not allowed:
            raise ValueError("final effect slot requires at least one public exploit")
        self.allowed_action_ids = allowed
        self.state = FinalSlotState.OPEN
        self.selected_action_id = None
        self.requested_action_id = None
        self.accepted_effect_id = None
        self.attempt_count = 0
        self.execution_failure = None
        self._bound_step = None

    def select_once(self, action_id: str) -> str:
        if self.state is not FinalSlotState.OPEN or self.attempt_count:
            raise FinalSlotError("final exploit selection is already locked; fallback is forbidden")
        if action_id not in self.allowed_action_ids:
            failure = f"action {action_id!r} is not in the public final exploit set"
            # Calling select_once is itself the sole request.  An invalid ID
            # must consume and seal that request before raising, otherwise a
            # caller could catch the error and promote a different candidate.
            self.requested_action_id = action_id
            self.attempt_count = 1
            self.execution_failure = failure
            self.state = FinalSlotState.SEALED
            raise FinalSlotError(failure)
        self.requested_action_id = action_id
        self.selected_action_id = action_id
        self.attempt_count = 1
        self.state = FinalSlotState.SELECTED
        return action_id

    def bind_runtime_step(self, step: RuntimeStep) -> None:
        if self.state is not FinalSlotState.SELECTED or self.selected_action_id is None:
            raise FinalSlotError("select an exploit before binding its runtime result")
        if self._bound_step is not None:
            raise FinalSlotError("the final effect slot already has a runtime result")
        if step.private_record.action_id != self.selected_action_id:
            raise FinalSlotError("runtime result does not correspond to the locked exploit")
        self._bound_step = step
        if (
            step.private_record.monitor_decision is MonitorDecision.ALLOW
            and step.staged_effect is not None
        ):
            self.accepted_effect_id = step.staged_effect.effect_id
        # A blocked action deliberately leaves accepted_effect_id empty.  Its
        # rejected staged record remains available to the trusted evaluator.
        self.state = FinalSlotState.SEALED

    def reject_invalid_attempt(self, failure: str, requested_action_id: str | None = None) -> None:
        """Lock the one attempt without substituting another candidate."""

        if self.state is not FinalSlotState.OPEN or self.attempt_count:
            raise FinalSlotError("final exploit selection is already locked; fallback is forbidden")
        if not failure:
            raise ValueError("an invalid final attempt needs a failure reason")
        self.requested_action_id = requested_action_id
        self.attempt_count = 1
        self.execution_failure = failure
        self.state = FinalSlotState.SEALED

    def seal_unexecuted(self, failure: str) -> None:
        if self.state is not FinalSlotState.SELECTED:
            raise FinalSlotError("only a selected slot can be sealed without execution")
        if not failure:
            raise ValueError("an unexecuted final selection needs a failure reason")
        self.execution_failure = failure
        self.state = FinalSlotState.SEALED

    @property
    def bound_step(self) -> RuntimeStep | None:
        """Trusted-only runtime record; never pass the slot to an attacker."""

        return self._bound_step

    @property
    def empty(self) -> bool:
        return self.accepted_effect_id is None

    def to_trusted_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "requested_action_id": self.requested_action_id,
            "selected_action_id": self.selected_action_id,
            "accepted_effect_id": self.accepted_effect_id,
            "attempt_count": self.attempt_count,
            "execution_failure": self.execution_failure,
        }


__all__ = [
    "FinalEffectSlot",
    "FinalSlotError",
    "FinalSlotState",
]
