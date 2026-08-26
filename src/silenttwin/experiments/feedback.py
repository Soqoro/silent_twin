"""Interactive feedback-source bindings for trusted trial execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from silenttwin.runtime.base import Runtime, RuntimeFinalization
from silenttwin.runtime.retirement import ProtectedSession
from silenttwin.schemas import RuntimeStep, SessionState, VisibleEvent
from silenttwin.worlds.finite_state import FiniteStateWorld


class FeedbackKind(str, Enum):
    GENUINE = "genuine"
    SHUFFLED = "shuffled"
    CONSTANT = "constant"

    @classmethod
    def coerce(cls, value: "FeedbackKind | str") -> "FeedbackKind":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {"adaptive": "genuine", "ordinary": "genuine"}
        return cls(aliases.get(normalized, normalized))


@dataclass(frozen=True, slots=True)
class FeedbackBatch:
    action_id: str
    events: tuple[VisibleEvent, ...]
    executed: bool
    failure: str | None = None


class FeedbackSource(ABC):
    """Trusted source that returns only runtime-rendered public observations."""

    kind: FeedbackKind
    source_state: str

    @property
    @abstractmethod
    def active(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def execute_probe(self, action_id: str) -> FeedbackBatch:
        raise NotImplementedError

    @abstractmethod
    def end_probing(self) -> tuple[VisibleEvent, ...]:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> tuple[RuntimeFinalization, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def private_steps(self) -> tuple[RuntimeStep, ...]:
        raise NotImplementedError


class RuntimeFeedbackSource(FeedbackSource):
    def __init__(
        self,
        *,
        kind: FeedbackKind | str,
        source_state: str,
        runtime: Runtime,
        session: ProtectedSession,
    ) -> None:
        selected = FeedbackKind.coerce(kind)
        if selected is FeedbackKind.CONSTANT:
            raise ValueError("use ConstantFeedbackSource for constant feedback")
        self.kind = selected
        self.source_state = source_state
        self.runtime = runtime
        self.session = session
        self._steps: list[RuntimeStep] = []
        self._finalized = False

    @property
    def active(self) -> bool:
        return self.session.active

    def execute_probe(self, action_id: str) -> FeedbackBatch:
        if not self.session.active:
            return FeedbackBatch(
                action_id=action_id,
                events=(),
                executed=False,
                failure="feedback_session_not_active",
            )
        before = len(self.session.transcript)
        step = self.runtime.execute(self.session, action_id)
        self._steps.append(step)
        events = tuple(self.session.transcript.events[before:])
        return FeedbackBatch(action_id=action_id, events=events, executed=True)

    def end_probing(self) -> tuple[VisibleEvent, ...]:
        if not self.session.active:
            return ()
        return self.runtime.end_probing(self.session)

    @property
    def private_steps(self) -> tuple[RuntimeStep, ...]:
        return tuple(self._steps)

    def finalize(self) -> tuple[RuntimeFinalization, ...]:
        if self._finalized or self.session.state in {SessionState.COMMITTED, SessionState.ABORTED}:
            return ()
        self._finalized = True
        return (self.runtime.finalize(self.session),)


class ConstantFeedbackSource(FeedbackSource):
    """Privately evaluates probes while emitting a fixed public response schema.

    A verdict-dependent runtime may retire an internal source session.  The next
    probe then uses a fresh internal source session, ensuring that the number of
    delivered constant responses is itself state independent.
    """

    kind = FeedbackKind.CONSTANT

    def __init__(
        self,
        *,
        source_state: str,
        world: FiniteStateWorld,
        runtime_factory: Callable[[], Runtime],
    ) -> None:
        self.source_state = source_state
        self.world = world
        self.runtime_factory = runtime_factory
        self._bindings: list[tuple[Runtime, ProtectedSession]] = []
        self._steps: list[RuntimeStep] = []
        self._finalized = False

    def _binding(self) -> tuple[Runtime, ProtectedSession]:
        if self._bindings and self._bindings[-1][1].active:
            return self._bindings[-1]
        runtime = self.runtime_factory()
        session = runtime.start_session(self.world)
        self._bindings.append((runtime, session))
        return runtime, session

    @property
    def active(self) -> bool:
        return not self._finalized

    def execute_probe(self, action_id: str) -> FeedbackBatch:
        if self._finalized:
            return FeedbackBatch(
                action_id=action_id,
                events=(),
                executed=False,
                failure="feedback_source_finalized",
            )
        runtime, session = self._binding()
        step = runtime.execute(session, action_id)
        self._steps.append(step)
        action = self.world.action(action_id)
        constant = VisibleEvent(
            event_type="tool_result",
            sequence=0,  # the TrialRunner assigns the delivered sequence
            action_id=action.action_id,
            tool_name=action.tool_name,
            status="constant",
            value={"result": "probe_recorded", "temporary": True},
            identifier="constant",
            timing_bucket="fixed",
            retries=0,
        )
        return FeedbackBatch(action_id=action_id, events=(constant,), executed=True)

    def end_probing(self) -> tuple[VisibleEvent, ...]:
        return ()

    @property
    def private_steps(self) -> tuple[RuntimeStep, ...]:
        return tuple(self._steps)

    def finalize(self) -> tuple[RuntimeFinalization, ...]:
        if self._finalized:
            return ()
        self._finalized = True
        results: list[RuntimeFinalization] = []
        for runtime, session in self._bindings:
            if session.state not in {SessionState.COMMITTED, SessionState.ABORTED}:
                results.append(runtime.finalize(session))
        return tuple(results)


__all__ = [
    "ConstantFeedbackSource",
    "FeedbackBatch",
    "FeedbackKind",
    "FeedbackSource",
    "RuntimeFeedbackSource",
]
