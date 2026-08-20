"""Protected-session lifecycle and trusted-report delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from silenttwin.schemas import (
    AgentVisibleTranscript,
    PrivateEvaluationRecord,
    SessionState,
    StagedEffect,
    TrustedReport,
    VisibleEvent,
)


class SessionLifecycleError(RuntimeError):
    pass


class SessionRetiredError(SessionLifecycleError):
    pass


@dataclass
class ProtectedSession:
    """Agent context with an enforced ACTIVE -> RETIRED -> terminal lifecycle."""

    session_id: str
    world: Any
    transcript: AgentVisibleTranscript = field(default_factory=AgentVisibleTranscript)
    state: SessionState = SessionState.ACTIVE
    staged_effects: list[StagedEffect] = field(default_factory=list, repr=False)
    private_records: list[PrivateEvaluationRecord] = field(default_factory=list, repr=False)
    temporary_environment: Any = field(default=None, repr=False)
    tool_invoker: Callable[[str, Mapping[str, Any]], Any] | None = field(default=None, repr=False)
    real_world_reader: Callable[[str], Any] | None = field(default=None, repr=False)

    def ensure_active(self, operation: str = "operation") -> None:
        if self.state is not SessionState.ACTIVE:
            raise SessionRetiredError(
                f"cannot perform {operation}: protected session {self.session_id} is {self.state.value}"
            )

    assert_active = ensure_active

    @property
    def active(self) -> bool:
        return self.state is SessionState.ACTIVE

    @property
    def retired(self) -> bool:
        return self.state is not SessionState.ACTIVE

    def append_visible(self, event: VisibleEvent) -> VisibleEvent:
        self.ensure_active("append a visible event")
        return self.transcript.append(event)

    def invoke_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        self.ensure_active("invoke a tool")
        if self.tool_invoker is None:
            raise RuntimeError("no tool invoker is bound to this session")
        return self.tool_invoker(tool_name, arguments)

    call_tool = invoke_tool

    def receive_callback(self, callback: Mapping[str, Any]) -> VisibleEvent:
        self.ensure_active("receive a callback")
        event = VisibleEvent(
            event_type="callback",
            sequence=len(self.transcript),
            status="ok",
            callback=callback,
        )
        return self.append_visible(event)

    deliver_callback = receive_callback

    def read_real_world(self, key: str) -> Any:
        self.ensure_active("read the real world")
        if self.real_world_reader is None:
            raise RuntimeError("no real-world reader is bound to this session")
        return self.real_world_reader(key)

    real_world_read = read_real_world

    def receive_settlement(self, message: Mapping[str, Any]) -> VisibleEvent:
        self.ensure_active("receive settlement")
        event = VisibleEvent(
            event_type="settlement",
            sequence=len(self.transcript),
            status="settled",
            settlement_message=message,
        )
        return self.append_visible(event)

    deliver_settlement = receive_settlement

    def retain_memory(self, memory: Mapping[str, Any]) -> VisibleEvent:
        self.ensure_active("update visible memory")
        event = VisibleEvent(
            event_type="memory",
            sequence=len(self.transcript),
            status="ok",
            visible_memory=memory,
        )
        return self.append_visible(event)

    def receive_trusted_repair(self, repair: Mapping[str, Any]) -> VisibleEvent:
        """Compatibility hook whose lifecycle check prevents post-retirement repair."""

        self.ensure_active("receive trusted repair output")
        return self.retain_memory({"trusted_repair": dict(repair)})

    deliver_trusted_repair = receive_trusted_repair

    def retire(self) -> None:
        if self.state is not SessionState.ACTIVE:
            raise SessionLifecycleError(
                f"session {self.session_id} cannot transition {self.state.value} -> retired"
            )
        self.state = SessionState.RETIRED
        self.transcript.seal()
        # Revocation is explicit rather than relying only on state checks.
        self.tool_invoker = None
        self.real_world_reader = None

    def mark_committed(self) -> None:
        if self.state is not SessionState.RETIRED:
            raise SessionLifecycleError(
                f"session {self.session_id} must be retired before it can be committed"
            )
        self.state = SessionState.COMMITTED

    def abort(self) -> None:
        if self.state is SessionState.ACTIVE:
            self.retire()
        if self.state is not SessionState.RETIRED:
            raise SessionLifecycleError(
                f"session {self.session_id} cannot transition {self.state.value} -> aborted"
            )
        self.state = SessionState.ABORTED


SessionContext = ProtectedSession


class RetirementManager:
    def retire(self, session: ProtectedSession) -> None:
        session.retire()

    def commit(self, session: ProtectedSession) -> None:
        session.mark_committed()

    def abort(self, session: ProtectedSession) -> None:
        session.abort()


class TrustedReporter:
    """A distinct trusted sink; reports never enter the protected transcript."""

    def __init__(self, sink: Callable[[TrustedReport], Any] | None = None) -> None:
        self._sink = sink
        self.reports: list[TrustedReport] = []

    def deliver(self, report: TrustedReport) -> None:
        self.reports.append(report)
        if self._sink is not None:
            self._sink(report)

    report = deliver


__all__ = [
    "ProtectedSession",
    "RetirementManager",
    "SessionContext",
    "SessionLifecycleError",
    "SessionRetiredError",
    "TrustedReporter",
]
