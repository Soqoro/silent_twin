"""Scientific execution backends.

AgentDojo remains optional: importing this package does not import AgentDojo,
Transformers, PyTorch, or CUDA libraries.
"""

from .base import (
    BackendActionResult,
    BackendEpisode,
    BackendError,
    BackendErrorStage,
    BackendExecutionError,
    BackendGrades,
    BackendIdentity,
    BackendProtocolError,
    EnvironmentHandle,
    EnvironmentRole,
    FinalAttemptOutcome,
    GuardEvaluation,
    PublicPlan,
    PublicScenario,
    TrustedActionRecord,
    TrustedPlan,
)

__all__ = [
    "BackendActionResult",
    "BackendEpisode",
    "BackendError",
    "BackendErrorStage",
    "BackendExecutionError",
    "BackendGrades",
    "BackendIdentity",
    "BackendProtocolError",
    "EnvironmentHandle",
    "EnvironmentRole",
    "FinalAttemptOutcome",
    "GuardEvaluation",
    "PublicPlan",
    "PublicScenario",
    "TrustedActionRecord",
    "TrustedPlan",
    "AgentDojoBackend",
    "FiniteStateBackend",
]


def __getattr__(name: str):
    if name == "AgentDojoBackend":
        from .agentdojo import AgentDojoBackend

        return AgentDojoBackend
    if name == "FiniteStateBackend":
        from .finite_state import FiniteStateBackend

        return FiniteStateBackend
    raise AttributeError(name)
