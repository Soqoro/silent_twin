"""Runtime methods and runtime factory."""

from __future__ import annotations

from typing import Any

from silenttwin.runtime.arm_style import ARMStyle, ARMStyleRuntime, ArmStyleRuntime
from silenttwin.runtime.base import BaseRuntime, Runtime, RuntimeFinalization
from silenttwin.runtime.binary_denial import BinaryDenial, BinaryDenialRuntime
from silenttwin.runtime.controller import (
    AtomicityError,
    CommitError,
    DependencyAwareController,
    FinalController,
    InadmissibleEffectError,
    InvalidCommitTokenError,
)
from silenttwin.runtime.dependency_graph import (
    DependencyClosure,
    DependencyCycleError,
    DependencyError,
    DependencyGraph,
    MissingDependencyError,
)
from silenttwin.runtime.detailed_refusal import DetailedRefusal, DetailedRefusalRuntime
from silenttwin.runtime.generic_failure import GenericFailure, GenericFailureRuntime
from silenttwin.runtime.opaque_termination import OpaqueTermination, OpaqueTerminationRuntime
from silenttwin.runtime.randomized_feedback import RandomizedFeedback, RandomizedFeedbackRuntime
from silenttwin.runtime.retirement import (
    ProtectedSession,
    RetirementManager,
    SessionContext,
    SessionLifecycleError,
    SessionRetiredError,
    TrustedReporter,
)
from silenttwin.runtime.silenttwin import (
    ExactSilentTwinRuntime,
    SILENTTWIN_ABLATIONS,
    SilentTwin,
    SilentTwinRuntime,
)
from silenttwin.runtime.visible_settlement import VisibleSettlement, VisibleSettlementRuntime


RUNTIME_CLASSES: dict[str, type[Runtime]] = {
    "detailed_refusal": DetailedRefusalRuntime,
    "binary_denial": BinaryDenialRuntime,
    "generic_failure": GenericFailureRuntime,
    "visible_settlement": VisibleSettlementRuntime,
    "opaque_termination": OpaqueTerminationRuntime,
    "randomized_feedback": RandomizedFeedbackRuntime,
    "arm_style": ARMStyleRuntime,
    "silenttwin": SilentTwinRuntime,
}


def get_runtime(name: str = "silenttwin", **kwargs: Any) -> Runtime:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"silent_twin", "silenttwin_exact", "exact_silenttwin", "full_silenttwin"}:
        normalized = "silenttwin"
    if normalized.startswith("silenttwin:"):
        return SilentTwinRuntime(variant=normalized.split(":", 1)[1], **kwargs)
    if normalized.startswith("silenttwin_") and normalized != "silenttwin":
        return SilentTwinRuntime(variant=normalized.removeprefix("silenttwin_"), **kwargs)
    try:
        runtime_class = RUNTIME_CLASSES[normalized]
    except KeyError as exc:
        if normalized in SILENTTWIN_ABLATIONS:
            return SilentTwinRuntime(variant=normalized, **kwargs)
        names = sorted(set(RUNTIME_CLASSES) | set(SILENTTWIN_ABLATIONS))
        raise ValueError(f"unknown runtime {name!r}; choose one of: {', '.join(names)}") from exc
    return runtime_class(**kwargs)


make_runtime = get_runtime
runtime_for_name = get_runtime


__all__ = [
    "ARMStyle",
    "ARMStyleRuntime",
    "ArmStyleRuntime",
    "AtomicityError",
    "BaseRuntime",
    "BinaryDenial",
    "BinaryDenialRuntime",
    "CommitError",
    "DependencyAwareController",
    "DependencyClosure",
    "DependencyCycleError",
    "DependencyError",
    "DependencyGraph",
    "DetailedRefusal",
    "DetailedRefusalRuntime",
    "ExactSilentTwinRuntime",
    "FinalController",
    "GenericFailure",
    "GenericFailureRuntime",
    "InadmissibleEffectError",
    "InvalidCommitTokenError",
    "MissingDependencyError",
    "OpaqueTermination",
    "OpaqueTerminationRuntime",
    "ProtectedSession",
    "RUNTIME_CLASSES",
    "RandomizedFeedback",
    "RandomizedFeedbackRuntime",
    "RetirementManager",
    "Runtime",
    "RuntimeFinalization",
    "SILENTTWIN_ABLATIONS",
    "SessionContext",
    "SessionLifecycleError",
    "SessionRetiredError",
    "SilentTwin",
    "SilentTwinRuntime",
    "TrustedReporter",
    "VisibleSettlement",
    "VisibleSettlementRuntime",
    "get_runtime",
    "make_runtime",
    "runtime_for_name",
]
