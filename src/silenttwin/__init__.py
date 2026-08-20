"""SilentTwin: guardrail-feedback privacy experiments.

The top-level package exposes the small Tier-1 data plane.  Higher-level run
configuration, experiment orchestration, metrics, and I/O remain in their
respective modules.
"""

from silenttwin.attackers import (
    AttackRunResult,
    Attacker,
    BayesianAttacker,
    BlackBoxAttacker,
    LLMAttacker,
    MockLLMAttacker,
    MockModelClient,
    ModelClient,
    RandomAttacker,
    make_attacker,
)
from silenttwin.policy import Monitor, SyntheticMonitor, TagTruePolicy, TruePolicy, make_monitor, make_true_policy
from silenttwin.runtime import (
    DependencyAwareController,
    DependencyGraph,
    ProtectedSession,
    Runtime,
    SILENTTWIN_ABLATIONS,
    SessionRetiredError,
    SilentTwinRuntime,
    get_runtime,
    make_runtime,
)
from silenttwin.schemas import (
    SCHEMA_VERSION,
    ActionProposal,
    AgentVisibleTranscript,
    CommitReport,
    CommitToken,
    EffectStatus,
    MonitorDecision,
    PrivateEvaluationRecord,
    SessionState,
    StagedEffect,
    TruePolicyLabel,
    TrustedReport,
    VisibleEvent,
    canonical_json,
    stable_digest,
)
from silenttwin.worlds import FiniteStateWorld, WorldPair, build_world_pair, make_world_pair

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "ActionProposal",
    "AgentVisibleTranscript",
    "AttackRunResult",
    "Attacker",
    "BayesianAttacker",
    "BlackBoxAttacker",
    "CommitReport",
    "CommitToken",
    "DependencyAwareController",
    "DependencyGraph",
    "EffectStatus",
    "FiniteStateWorld",
    "LLMAttacker",
    "MockLLMAttacker",
    "MockModelClient",
    "ModelClient",
    "Monitor",
    "MonitorDecision",
    "PrivateEvaluationRecord",
    "ProtectedSession",
    "RandomAttacker",
    "Runtime",
    "SILENTTWIN_ABLATIONS",
    "SessionRetiredError",
    "SessionState",
    "SilentTwinRuntime",
    "StagedEffect",
    "SyntheticMonitor",
    "TagTruePolicy",
    "TruePolicy",
    "TruePolicyLabel",
    "TrustedReport",
    "VisibleEvent",
    "WorldPair",
    "build_world_pair",
    "canonical_json",
    "get_runtime",
    "make_attacker",
    "make_monitor",
    "make_runtime",
    "make_true_policy",
    "make_world_pair",
    "stable_digest",
]
