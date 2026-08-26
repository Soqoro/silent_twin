"""AgentDojo-specific orchestration and analysis.

The modules imported here are deliberately dependency-free.  The compatibility
and execution layers import the pinned :mod:`agentdojo` package lazily; grid
inspection and aggregation never do.
"""

from .config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
    AgentDojoExperimentConfig,
)
from .compat import AgentDojoCompatibilityError, assert_compatible, load_suite
from .catalog import build_catalog
from .splits import build_split_manifest

__all__ = [
    "AGENTDOJO_BENCHMARK_VERSION",
    "AGENTDOJO_PACKAGE_VERSION",
    "AGENTDOJO_SOURCE_REVISION",
    "AGENTDOJO_SUITES",
    "AgentDojoCompatibilityError",
    "AgentDojoExperimentConfig",
    "assert_compatible",
    "build_catalog",
    "build_split_manifest",
    "load_suite",
]
