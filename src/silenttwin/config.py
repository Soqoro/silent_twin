"""Validated, hashable experiment configuration.

Output locations and overwrite policy intentionally do not participate in the
configuration hash.  The hash describes the scientific computation, so moving
an otherwise identical run does not change its identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "silenttwin.result.v1"
MANIFEST_SCHEMA_VERSION = "silenttwin.manifest.v1"
DEFAULT_NUM_SAMPLES = 16

EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5")
TIERS = ("tier1", "tier2")
WORLD_SUITES = ("email", "files", "calendar", "payments")
RUNTIMES = (
    "detailed_refusal",
    "binary_denial",
    "generic_failure",
    "visible_settlement",
    "opaque_termination",
    "randomized_feedback",
    "arm_style",
    "silenttwin",
    # Selected degraded variants used directly in Experiment 3.
    "silenttwin_timing_leak",
    "silenttwin_identifier_leak",
    "silenttwin_decision_dependent_twin_mismatch",
)
ATTACKERS = ("bayesian", "black_box", "mock_llm", "random")
E2_CONDITIONS = ("adaptive", "random", "oracle", "shuffled")
E4_WORKFLOWS = ("independent", "rejection_dependent", "atomic")
E5_ABLATIONS = (
    "visible_settlement",
    "timing_leak",
    "identifier_leak",
    "error_schema_leak",
    "verdict_dependent_termination",
    "callback_leak",
    "later_world_read",
    "retained_memory",
    "retained_kv_cache",
    "incomplete_data_dependencies",
    "incomplete_control_dependencies",
    "missing_atomic_group",
    "public_fidelity_degradation",
    "decision_dependent_twin_mismatch",
)


def canonical_json(value: Any) -> str:
    """Serialize a JSON value in the canonical form used for all hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment: str
    tier: str = "tier1"
    world_suite: str = "email"
    runtime: str = "silenttwin"
    attacker: str = "bayesian"
    query_budget: int = 0
    seed: int = 42
    num_samples: int = DEFAULT_NUM_SAMPLES
    output_dir: Path = Path("outputs/silenttwin")
    condition: str | None = None
    workflow: str | None = None
    ablation: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.num_samples == -1:
            object.__setattr__(self, "num_samples", DEFAULT_NUM_SAMPLES)
        self.validate()

    def validate(self) -> None:
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"unknown experiment {self.experiment!r}; expected one of {EXPERIMENTS}")
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; expected one of {TIERS}")
        if self.world_suite not in WORLD_SUITES:
            raise ValueError(
                f"unknown world suite {self.world_suite!r}; expected one of {WORLD_SUITES}"
            )
        if self.runtime not in RUNTIMES:
            raise ValueError(f"unknown runtime {self.runtime!r}; expected one of {RUNTIMES}")
        if self.attacker not in ATTACKERS:
            raise ValueError(f"unknown attacker {self.attacker!r}; expected one of {ATTACKERS}")
        if self.query_budget < 0:
            raise ValueError("query budget must be non-negative")
        if self.num_samples <= 0:
            raise ValueError("num samples must be positive or -1")
        if self.experiment == "e2":
            condition = self.condition or "adaptive"
            object.__setattr__(self, "condition", condition)
            if condition not in E2_CONDITIONS:
                raise ValueError(f"unknown E2 condition {condition!r}; expected one of {E2_CONDITIONS}")
        elif self.condition is not None:
            raise ValueError("--condition is only valid for e2")
        if self.experiment == "e4":
            workflow = self.workflow or "independent"
            object.__setattr__(self, "workflow", workflow)
            if workflow not in E4_WORKFLOWS:
                raise ValueError(f"unknown E4 workflow {workflow!r}; expected one of {E4_WORKFLOWS}")
        elif self.workflow is not None:
            raise ValueError("--workflow is only valid for e4")
        if self.experiment == "e5":
            # ``none`` is useful as an exact-SilentTwin aggregation reference,
            # although the shell grid deliberately enumerates only ablations.
            ablation = self.ablation or "none"
            object.__setattr__(self, "ablation", ablation)
            if ablation != "none" and ablation not in E5_ABLATIONS:
                raise ValueError(f"unknown E5 ablation {ablation!r}; expected one of {E5_ABLATIONS}")
        elif self.ablation is not None:
            raise ValueError("--ablation is only valid for e5")
        if self.tier == "tier2":
            raise ValueError(
                "Tier 2 is an extension interface only; configure a model/workflow adapter before running"
            )

    def scientific_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("output_dir")
        data.pop("overwrite")
        return data

    @property
    def configuration_hash(self) -> str:
        return stable_hash(self.scientific_dict())

    def as_manifest_config(self) -> dict[str, Any]:
        return self.scientific_dict()
