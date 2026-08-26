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
import re
from typing import Any


SCHEMA_VERSION = "silenttwin.result.v2"
MANIFEST_SCHEMA_VERSION = "silenttwin.manifest.v2"
DEFAULT_NUM_SAMPLES = 16
DEFAULT_DATASET_REVISION = "silenttwin-tier1-v1"
DEFAULT_ANALYSIS_REVISION = "silenttwin-analysis-v1"
_IMMUTABLE_MODEL_REVISION = re.compile(
    r"^(?:[0-9a-fA-F]{40,64}|sha256:[0-9a-fA-F]{64})$"
)

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
ATTACKERS = ("bayesian", "black_box", "mock_llm", "random", "majority", "llm")
E2_CONDITIONS = ("no_probe", "adaptive", "genuine", "random", "oracle", "shuffled")
PAIR_FAMILIES = ("authorization", "monitor_blind_spot")
PAIR_FAMILY_ALIASES = {"blind_spot": "monitor_blind_spot"}
DATASET_SPLITS = ("train", "development", "test")
FEEDBACK_SOURCES = ("genuine", "shuffled", "constant")
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


def is_immutable_model_revision(value: str | None) -> bool:
    """Return whether a revision is an immutable commit or tree digest."""

    return bool(value and _IMMUTABLE_MODEL_REVISION.fullmatch(value))


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
    sample_start: int = 0
    output_dir: Path = Path("outputs/silenttwin")
    pair_family: str = "monitor_blind_spot"
    template_id: str | None = None
    dataset_split: str = "development"
    dataset_revision: str = DEFAULT_DATASET_REVISION
    analysis_revision: str = DEFAULT_ANALYSIS_REVISION
    feedback_source: str = "genuine"
    condition: str | None = None
    workflow: str | None = None
    ablation: str | None = None
    confidence_threshold: float = 0.9
    model_id: str | None = None
    model_revision: str | None = None
    model_cache_dir: str | None = None
    dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    decoding_seed: int | None = None
    batch_size: int = 1
    grid_hash: str | None = None
    grid_task_id: int | None = None
    shard_id: str | None = None
    pilot_id: str | None = None
    sample_size_freeze_hash: str | None = None
    development_manifest_hash: str | None = None
    frozen_public_instances: int | None = None
    primary_contrast_id: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.num_samples == -1:
            object.__setattr__(self, "num_samples", DEFAULT_NUM_SAMPLES)
        if self.decoding_seed is None:
            object.__setattr__(self, "decoding_seed", self.seed)
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
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.query_budget < 0:
            raise ValueError("query budget must be non-negative")
        if self.num_samples <= 0:
            raise ValueError("num samples must be positive or -1")
        if self.sample_start < 0:
            raise ValueError("sample start must be non-negative")
        if self.pair_family in PAIR_FAMILY_ALIASES:
            object.__setattr__(self, "pair_family", PAIR_FAMILY_ALIASES[self.pair_family])
        if self.pair_family not in PAIR_FAMILIES:
            raise ValueError(
                f"unknown pair family {self.pair_family!r}; expected one of {PAIR_FAMILIES}"
            )
        if self.dataset_split not in DATASET_SPLITS:
            raise ValueError(
                f"unknown dataset split {self.dataset_split!r}; expected one of {DATASET_SPLITS}"
            )
        if self.dataset_revision != DEFAULT_DATASET_REVISION:
            raise ValueError(
                f"unsupported dataset revision {self.dataset_revision!r}; "
                f"the executable registry provides exactly {DEFAULT_DATASET_REVISION!r}"
            )
        if not self.analysis_revision:
            raise ValueError("analysis revision must be non-empty")
        if self.feedback_source not in FEEDBACK_SOURCES:
            raise ValueError(
                f"unknown feedback source {self.feedback_source!r}; expected one of {FEEDBACK_SOURCES}"
            )
        if not 0.5 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in (0.5, 1]")
        if self.attacker == "black_box" and self.dataset_split == "train":
            raise ValueError(
                "black-box evaluation must use held-out development or test templates"
            )
        if self.experiment == "e2":
            condition = self.condition or "adaptive"
            if condition not in E2_CONDITIONS:
                raise ValueError(f"unknown E2 condition {condition!r}; expected one of {E2_CONDITIONS}")
            # ``adaptive`` is the historical name for genuine, target-correlated
            # feedback.  Canonicalizing it avoids two hashes for one treatment.
            condition = "genuine" if condition == "adaptive" else condition
            object.__setattr__(self, "condition", condition)
            if self.pair_family != "monitor_blind_spot":
                raise ValueError("E2 requires monitor-blind-spot pairs")
            if self.num_samples % 4 or self.sample_start % 4:
                raise ValueError(
                    "E2 shards must start and end on complete four-cell target/donor blocks"
                )
            if condition in {"no_probe", "oracle"} and self.query_budget != 0:
                raise ValueError(f"E2 condition {condition!r} requires query budget 0")
            if condition in {"genuine", "shuffled", "random"} and self.query_budget not in {
                4,
                16,
                32,
            }:
                raise ValueError(
                    f"E2 condition {condition!r} requires query budget 4, 16, or 32"
                )
            if (
                self.runtime == "opaque_termination"
                and condition in {"genuine", "random"}
                and self.query_budget > 0
            ):
                raise ValueError(
                    "opaque_termination target feedback can retire before E2's mandatory final action; "
                    "use shuffled feedback or another runtime"
                )
            expected_feedback = "shuffled" if condition == "shuffled" else "genuine"
            # Random changes only final selection. It still receives the same
            # genuine online interaction as the adaptive cell, so the hashed
            # treatment must say genuine rather than retain a caller fiction.
            object.__setattr__(self, "feedback_source", expected_feedback)
        elif self.condition is not None:
            raise ValueError("--condition is only valid for e2")
        if self.experiment == "e1" and self.query_budget not in {0, 4, 16, 32}:
            raise ValueError("E1 query budget must be one of 0, 4, 16, or 32")
        if self.experiment == "e1":
            e1_block = 4 if self.feedback_source == "shuffled" else 2
            if self.num_samples % e1_block or self.sample_start % e1_block:
                raise ValueError(
                    "E1 shards must contain complete balanced state/feedback blocks"
                )
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
        if self.max_new_tokens <= 0:
            raise ValueError("max new tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top-p must be in (0, 1]")
        if self.decoding_seed is None or self.decoding_seed < 0:
            raise ValueError("decoding seed must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.grid_task_id is not None and self.grid_task_id < 0:
            raise ValueError("grid task ID must be non-negative")
        freeze_fields = (
            self.sample_size_freeze_hash,
            self.development_manifest_hash,
            self.frozen_public_instances,
            self.primary_contrast_id,
        )
        if self.dataset_split == "test":
            if any(value is None for value in freeze_fields):
                raise ValueError(
                    "held-out test execution requires a validated sample-size freeze "
                    "with hash, development evidence, public-instance count, and contrast"
                )
            for label, value in (
                ("sample-size freeze", self.sample_size_freeze_hash),
                ("development manifest", self.development_manifest_hash),
            ):
                if not isinstance(value, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", value
                ):
                    raise ValueError(f"{label} hash must be 64 lowercase hex characters")
            if (
                not isinstance(self.frozen_public_instances, int)
                or isinstance(self.frozen_public_instances, bool)
                or self.frozen_public_instances <= 0
            ):
                raise ValueError("frozen public-instance count must be positive")
            if not self.primary_contrast_id:
                raise ValueError("held-out test execution requires a primary contrast ID")
            rows_per_public_instance = (
                4
                if self.experiment == "e2"
                or (self.experiment == "e1" and self.feedback_source == "shuffled")
                else 2
                if self.experiment == "e1"
                else 1
            )
            frozen_rows = self.frozen_public_instances * rows_per_public_instance
            if self.sample_start + self.num_samples > frozen_rows:
                raise ValueError(
                    "held-out shard exceeds the exact frozen public-instance count: "
                    f"end row {self.sample_start + self.num_samples}, frozen rows {frozen_rows}"
                )
            if self.pilot_id in {"pilot_a", "pilot_b", "pilot_c", "pilot_d"}:
                raise ValueError(
                    "development pilot presets cannot be relabelled as held-out evaluation"
                )
        elif any(value is not None for value in freeze_fields):
            raise ValueError("sample-size freeze fields are valid only for the held-out test split")
        if self.tier == "tier2":
            if self.attacker != "llm":
                raise ValueError("Tier 2 requires attacker='llm'")
            if not self.model_id:
                raise ValueError("Tier 2 requires --model-id for an already available local model")
            if not is_immutable_model_revision(self.model_revision):
                raise ValueError(
                    "Tier 2 requires --model-revision as an exact 40-64 hex commit "
                    "or sha256:<64-hex> local-checkpoint fingerprint"
                )
            if not self.model_cache_dir:
                raise ValueError("Tier 2 requires a persistent --model-cache-dir")
            if self.dtype == "auto":
                object.__setattr__(self, "dtype", "bfloat16")
            if self.dtype not in {"float32", "float16", "bfloat16"}:
                raise ValueError("Tier-2 dtype must be float32, float16, or bfloat16")
            if self.batch_size != 1:
                raise ValueError(
                    "the single-episode Tier-2 client currently requires --batch-size 1; "
                    "larger values would be operationally inert"
                )
        elif self.attacker == "llm":
            raise ValueError("attacker='llm' requires tier2")

    def scientific_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("output_dir")
        data.pop("overwrite")
        data.pop("grid_hash")
        data.pop("grid_task_id")
        data.pop("shard_id")
        data.pop("pilot_id")
        # Cache placement is an operational concern.  The immutable
        # ``model_revision`` (a commit or full checkpoint fingerprint) binds
        # the scientific model identity, while moving the very same bytes to a
        # different persistent cache must not fork the experiment cohort.
        # Preserve the established Tier-1 v2 record identity (where this
        # field is always ``None``) while correcting the operational-path
        # confound for local-model Tier-2 records.
        if self.tier == "tier2":
            data.pop("model_cache_dir")
        return data

    def operational_dict(self) -> dict[str, Any]:
        """Return non-scientific run locators for provenance only."""

        return {
            "output_dir": str(self.output_dir),
            "model_cache_dir": self.model_cache_dir,
            "grid_hash": self.grid_hash,
            "grid_task_id": self.grid_task_id,
            "shard_id": self.shard_id,
            "pilot_id": self.pilot_id,
        }

    @property
    def configuration_hash(self) -> str:
        return stable_hash(self.scientific_dict())

    def as_manifest_config(self) -> dict[str, Any]:
        return self.scientific_dict()
