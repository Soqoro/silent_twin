"""Deterministic, model-free AgentDojo experiment grids.

This module reads frozen JSON artifacts only.  It must remain safe on a login
node without AgentDojo, torch, transformers, CUDA, or model files installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .config import (
    AGENTDOJO_EXPERIMENTS,
    AGENTDOJO_SUITES,
    ECOLOGICAL_ATTACKS,
    ECOLOGICAL_ATTACK_TARGET_PIPELINE,
    ECOLOGICAL_DEFENSES,
    ECOLOGICAL_RELEASED_ATTACKS,
    E3_CHANNELS,
    E4_WORKFLOWS,
    E5_ABLATIONS,
    FEEDBACK_POLICIES,
    FEEDBACK_SOURCES,
    SETTLEMENT_RUNTIMES,
    AgentDojoConfigError,
    AgentDojoExperimentConfig,
    CONTROLLED_MODEL_PROMPT_HASH,
    CONTROLLED_SYSTEM_PROMPT_HASH,
    ECOLOGICAL_MODEL_PROMPT_HASH,
    ECOLOGICAL_SYSTEM_PROMPT_HASH,
    ModelIdentity,
    bundle_hash,
    canonical_json,
    load_json_object,
    require_hash,
    stable_hash,
)
from .freeze import UpstreamBindings, validate_agentdojo_sample_size_freeze
from .action_eligibility import ESTIMATION_ONLY_DISPOSITION
from .pair_mining import (
    PairMiningError,
    SUBSET_STRATEGY_SCHEMA_VERSION,
    monitor_pair_binding,
    validate_pair_registry,
)
from .recipient_separation import (
    RECIPIENT_SEPARATION_ATTACKER_IDENTITY,
    RECIPIENT_SEPARATION_DISPOSITION,
    RECIPIENT_SEPARATION_STRATEGY_SCHEMA_VERSION,
)


GRID_SCHEMA_VERSION = "silenttwin.agentdojo.grid.v1"
CATALOG_SCHEMA_VERSION = "silenttwin.agentdojo.catalog.v1"
SPLITS_SCHEMA_VERSION = "silenttwin.agentdojo.splits.v1"
STRATEGY_SCHEMA_VERSION = "silenttwin.agentdojo.candidate_strategy_catalog.v1"
STRATEGY_SCHEMA_VERSION_V2 = SUBSET_STRATEGY_SCHEMA_VERSION
STRATEGY_SCHEMA_VERSION_RECIPIENT_SEPARATION = (
    RECIPIENT_SEPARATION_STRATEGY_SCHEMA_VERSION
)
PAIR_SCHEMA_VERSION = "silenttwin.agentdojo.pair_registry.v1"
GRID_PLAN_SCHEMA_VERSION = "silenttwin.agentdojo.grid_plan.v1"
FAKE_SMOKE_ARTIFACT_CLASS = "deterministic_fake_smoke_fixture"
FAKE_SMOKE_EVIDENCE_CLASS = "engineering_smoke_only"
SCENARIO_REQUIRED_FIELDS = (
    "scenario_id",
    "suite",
    "user_task_id",
    "injection_task_id",
    "injection_vector_id",
    "user_prompt_hash",
    "injection_goal_hash",
    "tool_schema_hash",
    "initial_environment_hash",
    "clean_initial_environment_hash",
    "structural_group_id",
    "dataset_split",
    "agentdojo_package_version",
    "agentdojo_source_revision",
    "agentdojo_benchmark_version",
)


class AgentDojoGridError(AgentDojoConfigError):
    """A frozen artifact or deterministic grid is invalid."""


def is_estimation_only_protocol_disposition(value: Any) -> bool:
    """Return whether a protocol is restricted to nonconfirmatory estimation."""

    return value in {
        ESTIMATION_ONLY_DISPOSITION,
        RECIPIENT_SEPARATION_DISPOSITION,
    }


def _artifact_payload(document: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(document)
    result.pop(hash_field, None)
    return result


def validate_hashed_document(
    document: Mapping[str, Any],
    *,
    schema: str,
    hash_field: str,
    label: str,
) -> str:
    if document.get("schema_version") != schema:
        raise AgentDojoGridError(f"unsupported {label} schema")
    recorded = document.get(hash_field)
    require_hash(hash_field, str(recorded))
    calculated = stable_hash(_artifact_payload(document, hash_field))
    if recorded != calculated:
        raise AgentDojoGridError(f"{label} {hash_field} is invalid")
    return str(recorded)


def hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    catalog: Mapping[str, Any]
    splits: Mapping[str, Any]
    strategy_catalog: Mapping[str, Any]
    pair_registry: Mapping[str, Any]
    analysis_plan: Mapping[str, Any]
    dependency_lock_hash: str
    upstream: UpstreamBindings
    scenarios: tuple[Mapping[str, Any], ...]


def load_frozen_inputs(
    *,
    catalog_path: Path | str,
    splits_path: Path | str,
    strategy_catalog_path: Path | str,
    pair_registry_path: Path | str,
    analysis_plan_path: Path | str,
    dependency_lock_path: Path | str,
) -> FrozenInputs:
    catalog = load_json_object(catalog_path, label="AgentDojo catalog")
    splits = load_json_object(splits_path, label="AgentDojo split manifest")
    strategies = load_json_object(
        strategy_catalog_path, label="AgentDojo candidate-strategy catalog"
    )
    pairs = load_json_object(pair_registry_path, label="AgentDojo pair registry")
    analysis = load_json_object(analysis_plan_path, label="AgentDojo analysis plan")
    catalog_hash = validate_hashed_document(
        catalog,
        schema=CATALOG_SCHEMA_VERSION,
        hash_field="catalog_hash",
        label="AgentDojo catalog",
    )
    split_hash = validate_hashed_document(
        splits,
        schema=SPLITS_SCHEMA_VERSION,
        hash_field="split_manifest_hash",
        label="AgentDojo split manifest",
    )
    strategy_schema = str(strategies.get("schema_version", ""))
    if strategy_schema not in {
        STRATEGY_SCHEMA_VERSION,
        STRATEGY_SCHEMA_VERSION_V2,
        STRATEGY_SCHEMA_VERSION_RECIPIENT_SEPARATION,
    }:
        raise AgentDojoGridError("unsupported candidate-strategy catalog schema")
    strategy_hash = validate_hashed_document(
        strategies,
        schema=strategy_schema,
        hash_field="candidate_strategy_catalog_hash",
        label="candidate-strategy catalog",
    )
    pair_hash = validate_hashed_document(
        pairs,
        schema=PAIR_SCHEMA_VERSION,
        hash_field="pair_registry_hash",
        label="pair registry",
    )
    fixture_documents = (strategies, pairs)
    fixture_markers = tuple(
        document.get("artifact_class") == FAKE_SMOKE_ARTIFACT_CLASS
        for document in fixture_documents
    )
    if any(fixture_markers) and not all(fixture_markers):
        raise AgentDojoGridError(
            "fake-smoke strategy and pair artifacts must be selected together"
        )
    if all(fixture_markers):
        for label, document in zip(
            ("candidate-strategy catalog", "pair registry"), fixture_documents
        ):
            if (
                document.get("evidence_class") != FAKE_SMOKE_EVIDENCE_CLASS
                or document.get("scientific_evidence_eligible") is not False
                or document.get("derived_from_synthetic_monitor_decisions") is not True
                or not isinstance(document.get("claim_boundary"), str)
                or not document.get("claim_boundary")
            ):
                raise AgentDojoGridError(
                    f"{label} has an incomplete fake-smoke claim boundary"
                )
    if splits.get("catalog_hash") != catalog_hash:
        raise AgentDojoGridError("split manifest is bound to another catalog")
    if strategies.get("catalog_hash") != catalog_hash or strategies.get(
        "split_manifest_hash"
    ) != split_hash:
        raise AgentDojoGridError("candidate strategies use another catalog/split")
    if pairs.get("catalog_hash") != catalog_hash or pairs.get(
        "split_manifest_hash"
    ) != split_hash or pairs.get("candidate_strategy_catalog_hash") != strategy_hash:
        raise AgentDojoGridError("pair registry uses another upstream chain")
    if is_estimation_only_protocol_disposition(
        pairs.get("protocol_disposition")
    ):
        try:
            validate_pair_registry(
                pairs,
                catalog=catalog,
                split_manifest=splits,
                strategy_catalog=strategies,
            )
        except (PairMiningError, ValueError) as exc:
            raise AgentDojoGridError(
                f"pair registry validation failed: {exc}"
            ) from exc
    if strategies.get("frozen_before_development_pair_validation") is not True:
        raise AgentDojoGridError(
            "candidate strategies were not frozen before development validation"
        )
    for field_name in ("strategies", "monitor_profiles"):
        frozen_rows = strategies.get(field_name)
        if not isinstance(frozen_rows, list) or not frozen_rows:
            raise AgentDojoGridError(f"candidate catalog lacks {field_name}")
        if any(
            not isinstance(row, Mapping) or row.get("frozen_on_split") != "train"
            for row in frozen_rows
        ):
            raise AgentDojoGridError(f"candidate catalog {field_name} are not train-frozen")
    if pairs.get("test_outcomes_inspected") is not False:
        raise AgentDojoGridError("pair registry is contaminated by held-out outcomes")
    pair_rows = pairs.get("pairs")
    if not isinstance(pair_rows, list) or len(pair_rows) != len(AGENTDOJO_SUITES):
        raise AgentDojoGridError("pair registry must contain one pair per suite")
    if {row.get("suite") for row in pair_rows if isinstance(row, Mapping)} != set(
        AGENTDOJO_SUITES
    ) or any(
        not isinstance(row, Mapping)
        or row.get("selection_split") != "train"
        or row.get("validation_split") != "development"
        for row in pair_rows
    ):
        raise AgentDojoGridError("pair registry did not select on train and validate on development")
    test_instantiations = pairs.get("test_instantiations")
    if not isinstance(test_instantiations, list) or any(
        not isinstance(row, Mapping)
        or row.get("status") != "unobserved_pre_execution"
        or row.get("selected_by_test_outcome") is not False
        for row in test_instantiations
    ):
        raise AgentDojoGridError("pair registry test rows were not frozen before execution")
    if catalog.get("suite_order") != list(AGENTDOJO_SUITES):
        raise AgentDojoGridError("catalog suite order differs from the pinned release")
    if analysis.get("schema_version") != "silenttwin.agentdojo.analysis_plan.v1":
        raise AgentDojoGridError("unsupported AgentDojo analysis-plan schema")
    if analysis.get("environment_backend") != "agentdojo" or analysis.get(
        "independent_unit"
    ) != "structural_group_id":
        raise AgentDojoGridError("analysis plan uses another backend/independent unit")
    if analysis.get("suite_stratification") != list(AGENTDOJO_SUITES):
        raise AgentDojoGridError("analysis plan must declare all suite strata in pinned order")
    if analysis.get("suite_weighting") not in {"equal_suite", "task_weighted"}:
        raise AgentDojoGridError("analysis plan lacks an explicit suite-weighting rule")

    raw_scenarios = catalog.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise AgentDojoGridError("catalog scenarios must be a non-empty array")
    scenarios: list[Mapping[str, Any]] = []
    scenario_ids: set[str] = set()
    for index, row in enumerate(raw_scenarios):
        if not isinstance(row, Mapping):
            raise AgentDojoGridError(f"catalog scenario {index} is not an object")
        missing = [name for name in SCENARIO_REQUIRED_FIELDS if not row.get(name)]
        if missing:
            raise AgentDojoGridError(f"catalog scenario {index} lacks {missing}")
        if row["suite"] not in AGENTDOJO_SUITES:
            raise AgentDojoGridError(f"catalog scenario {index} has an unknown suite")
        identifier = str(row["scenario_id"])
        if identifier in scenario_ids:
            raise AgentDojoGridError(f"duplicate scenario_id {identifier!r}")
        scenario_ids.add(identifier)
        if row["dataset_split"] not in {"train", "development", "test"}:
            raise AgentDojoGridError(f"catalog scenario {index} has an invalid split")
        if (
            row["agentdojo_package_version"]
            != catalog.get("agentdojo_package_version")
            or row["agentdojo_source_revision"]
            != catalog.get("agentdojo_source_revision")
            or row["agentdojo_benchmark_version"]
            != catalog.get("agentdojo_benchmark_version")
        ):
            raise AgentDojoGridError(
                f"catalog scenario {index} is not bound to top-level AgentDojo versions"
            )
        if row.get("catalog_binding") not in {None, "top_level_catalog_hash"} and row.get(
            "catalog_hash"
        ) != catalog_hash:
            raise AgentDojoGridError(f"catalog scenario {index} has an invalid catalog binding")
        for name in (
            "user_prompt_hash",
            "injection_goal_hash",
            "tool_schema_hash",
            "initial_environment_hash",
            "clean_initial_environment_hash",
            "released_attack_rendering_hash",
            "released_attack_initial_environment_hash",
        ):
            require_hash(f"scenario.{name}", str(row[name]))
        if row.get("released_attack_name") != "direct":
            raise AgentDojoGridError(
                f"catalog scenario {index} lacks the frozen released direct attack"
            )
        if row.get("released_attack_target_pipeline") != "silenttwin-local-tool-loop":
            raise AgentDojoGridError(
                f"catalog scenario {index} has an invalid released attack target pipeline"
            )
        scenarios.append(dict(row))
    expected_test_rows = {
        (str(row["scenario_id"]), str(row["suite"]), str(row["structural_group_id"]))
        for row in scenarios
        if row["dataset_split"] == "test"
    }
    observed_test_rows = {
        (
            str(row.get("scenario_id")),
            str(row.get("suite")),
            str(row.get("structural_group_id")),
        )
        for row in test_instantiations
        if isinstance(row, Mapping)
    }
    if is_estimation_only_protocol_disposition(
        pairs.get("protocol_disposition")
    ):
        if observed_test_rows:
            raise AgentDojoGridError(
                "estimation-only pair registry instantiates held-out scenarios"
            )
    elif observed_test_rows != expected_test_rows:
        raise AgentDojoGridError(
            "pair registry must instantiate every frozen test scenario without filtering"
        )
    registry_revision = catalog.get("scenario_registry_revision")
    registry_hash = catalog.get("scenario_registry_hash")
    if not isinstance(registry_revision, str) or not registry_revision:
        raise AgentDojoGridError("catalog lacks scenario_registry_revision")
    require_hash("scenario_registry_hash", str(registry_hash))
    if registry_hash != stable_hash(scenarios):
        raise AgentDojoGridError("catalog scenario_registry_hash is invalid")
    analysis_hash = stable_hash(analysis)
    lock_hash = hash_file(dependency_lock_path)
    upstream = UpstreamBindings(
        catalog_hash=catalog_hash,
        scenario_registry_revision=registry_revision,
        scenario_registry_hash=str(registry_hash),
        split_manifest_hash=split_hash,
        candidate_strategy_catalog_hash=strategy_hash,
        pair_registry_hash=pair_hash,
        analysis_plan_hash=analysis_hash,
        dependency_lock_hash=lock_hash,
        package_version=str(catalog.get("agentdojo_package_version", "")),
        source_revision=str(catalog.get("agentdojo_source_revision", "")),
        benchmark_version=str(catalog.get("agentdojo_benchmark_version", "")),
    )
    return FrozenInputs(
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategies,
        pair_registry=pairs,
        analysis_plan=analysis,
        dependency_lock_hash=lock_hash,
        upstream=upstream,
        scenarios=tuple(scenarios),
    )


def validate_structural_splits(inputs: FrozenInputs) -> dict[str, tuple[str, ...]]:
    raw = inputs.splits.get("splits")
    if not isinstance(raw, Mapping) or set(raw) != {"train", "development", "test"}:
        raise AgentDojoGridError("split manifest must contain train/development/test")
    catalog_groups = {str(row["structural_group_id"]) for row in inputs.scenarios}
    result: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    seen_scenarios: set[str] = set()
    for split in ("train", "development", "test"):
        entry = raw[split]
        identifiers = (
            entry.get("structural_group_ids") if isinstance(entry, Mapping) else None
        )
        if not isinstance(identifiers, list) or not identifiers:
            raise AgentDojoGridError(f"{split} split has no structural_group_ids")
        values = tuple(str(item) for item in identifiers)
        if len(set(values)) != len(values):
            raise AgentDojoGridError(f"{split} split repeats structural groups")
        overlap = seen & set(values)
        if overlap:
            raise AgentDojoGridError(
                f"structural groups cross split boundaries: {sorted(overlap)}"
            )
        seen.update(values)
        result[split] = values
        scenario_ids = entry.get("scenario_ids") if isinstance(entry, Mapping) else None
        if not isinstance(scenario_ids, list) or not scenario_ids:
            raise AgentDojoGridError(f"{split} split has no scenario_ids")
        normalized_scenarios = tuple(str(item) for item in scenario_ids)
        if len(set(normalized_scenarios)) != len(normalized_scenarios):
            raise AgentDojoGridError(f"{split} split repeats scenarios")
        if seen_scenarios & set(normalized_scenarios):
            raise AgentDojoGridError("scenario IDs cross split boundaries")
        seen_scenarios.update(normalized_scenarios)
        expected_scenarios = {
            str(row["scenario_id"])
            for row in inputs.scenarios
            if row["dataset_split"] == split
        }
        if set(normalized_scenarios) != expected_scenarios:
            raise AgentDojoGridError(
                f"{split} scenario IDs disagree with the frozen catalog assignment"
            )
        expected_groups = {
            str(row["structural_group_id"])
            for row in inputs.scenarios
            if row["dataset_split"] == split
        }
        if set(values) != expected_groups:
            raise AgentDojoGridError(
                f"{split} structural groups disagree with the frozen catalog assignment"
            )
    if seen != catalog_groups:
        raise AgentDojoGridError("split manifest is not an exhaustive catalog partition")
    if seen_scenarios != {str(row["scenario_id"]) for row in inputs.scenarios}:
        raise AgentDojoGridError("split manifest is not an exhaustive scenario partition")
    return result


@dataclass(frozen=True, slots=True)
class ScenarioBundle:
    suite: str
    dataset_split: str
    scenario_ids: tuple[str, ...]
    structural_group_ids: tuple[str, ...]
    bundle_hash: str


def scenario_bundles(
    inputs: FrozenInputs,
    *,
    suite: str,
    dataset_split: str,
    groups_per_bundle: int,
    selected_group_ids: Sequence[str] | None = None,
    eligible_scenario_ids: Sequence[str] | None = None,
) -> tuple[ScenarioBundle, ...]:
    if groups_per_bundle <= 0:
        raise AgentDojoGridError("groups_per_bundle must be positive")
    splits = validate_structural_splits(inputs)
    allowed = set(splits[dataset_split])
    suite_rows = [
        row
        for row in inputs.scenarios
        if row["suite"] == suite and str(row["structural_group_id"]) in allowed
    ]
    if eligible_scenario_ids is not None:
        eligible = tuple(str(item) for item in eligible_scenario_ids)
        if len(eligible) != len(set(eligible)):
            raise AgentDojoGridError("eligible scenario IDs are not unique")
        split_scenario_ids = {
            str(row["scenario_id"])
            for row in inputs.scenarios
            if row["dataset_split"] == dataset_split
        }
        if not set(eligible) <= split_scenario_ids:
            raise AgentDojoGridError(
                "eligible scenario IDs are outside the selected structural split"
            )
        eligible_set = set(eligible)
        suite_rows = [
            row for row in suite_rows if str(row["scenario_id"]) in eligible_set
        ]
    suite_groups = sorted({str(row["structural_group_id"]) for row in suite_rows})
    if selected_group_ids is not None:
        requested = tuple(str(item) for item in selected_group_ids)
        if len(set(requested)) != len(requested) or not set(requested) <= set(suite_groups):
            raise AgentDojoGridError(f"frozen {suite} structural groups are invalid")
        suite_groups = list(requested)
    if not suite_groups:
        raise AgentDojoGridError(f"no {dataset_split} structural groups for suite {suite}")
    result: list[ScenarioBundle] = []
    for start in range(0, len(suite_groups), groups_per_bundle):
        group_ids = tuple(suite_groups[start : start + groups_per_bundle])
        scenario_ids = tuple(
            sorted(
                str(row["scenario_id"])
                for row in suite_rows
                if str(row["structural_group_id"]) in group_ids
            )
        )
        digest = bundle_hash(
            suite=suite,
            dataset_split=dataset_split,
            scenario_ids=scenario_ids,
            structural_group_ids=group_ids,
        )
        result.append(
            ScenarioBundle(suite, dataset_split, scenario_ids, group_ids, digest)
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GridCell:
    cell_index: int
    configuration: Mapping[str, Any]
    configuration_hash: str
    shard_id: str

    def record(self, *, task_id: int, batch_offset: int) -> dict[str, Any]:
        return {
            "record_type": "grid_member",
            "schema_version": GRID_SCHEMA_VERSION,
            "task_id": task_id,
            "batch_offset": batch_offset,
            "cell_index": self.cell_index,
            "configuration": dict(self.configuration),
            "configuration_hash": self.configuration_hash,
            "shard_id": self.shard_id,
        }


@dataclass(frozen=True, slots=True)
class GridTask:
    task_id: int
    suite: str
    scenario_bundle_hash: str
    replicate: int
    cells: tuple[GridCell, ...]

    @property
    def batch_hash(self) -> str:
        return stable_hash(
            [(cell.configuration_hash, cell.shard_id) for cell in self.cells]
        )


@dataclass(frozen=True, slots=True)
class AgentDojoGrid:
    experiment_id: str
    tier2_track: str
    dataset_split: str
    tasks: tuple[GridTask, ...]
    upstream_binding_hash: str
    heldout_freeze_binding: Mapping[str, Any] | None = None
    protocol_disposition: str = "legacy_full_catalog"
    action_eligibility_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if self.protocol_disposition not in {
            "legacy_full_catalog",
            ESTIMATION_ONLY_DISPOSITION,
            RECIPIENT_SEPARATION_DISPOSITION,
        }:
            raise AgentDojoGridError("grid has an unknown protocol disposition")
        if is_estimation_only_protocol_disposition(self.protocol_disposition):
            require_hash(
                "action_eligibility_manifest_hash",
                str(self.action_eligibility_manifest_hash),
            )
            if self.dataset_split == "test" or self.heldout_freeze_binding is not None:
                raise AgentDojoGridError(
                    "estimation-only grids cannot contain held-out execution"
                )
        elif self.action_eligibility_manifest_hash is not None:
            raise AgentDojoGridError(
                "legacy grid has an unbound action-eligibility hash"
            )

    @property
    def cells(self) -> tuple[GridCell, ...]:
        return tuple(cell for task in self.tasks for cell in task.cells)

    @property
    def grid_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": GRID_SCHEMA_VERSION,
                "experiment_id": self.experiment_id,
                "tier2_track": self.tier2_track,
                "dataset_split": self.dataset_split,
                "upstream_binding_hash": self.upstream_binding_hash,
                "protocol_disposition": self.protocol_disposition,
                "action_eligibility_manifest_hash": (
                    self.action_eligibility_manifest_hash
                ),
                "heldout_freeze_binding": (
                    dict(self.heldout_freeze_binding)
                    if self.heldout_freeze_binding is not None
                    else None
                ),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "suite": task.suite,
                        "scenario_bundle_hash": task.scenario_bundle_hash,
                        "replicate": task.replicate,
                        "members": [
                            (cell.configuration_hash, cell.shard_id)
                            for cell in task.cells
                        ],
                    }
                    for task in self.tasks
                ],
            }
        )

    def metadata(self) -> dict[str, Any]:
        observed_suites = {task.suite for task in self.tasks}
        full_suite_coverage = observed_suites == set(AGENTDOJO_SUITES)
        estimation_only = is_estimation_only_protocol_disposition(
            self.protocol_disposition
        )
        confirmatory_coverage = full_suite_coverage and not estimation_only
        return {
            "record_type": "grid_metadata",
            "schema_version": GRID_SCHEMA_VERSION,
            "environment_backend": "agentdojo",
            "experiment_id": self.experiment_id,
            "tier2_track": self.tier2_track,
            "dataset_split": self.dataset_split,
            "suite_order": list(AGENTDOJO_SUITES),
            "suite_coverage_status": (
                "full_four_suite_estimation_only"
                if full_suite_coverage and estimation_only
                else "full_four_suite"
                if full_suite_coverage
                else "development_subset_nonconfirmatory"
            ),
            "confirmatory_suite_coverage_eligible": confirmatory_coverage,
            "total_tasks": len(self.tasks),
            "total_configurations": len(self.cells),
            "valid_array_range": f"0-{len(self.tasks) - 1}",
            "upstream_binding_hash": self.upstream_binding_hash,
            "protocol_disposition": self.protocol_disposition,
            "action_eligibility_manifest_hash": (
                self.action_eligibility_manifest_hash
            ),
            "heldout_status": (
                "freeze_bound_before_test"
                if self.heldout_freeze_binding is not None
                else "not_applicable_non_test"
            ),
            "heldout_freeze_binding": (
                dict(self.heldout_freeze_binding)
                if self.heldout_freeze_binding is not None
                else None
            ),
            "grid_hash": self.grid_hash,
            "model_free": True,
        }

    def records(self) -> list[dict[str, Any]]:
        records = [self.metadata()]
        for task in self.tasks:
            records.extend(
                cell.record(task_id=task.task_id, batch_offset=index)
                for index, cell in enumerate(task.cells)
            )
        return records

    def task(self, task_id: int) -> GridTask:
        if task_id < 0 or task_id >= len(self.tasks):
            raise AgentDojoGridError(
                f"array task id {task_id} is out of range; valid range is 0-{len(self.tasks)-1}"
            )
        return self.tasks[task_id]


def _models_from_plan(plan: Mapping[str, Any]) -> tuple[ModelIdentity, ...]:
    rows = plan.get("models", [])
    if not isinstance(rows, list):
        raise AgentDojoGridError("grid plan models must be an array")
    if any(not isinstance(row, Mapping) for row in rows):
        raise AgentDojoGridError("every grid-plan model must be an object")
    return tuple(ModelIdentity.from_mapping(row) for row in rows)


def _validate_recipient_separation_models(
    models: Sequence[ModelIdentity],
) -> None:
    """Require the exact preregistered attacker, with only runtime rebound."""

    if len(models) != 1:
        raise AgentDojoGridError(
            "recipient-separation grid requires exactly one attacker identity"
        )
    observed = models[0].scientific_dict()
    runtime_fingerprint = observed.pop("runtime_fingerprint", None)
    if (
        observed != RECIPIENT_SEPARATION_ATTACKER_IDENTITY
        or not isinstance(runtime_fingerprint, str)
        or not runtime_fingerprint.startswith("sha256:")
    ):
        raise AgentDojoGridError(
            "recipient-separation attacker differs from the frozen protocol identity"
        )


def _models_for_cell(
    models: Sequence[ModelIdentity],
    *,
    experiment_id: str,
    values: Mapping[str, Any],
    strategy_catalog: Mapping[str, Any],
) -> tuple[ModelIdentity, ...]:
    """Keep only identities that the selected protocol can actually load.

    This makes checkpoint and runtime preflight exact: a worker is never made
    to provision an attacker checkpoint for E4, a victim checkpoint for a
    controlled experiment, or a detector checkpoint for an ecological cell
    that does not use that detector.
    """

    required: set[str] = set()
    if experiment_id in {"e1", "e2", "e3"}:
        required.add("attacker")
    elif experiment_id == "e4":
        pass
    elif experiment_id == "e5":
        if values.get("ablation") not in {
            "incomplete_data_dependencies",
            "incomplete_control_dependencies",
            "missing_atomic_group",
        }:
            required.add("attacker")
    elif experiment_id == "ecological":
        required.add("victim")
        if values.get("ecological_defense") == "transformers_pi_detector":
            required.add("monitor")
    if (
        experiment_id != "ecological"
        and values.get("monitor_family") != "deterministic_task_policy"
    ):
        required.add("monitor")

    selected = tuple(model for model in models if model.role in required)
    observed = {model.role for model in selected}
    if observed != required:
        missing = sorted(required - observed)
        raise AgentDojoGridError(
            f"grid plan lacks required model identities for roles {missing}"
        )
    rebound: list[ModelIdentity] = []
    for model in selected:
        if model.role == "attacker" and model.prompt_hash != CONTROLLED_MODEL_PROMPT_HASH:
            raise AgentDojoGridError(
                "attacker prompt_hash does not match the controlled prompt template"
            )
        if model.role == "victim" and model.prompt_hash != ECOLOGICAL_MODEL_PROMPT_HASH:
            raise AgentDojoGridError(
                "victim prompt_hash does not match the ecological prompt templates"
            )
        if model.role == "monitor" and values.get("monitor_family") not in {
            "deterministic_task_policy",
            "transformers_pi_detector",
        }:
            profiles = {
                str(row.get("profile_id")): row
                for row in strategy_catalog.get("monitor_profiles", ())
                if isinstance(row, Mapping)
            }
            try:
                theta0 = profiles[str(values["profile_theta0"])]
                theta1 = profiles[str(values["profile_theta1"])]
            except KeyError as exc:
                raise AgentDojoGridError(
                    "learned monitor identity references an unknown frozen profile"
                ) from exc
            identity_binding = {
                "implementation": model.implementation,
                "model_id": model.model_id,
                "model_revision": model.model_revision,
                "tokenizer_revision": model.tokenizer_revision,
                "checkpoint_fingerprint": model.checkpoint_fingerprint,
                "runtime_fingerprint": model.runtime_fingerprint,
                "reasoning_mode": model.reasoning_mode,
                "dtype": model.dtype,
                "decoding": {
                    "temperature": model.temperature,
                    "top_p": model.top_p,
                    "max_new_tokens": model.max_new_tokens,
                },
            }
            for theta_name, profile in (("profile_theta0", theta0), ("profile_theta1", theta1)):
                mismatched = [
                    field
                    for field, expected in identity_binding.items()
                    if profile.get(field) != expected
                ]
                if mismatched:
                    raise AgentDojoGridError(
                        "grid-plan monitor identity differs from frozen "
                        f"{theta_name} on {mismatched}"
                    )
            # The client loads one transport.  Its profile-facing identity is
            # the ordered pair of private prompt/policy adapters, not either
            # theta in isolation.  Threshold remains profile-local.
            model = replace(
                model,
                prompt_hash=stable_hash(
                    {
                        "theta0": theta0.get("prompt_hash"),
                        "theta1": theta1.get("prompt_hash"),
                    }
                ),
                policy_hash=stable_hash(
                    {
                        "theta0": theta0.get("policy_hash"),
                        "theta1": theta1.get("policy_hash"),
                    }
                ),
                threshold=None,
            )
        rebound.append(model)
    return tuple(rebound)


def _experiment_cells(plan: Mapping[str, Any], experiment_id: str) -> tuple[dict[str, Any], ...]:
    experiments = plan.get("experiments")
    spec = experiments.get(experiment_id) if isinstance(experiments, Mapping) else None
    if not isinstance(spec, Mapping):
        raise AgentDojoGridError(f"grid plan has no specification for {experiment_id}")
    rows = spec.get("cells")
    factor_grid = spec.get("factor_grid")
    if rows is not None and factor_grid is not None:
        raise AgentDojoGridError("grid plan must use cells or factor_grid, not both")
    if factor_grid is not None:
        if not isinstance(factor_grid, Mapping) or not factor_grid:
            raise AgentDojoGridError("factor_grid must be a non-empty object")
        factor_order = spec.get("factor_order", list(factor_grid))
        if (
            not isinstance(factor_order, list)
            or len(set(factor_order)) != len(factor_order)
            or set(factor_order) != set(factor_grid)
        ):
            raise AgentDojoGridError("factor_order must list every factor exactly once")
        levels: list[list[Any]] = []
        for name in factor_order:
            values = factor_grid[name]
            if not isinstance(values, list) or not values:
                raise AgentDojoGridError(f"factor {name!r} must have non-empty levels")
            if len({canonical_json(item) for item in values}) != len(values):
                raise AgentDojoGridError(f"factor {name!r} repeats a level")
            levels.append(values)
        defaults = spec.get("defaults", {})
        if not isinstance(defaults, Mapping):
            raise AgentDojoGridError("factor-grid defaults must be an object")
        rows = [
            {
                **dict(defaults),
                **dict(zip((str(name) for name in factor_order), combination)),
            }
            for combination in itertools.product(*levels)
        ]
    if not isinstance(rows, list) or not rows:
        raise AgentDojoGridError(f"grid plan has no cells for {experiment_id}")
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise AgentDojoGridError("grid-cell defaults must be an object")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise AgentDojoGridError(f"grid-plan cell {index} is not an object")
        result.append({**dict(defaults), **dict(row)})
    if len({stable_hash(row) for row in result}) != len(result):
        raise AgentDojoGridError(f"grid plan repeats a cell for {experiment_id}")
    return tuple(result)


def _coverage_spec(
    analysis_plan: Mapping[str, Any], experiment_id: str
) -> Mapping[str, Any]:
    coverage = analysis_plan.get("required_grid_coverage")
    if not isinstance(coverage, Mapping):
        raise AgentDojoGridError(
            "analysis plan lacks required_grid_coverage preregistration"
        )
    spec = coverage.get(experiment_id)
    if not isinstance(spec, Mapping):
        raise AgentDojoGridError(
            f"analysis plan lacks {experiment_id.upper()} grid-coverage preregistration"
        )
    return spec


def _exact_string_sequence(
    value: Any, expected: Sequence[str], *, label: str
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or tuple(value) != tuple(expected)
    ):
        raise AgentDojoGridError(f"analysis plan has invalid {label}")
    return tuple(value)


def _require_exact_cell_projection(
    cells: Sequence[Mapping[str, Any]],
    *,
    expected: set[tuple[Any, ...]],
    projection: Any,
    label: str,
) -> None:
    observed = {projection(row) for row in cells}
    if observed != expected or len(cells) != len(expected):
        missing = sorted(expected - observed, key=repr)
        unexpected = sorted(observed - expected, key=repr)
        raise AgentDojoGridError(
            f"{label} grid must contain its exact preregistered coverage; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_preregistered_cells(
    experiment_id: str,
    cells: Sequence[Mapping[str, Any]],
    *,
    analysis_plan: Mapping[str, Any],
) -> None:
    if experiment_id == "e1":
        spec = _coverage_spec(analysis_plan, experiment_id)
        policies = _exact_string_sequence(
            spec.get("feedback_policies"),
            FEEDBACK_POLICIES,
            label="E1 feedback policies",
        )
        sources = _exact_string_sequence(
            spec.get("feedback_sources"),
            FEEDBACK_SOURCES,
            label="E1 feedback sources",
        )
        budgets = spec.get("query_budgets")
        permitted_budget_schedules = ([0, 4, 16, 32],)
        if (
            analysis_plan.get("protocol_revision")
            == "scientific-v6-feedback-recipient-separation-v1"
        ):
            permitted_budget_schedules = ([0, 4, 16],)
        if budgets not in permitted_budget_schedules or spec.get(
            "crossing"
        ) != "complete_cartesian_product":
            raise AgentDojoGridError("analysis plan has invalid E1 coverage semantics")
        required = set(itertools.product(policies, sources, budgets))
        _require_exact_cell_projection(
            cells,
            expected=required,
            projection=lambda row: (
                row.get("feedback_policy"),
                row.get("feedback_source"),
                row.get("query_budget"),
            ),
            label="E1",
        )
    elif experiment_id == "e2":
        required = {
            ("no_probe", 0),
            ("genuine", 4),
            ("genuine", 16),
            ("matched_shuffled", 4),
            ("matched_shuffled", 16),
            ("constant", 4),
            ("constant", 16),
            ("random_final", 4),
            ("random_final", 16),
            ("oracle", 0),
            ("silenttwin", 0),
            ("silenttwin", 4),
            ("silenttwin", 16),
        }
        observed = {(row.get("condition"), row.get("query_budget")) for row in cells}
        if observed != required or len(cells) != len(required):
            raise AgentDojoGridError(
                "E2 grid must contain the exact preregistered condition/Q cells"
            )
        if any(
            row.get("feedback_policy")
            != ("silenttwin" if row.get("condition") == "silenttwin" else "generic_failure")
            for row in cells
        ):
            raise AgentDojoGridError(
                "E2 controls require generic_failure and SilentTwin requires its own policy"
            )
    elif experiment_id == "e3":
        spec = _coverage_spec(analysis_plan, experiment_id)
        channels = _exact_string_sequence(
            spec.get("channels"), E3_CHANNELS, label="E3 channels"
        )
        budgets = spec.get("exact_silenttwin_query_budgets")
        overrides = spec.get("runtime_overrides")
        expected_overrides = {
            "visible_settlement": "visible_transactional_finalization"
        }
        if (
            budgets != [0, 16]
            or spec.get("degraded_query_budget") != 16
            or spec.get("feedback_policy") != "silenttwin"
            or spec.get("default_settlement_runtime")
            != "speculative_dependency_aware"
            or overrides != expected_overrides
        ):
            raise AgentDojoGridError("analysis plan has invalid E3 coverage semantics")
        expected = {
            (
                channel,
                budget,
                "silenttwin",
                expected_overrides.get(channel, "speculative_dependency_aware"),
            )
            for channel in channels
            for budget in ([0, 16] if channel == "exact_silenttwin" else [16])
        }
        _require_exact_cell_projection(
            cells,
            expected=expected,
            projection=lambda row: (
                row.get("closure_channel"),
                row.get("query_budget"),
                row.get("feedback_policy"),
                row.get("settlement_runtime"),
            ),
            label="E3",
        )
    elif experiment_id == "e4":
        spec = _coverage_spec(analysis_plan, experiment_id)
        workflows = _exact_string_sequence(
            spec.get("workflows"), E4_WORKFLOWS, label="E4 workflows"
        )
        runtimes = _exact_string_sequence(
            spec.get("settlement_runtimes"),
            SETTLEMENT_RUNTIMES[:3],
            label="E4 settlement runtimes",
        )
        if (
            spec.get("crossing") != "complete_cartesian_product"
            or spec.get("feedback_policy") != "silenttwin"
            or spec.get("query_budget") != 0
        ):
            raise AgentDojoGridError("analysis plan has invalid E4 coverage semantics")
        expected = {
            (workflow, runtime, "silenttwin", 0)
            for workflow, runtime in itertools.product(workflows, runtimes)
        }
        _require_exact_cell_projection(
            cells,
            expected=expected,
            projection=lambda row: (
                row.get("workflow"),
                row.get("settlement_runtime"),
                row.get("feedback_policy"),
                row.get("query_budget"),
            ),
            label="E4",
        )
    elif experiment_id == "e5":
        spec = _coverage_spec(analysis_plan, experiment_id)
        ablations = _exact_string_sequence(
            spec.get("ablations"), E5_ABLATIONS, label="E5 ablations"
        )
        structural = (
            "incomplete_data_dependencies",
            "incomplete_control_dependencies",
            "missing_atomic_group",
        )
        _exact_string_sequence(
            spec.get("query_budget_zero_ablations"),
            structural,
            label="E5 Q=0 ablations",
        )
        if (
            spec.get("default_query_budget") != 16
            or spec.get("feedback_policy") != "silenttwin"
        ):
            raise AgentDojoGridError("analysis plan has invalid E5 coverage semantics")
        expected = {
            (ablation, 0 if ablation in structural else 16, "silenttwin")
            for ablation in ablations
        }
        _require_exact_cell_projection(
            cells,
            expected=expected,
            projection=lambda row: (
                row.get("ablation"),
                row.get("query_budget"),
                row.get("feedback_policy"),
            ),
            label="E5",
        )
    elif experiment_id == "ecological":
        spec = _coverage_spec(analysis_plan, experiment_id)
        attacks = spec.get("threat_attack_pairs")
        expected_attacks = [
            {
                "threat_mode": "clean",
                "ecological_attack": ECOLOGICAL_ATTACKS[0],
                "released_attack_name": None,
                "released_attack_target_pipeline": None,
            },
            *[
                {
                    "threat_mode": attack,
                    "ecological_attack": attack,
                    "released_attack_name": ECOLOGICAL_RELEASED_ATTACKS[0],
                    "released_attack_target_pipeline": ECOLOGICAL_ATTACK_TARGET_PIPELINE,
                }
                for attack in ECOLOGICAL_ATTACKS[1:]
            ],
        ]
        defenses = _exact_string_sequence(
            spec.get("defenses"), ECOLOGICAL_DEFENSES, label="ecological defenses"
        )
        if (
            attacks != expected_attacks
            or spec.get("crossing") != "complete_cartesian_product"
            or spec.get("settlement_runtime") != "native_agentdojo_restricted"
            or spec.get("tool_protocol_revision")
            != "agentdojo-native-tool-loop-v1"
            or spec.get("feedback_policy") != "ecological_native"
            or spec.get("feedback_source") != "not_applicable"
        ):
            raise AgentDojoGridError(
                "analysis plan has invalid ecological coverage semantics"
            )
        expected = {
            (
                attack["threat_mode"],
                attack["ecological_attack"],
                attack["released_attack_name"],
                attack["released_attack_target_pipeline"],
                defense,
                "native_agentdojo_restricted",
                "agentdojo-native-tool-loop-v1",
                "ecological_native",
                "not_applicable",
            )
            for attack, defense in itertools.product(expected_attacks, defenses)
        }
        _require_exact_cell_projection(
            cells,
            expected=expected,
            projection=lambda row: (
                row.get("threat_mode"),
                row.get("ecological_attack"),
                row.get("released_attack_name"),
                row.get("released_attack_target_pipeline"),
                row.get("ecological_defense"),
                row.get("settlement_runtime"),
                row.get("tool_protocol_revision"),
                row.get("feedback_policy"),
                row.get("feedback_source"),
            ),
            label="ecological",
        )


def validate_grid_manifest_coverage(
    grid: Mapping[str, Any], analysis_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate every task against the hash-bound preregistered cell matrix."""

    metadata = grid.get("metadata")
    members = grid.get("members")
    if not isinstance(metadata, Mapping) or not isinstance(members, list):
        raise AgentDojoGridError("grid coverage validation requires a loaded manifest")
    experiment_id = str(metadata.get("experiment_id"))
    expected_track = "ecological" if experiment_id == "ecological" else "controlled"
    if (
        analysis_plan.get("schema_version")
        != "silenttwin.agentdojo.analysis_plan.v1"
        or analysis_plan.get("tier2_track") != expected_track
        or metadata.get("tier2_track") != expected_track
        or analysis_plan.get("suite_stratification") != list(AGENTDOJO_SUITES)
    ):
        raise AgentDojoGridError(
            "grid and analysis plan have incompatible preregistration semantics"
        )
    by_task: dict[int, list[Mapping[str, Any]]] = {}
    for member in members:
        if not isinstance(member, Mapping) or not isinstance(
            member.get("configuration"), Mapping
        ):
            raise AgentDojoGridError("grid coverage contains a malformed member")
        by_task.setdefault(int(member.get("task_id", -1)), []).append(
            member["configuration"]
        )
    for configurations in by_task.values():
        _validate_preregistered_cells(
            experiment_id,
            configurations,
            analysis_plan=analysis_plan,
        )
    observed_suites = {
        str(member["configuration"]["agentdojo_suite"]) for member in members
    }
    full_suite_coverage = observed_suites == set(AGENTDOJO_SUITES)
    estimation_only = is_estimation_only_protocol_disposition(
        metadata.get("protocol_disposition")
    )
    confirmatory_coverage = full_suite_coverage and not estimation_only
    if (
        metadata.get("confirmatory_suite_coverage_eligible")
        is not confirmatory_coverage
    ):
        raise AgentDojoGridError("grid suite coverage changed after manifest creation")
    return {
        "cell_coverage_status": "exact_preregistered_matrix",
        "suite_coverage_status": (
            "full_four_suite_estimation_only"
            if full_suite_coverage and estimation_only
            else "full_four_suite"
            if full_suite_coverage
            else "development_subset_nonconfirmatory"
        ),
        "confirmatory_suite_coverage_eligible": confirmatory_coverage,
    }


def build_grid(
    *,
    inputs: FrozenInputs,
    grid_plan: Mapping[str, Any],
    experiment_id: str,
    tier2_track: str,
    dataset_split: str,
    suites: Sequence[str] = AGENTDOJO_SUITES,
    replicates: Sequence[int] = (0,),
    groups_per_bundle: int = 8,
    sample_size_freeze: Mapping[str, Any] | None = None,
) -> AgentDojoGrid:
    if grid_plan.get("schema_version") != GRID_PLAN_SCHEMA_VERSION:
        raise AgentDojoGridError("unsupported AgentDojo grid-plan schema")
    if experiment_id not in AGENTDOJO_EXPERIMENTS:
        raise AgentDojoGridError(f"unknown AgentDojo experiment {experiment_id!r}")
    if dataset_split not in {"train", "development", "test"}:
        raise AgentDojoGridError(f"unknown AgentDojo dataset split {dataset_split!r}")
    pair_disposition = inputs.pair_registry.get(
        "protocol_disposition", "legacy_full_catalog"
    )
    estimation_only = is_estimation_only_protocol_disposition(pair_disposition)
    action_eligibility_hash: str | None = None
    eligible_scenarios: tuple[str, ...] | None = None
    if pair_disposition == RECIPIENT_SEPARATION_DISPOSITION:
        required_v6 = grid_plan.get("required_v6_artifact_hashes")
        expected_v6 = {
            "recipient_separation_protocol_hash": inputs.pair_registry.get(
                "recipient_separation_protocol_hash"
            ),
            "candidate_strategy_catalog_hash": (
                inputs.upstream.candidate_strategy_catalog_hash
            ),
            "pair_registry_hash": inputs.upstream.pair_registry_hash,
        }
        if (
            grid_plan.get("protocol_revision")
            != "scientific-v6-feedback-recipient-separation-v1"
            or grid_plan.get("protocol_disposition")
            != RECIPIENT_SEPARATION_DISPOSITION
            or grid_plan.get("design_phase")
            != "train_only_adaptive_feasibility"
            or grid_plan.get("template_only") is not False
            or required_v6 != expected_v6
        ):
            raise AgentDojoGridError(
                "recipient-separation grid plan is not an exact materialized "
                "scientific-v6 train freeze"
            )
    if estimation_only:
        action_eligibility_hash = str(
            inputs.pair_registry.get("action_eligibility_manifest_hash", "")
        )
        require_hash(
            "action_eligibility_manifest_hash", action_eligibility_hash
        )
        if dataset_split == "test":
            raise AgentDojoGridError(
                "estimation-only action-representable protocol forbids held-out grids"
            )
        if tier2_track == "controlled":
            cohorts = inputs.pair_registry.get("pilot_scenario_ids_by_split")
            selected = (
                cohorts.get(dataset_split)
                if isinstance(cohorts, Mapping)
                else None
            )
            if not isinstance(selected, list):
                raise AgentDojoGridError(
                    "estimation-only pair registry lacks its frozen scenario cohort"
                )
            eligible_scenarios = tuple(str(item) for item in selected)
            if not eligible_scenarios:
                raise AgentDojoGridError(
                    f"estimation-only protocol has no {dataset_split} scenarios"
                )
        if pair_disposition == RECIPIENT_SEPARATION_DISPOSITION:
            permitted_splits = inputs.pair_registry.get(
                "execution_permitted_splits"
            )
            if (
                not isinstance(permitted_splits, list)
                or dataset_split not in permitted_splits
            ):
                raise AgentDojoGridError(
                    "recipient-separation development execution requires its "
                    "separate immutable opening gate"
                )
    elif pair_disposition not in {None, "legacy_full_catalog"}:
        raise AgentDojoGridError("pair registry has an unknown protocol disposition")
    if dataset_split == "test" and experiment_id in {"e5", "ecological"}:
        raise AgentDojoGridError(
            f"{experiment_id} is development-only: no preregistered held-out "
            "power/sample-size freeze contract exists"
        )
    if inputs.analysis_plan.get("tier2_track") != tier2_track:
        raise AgentDojoGridError("analysis plan belongs to another Tier-2 track")
    if inputs.analysis_plan.get("suite_stratification") != list(AGENTDOJO_SUITES):
        raise AgentDojoGridError(
            "analysis plan must preregister the exact four-suite stratum order"
        )
    suite_values = tuple(str(item) for item in suites)
    if not suite_values or len(set(suite_values)) != len(suite_values):
        raise AgentDojoGridError("suites must be non-empty and unique")
    if any(item not in AGENTDOJO_SUITES for item in suite_values):
        raise AgentDojoGridError("grid includes an unknown AgentDojo suite")
    # Canonical suite ordering is fixed even if CLI flags arrive in another order.
    suite_values = tuple(item for item in AGENTDOJO_SUITES if item in suite_values)
    if dataset_split == "test" and suite_values != AGENTDOJO_SUITES:
        raise AgentDojoGridError(
            "held-out execution requires exact four-suite membership"
        )
    replicate_values = tuple(int(item) for item in replicates)
    if not replicate_values or len(set(replicate_values)) != len(replicate_values) or min(
        replicate_values
    ) < 0:
        raise AgentDojoGridError("replicates must be unique non-negative integers")
    cells = _experiment_cells(grid_plan, experiment_id)
    base = grid_plan.get("base_configuration", {})
    if not isinstance(base, Mapping):
        raise AgentDojoGridError("grid plan base_configuration must be an object")
    _validate_preregistered_cells(
        experiment_id,
        [{**dict(base), **dict(cell)} for cell in cells],
        analysis_plan=inputs.analysis_plan,
    )
    if experiment_id in {"e4", "e5"}:
        authored_workflows = inputs.strategy_catalog.get("mixed_workflows")
        if not isinstance(authored_workflows, list) or not authored_workflows:
            raise AgentDojoGridError(
                f"{experiment_id.upper()} requires nonempty train-frozen authored "
                "workflows before grid construction"
            )
    if experiment_id in {"e1", "e2", "e3", "e4"}:
        grid_primary = grid_plan.get("primary_contrasts", {}).get(experiment_id)
        analysis_primary = inputs.analysis_plan.get("primary_contrasts", {}).get(
            experiment_id
        )
        if not isinstance(grid_primary, str) or grid_primary != analysis_primary:
            raise AgentDojoGridError(
                "grid plan primary contrast differs from the frozen analysis plan"
            )
    uses_fake_smoke_artifacts = (
        inputs.strategy_catalog.get("artifact_class") == FAKE_SMOKE_ARTIFACT_CLASS
    )
    if uses_fake_smoke_artifacts:
        if base.get("fixture_mode") is not True:
            raise AgentDojoGridError(
                "deterministic fake-smoke artifacts are forbidden outside fixture_mode"
            )
        if (
            grid_plan.get("artifact_class") != FAKE_SMOKE_ARTIFACT_CLASS
            or grid_plan.get("evidence_class") != FAKE_SMOKE_EVIDENCE_CLASS
            or grid_plan.get("scientific_evidence_eligible") is not False
        ):
            raise AgentDojoGridError(
                "fake-smoke grid plan lacks its non-scientific evidence boundary"
            )
        required = grid_plan.get("required_fixture_artifact_hashes")
        expected = {
            "candidate_strategy_catalog_hash": inputs.upstream.candidate_strategy_catalog_hash,
            "pair_registry_hash": inputs.upstream.pair_registry_hash,
        }
        if required != expected:
            raise AgentDojoGridError(
                "fake-smoke grid plan is not bound to the selected fixture artifacts"
            )
    models = _models_from_plan(grid_plan)
    if pair_disposition == RECIPIENT_SEPARATION_DISPOSITION:
        _validate_recipient_separation_models(models)
    heldout_freeze: dict[str, Any] | None = None
    heldout_binding: dict[str, Any] | None = None
    if dataset_split == "test":
        if sample_size_freeze is None:
            raise AgentDojoGridError("held-out grid requires a sample-size freeze")
        primary = grid_plan.get("primary_contrasts", {}).get(experiment_id)
        if not isinstance(primary, str):
            raise AgentDojoGridError("grid plan lacks the held-out primary contrast")
        heldout_freeze = validate_agentdojo_sample_size_freeze(
            sample_size_freeze,
            experiment_id=experiment_id,
            primary_contrast_id=primary,
            upstream=inputs.upstream,
        )
        test_groups = validate_structural_splits(inputs)["test"]
        available_groups = {
            suite: sorted(
                {
                    str(row["structural_group_id"])
                    for row in inputs.scenarios
                    if row["suite"] == suite
                    and str(row["structural_group_id"]) in set(test_groups)
                }
            )
            for suite in AGENTDOJO_SUITES
        }
        recorded_available = heldout_freeze[
            "available_test_independent_unit_count_by_suite"
        ]
        expected_available = {
            suite: len(available_groups[suite]) for suite in AGENTDOJO_SUITES
        }
        if recorded_available != expected_available:
            raise AgentDojoGridError(
                "sample-size freeze test availability differs from the frozen catalog"
            )
        selected_ids = heldout_freeze["selected_structural_group_ids_by_suite"]
        counts = heldout_freeze["independent_unit_count_by_suite"]
        for suite in AGENTDOJO_SUITES:
            expected_ids = available_groups[suite][: int(counts[suite])]
            if selected_ids[suite] != expected_ids:
                raise AgentDojoGridError(
                    f"{suite} freeze does not use the canonical structural-group prefix"
                )
        heldout_binding = {
            "freeze_hash": heldout_freeze["freeze_hash"],
            "development_analysis_manifest_hash": heldout_freeze[
                "development_analysis_manifest_hash"
            ],
            "development_evidence_hash": heldout_freeze[
                "development_evidence_hash"
            ],
            "power_evidence_hash": heldout_freeze["power_evidence_hash"],
            "primary_contrast_id": primary,
            "claim_disposition": heldout_freeze["claim_disposition"],
            "independent_unit_count_by_suite": dict(counts),
            "available_test_independent_unit_count_by_suite": dict(
                recorded_available
            ),
            "selected_test_bundle_hash_by_suite": dict(
                heldout_freeze["selected_test_bundle_hash_by_suite"]
            ),
            "selected_structural_group_ids_by_suite": {
                suite: list(selected_ids[suite]) for suite in AGENTDOJO_SUITES
            },
            "structural_minimum_shortfalls": dict(
                heldout_freeze["structural_minimum_shortfalls"]
            ),
        }
    elif sample_size_freeze is not None:
        raise AgentDojoGridError("sample-size freezes are valid only for test")
    tasks: list[GridTask] = []
    cell_index = 0
    for suite in suite_values:
        resolved_monitor_binding = monitor_pair_binding(
            inputs.strategy_catalog,
            inputs.pair_registry,
            suite=suite,
        )
        if tier2_track == "ecological":
            # Ecological defenses do not execute the controlled action-monitor
            # pair.  Preserve their explicitly preregistered detector/no-action
            # family while still binding the pair IDs and profile digest.
            resolved_monitor_binding.pop("monitor_family")
        selected_groups: Sequence[str] | None = None
        frozen_count: int | None = None
        freeze_hash: str | None = None
        development_hash: str | None = None
        primary_contrast: str | None = None
        selected_bundle_hash: str | None = None
        if dataset_split == "test":
            assert heldout_freeze is not None
            frozen_count = int(
                heldout_freeze["independent_unit_count_by_suite"][suite]
            )
            freeze_hash = str(heldout_freeze["freeze_hash"])
            development_hash = str(heldout_freeze["development_evidence_hash"])
            primary_contrast = str(heldout_freeze["primary_contrast_id"])
            selected_bundle_hash = str(
                heldout_freeze["selected_test_bundle_hash_by_suite"][suite]
            )
            selected_by_suite = heldout_freeze.get(
                "selected_structural_group_ids_by_suite"
            )
            if isinstance(selected_by_suite, Mapping):
                raw_selected = selected_by_suite.get(suite)
                if isinstance(raw_selected, list):
                    selected_groups = tuple(str(item) for item in raw_selected)
        bundles = scenario_bundles(
            inputs,
            suite=suite,
            dataset_split=dataset_split,
            groups_per_bundle=(frozen_count or groups_per_bundle),
            selected_group_ids=selected_groups,
            eligible_scenario_ids=eligible_scenarios,
        )
        if dataset_split == "test":
            if len(bundles) != 1 or bundles[0].bundle_hash != selected_bundle_hash:
                raise AgentDojoGridError(
                    f"{suite} test scenarios do not match the frozen selected bundle"
                )
        for bundle, replicate in itertools.product(bundles, replicate_values):
            task_cells: list[GridCell] = []
            for cell in cells:
                values = {
                    **dict(base),
                    **cell,
                    # These values are derived from the validated frozen
                    # artifacts, never accepted from a grid-plan placeholder.
                    **resolved_monitor_binding,
                    "experiment_id": experiment_id,
                    "tier2_track": tier2_track,
                    "agentdojo_suite": suite,
                    "dataset_split": dataset_split,
                    "agentdojo_catalog_hash": inputs.upstream.catalog_hash,
                    "scenario_registry_revision": inputs.upstream.scenario_registry_revision,
                    "scenario_registry_hash": inputs.upstream.scenario_registry_hash,
                    "split_manifest_hash": inputs.upstream.split_manifest_hash,
                    "candidate_strategy_catalog_hash": inputs.upstream.candidate_strategy_catalog_hash,
                    "pair_registry_hash": inputs.upstream.pair_registry_hash,
                    "scenario_bundle_hash": bundle.bundle_hash,
                    "scenario_ids": bundle.scenario_ids,
                    "structural_group_ids": bundle.structural_group_ids,
                    "analysis_plan_hash": inputs.upstream.analysis_plan_hash,
                    "dependency_lock_hash": inputs.upstream.dependency_lock_hash,
                    "replicate": replicate,
                    "system_prompt_hash": (
                        ECOLOGICAL_SYSTEM_PROMPT_HASH
                        if tier2_track == "ecological"
                        else CONTROLLED_SYSTEM_PROMPT_HASH
                    ),
                    "models": _models_for_cell(
                        models,
                        experiment_id=experiment_id,
                        values={
                            **dict(base),
                            **cell,
                            **resolved_monitor_binding,
                        },
                        strategy_catalog=inputs.strategy_catalog,
                    ),
                    "sample_size_freeze_hash": freeze_hash,
                    "development_evidence_hash": development_hash,
                    "frozen_independent_unit_count": frozen_count,
                    "primary_contrast_id": primary_contrast,
                    "selected_test_bundle_hash": selected_bundle_hash,
                }
                config = AgentDojoExperimentConfig(**values)
                scientific = config.scientific_dict()
                config_hash = config.configuration_hash
                shard_id = stable_hash(
                    {
                        "configuration_hash": config_hash,
                        "scenario_bundle_hash": bundle.bundle_hash,
                    }
                )
                task_cells.append(
                    GridCell(cell_index, scientific, config_hash, shard_id)
                )
                cell_index += 1
            tasks.append(
                GridTask(
                    task_id=len(tasks),
                    suite=suite,
                    scenario_bundle_hash=bundle.bundle_hash,
                    replicate=replicate,
                    cells=tuple(task_cells),
                )
            )
    if not tasks:
        raise AgentDojoGridError("AgentDojo grid expansion produced no tasks")
    return AgentDojoGrid(
        experiment_id=experiment_id,
        tier2_track=tier2_track,
        dataset_split=dataset_split,
        tasks=tuple(tasks),
        upstream_binding_hash=inputs.upstream.binding_hash,
        heldout_freeze_binding=heldout_binding,
        protocol_disposition=(
            str(pair_disposition) if estimation_only else "legacy_full_catalog"
        ),
        action_eligibility_manifest_hash=action_eligibility_hash,
    )


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(dict(record)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(grid: AgentDojoGrid, path: Path | str) -> None:
    _atomic_write_jsonl(Path(path), grid.records())


def load_grid_manifest(path: Path | str) -> dict[str, Any]:
    candidate = Path(path)
    try:
        lines = candidate.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AgentDojoGridError(f"cannot read grid manifest {candidate}: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        raise AgentDojoGridError("AgentDojo grid manifest is empty or contains blanks")
    try:
        records = [json.loads(line) for line in lines]
    except json.JSONDecodeError as error:
        raise AgentDojoGridError(f"invalid AgentDojo grid JSON: {error}") from error
    if any(not isinstance(row, dict) for row in records):
        raise AgentDojoGridError("AgentDojo grid records must be objects")
    metadata = records[0]
    members = records[1:]
    if metadata.get("record_type") != "grid_metadata" or metadata.get(
        "schema_version"
    ) != GRID_SCHEMA_VERSION:
        raise AgentDojoGridError("invalid AgentDojo grid metadata")
    if metadata.get("environment_backend") != "agentdojo" or metadata.get(
        "model_free"
    ) is not True:
        raise AgentDojoGridError("grid metadata lacks AgentDojo/model-free identity")
    recorded_protocol_disposition = metadata.get("protocol_disposition")
    protocol_disposition = (
        "legacy_full_catalog"
        if recorded_protocol_disposition is None
        else recorded_protocol_disposition
    )
    action_eligibility_hash = metadata.get(
        "action_eligibility_manifest_hash"
    )
    if is_estimation_only_protocol_disposition(protocol_disposition):
        require_hash(
            "action_eligibility_manifest_hash", str(action_eligibility_hash)
        )
    elif (
        protocol_disposition != "legacy_full_catalog"
        or action_eligibility_hash is not None
    ):
        raise AgentDojoGridError("grid has an invalid protocol disposition")
    dataset_split = metadata.get("dataset_split")
    if (
        is_estimation_only_protocol_disposition(protocol_disposition)
        and dataset_split == "test"
    ):
        raise AgentDojoGridError(
            "estimation-only grid contains held-out execution"
        )
    heldout_binding = metadata.get("heldout_freeze_binding")
    if dataset_split == "test":
        if metadata.get("heldout_status") != "freeze_bound_before_test" or not isinstance(
            heldout_binding, Mapping
        ):
            raise AgentDojoGridError("held-out grid metadata lacks its freeze binding")
        for name in (
            "freeze_hash",
            "development_analysis_manifest_hash",
            "development_evidence_hash",
            "power_evidence_hash",
        ):
            require_hash(f"heldout_freeze_binding.{name}", str(heldout_binding.get(name)))
        if heldout_binding.get("claim_disposition") not in {
            "confirmatory_power_target_met",
            "underpowered_estimation_only",
        }:
            raise AgentDojoGridError("held-out grid has an invalid claim disposition")
        for name in (
            "independent_unit_count_by_suite",
            "available_test_independent_unit_count_by_suite",
            "selected_test_bundle_hash_by_suite",
            "selected_structural_group_ids_by_suite",
        ):
            raw = heldout_binding.get(name)
            if not isinstance(raw, Mapping) or set(raw) != set(AGENTDOJO_SUITES):
                raise AgentDojoGridError(
                    f"held-out grid binding has invalid {name}"
                )
    elif metadata.get("heldout_status") != "not_applicable_non_test" or heldout_binding is not None:
        raise AgentDojoGridError("non-test grid contains held-out freeze metadata")
    total_tasks = metadata.get("total_tasks")
    total_configurations = metadata.get("total_configurations")
    if not isinstance(total_tasks, int) or total_tasks <= 0:
        raise AgentDojoGridError("grid total_tasks must be positive")
    if not isinstance(total_configurations, int) or total_configurations != len(members):
        raise AgentDojoGridError("grid configuration count is incorrect")
    if metadata.get("valid_array_range") != f"0-{total_tasks - 1}":
        raise AgentDojoGridError("grid array range is incorrect")
    task_members: list[list[tuple[str, str]]] = [[] for _ in range(total_tasks)]
    task_meta: list[tuple[str, str, int] | None] = [None] * total_tasks
    seen: set[tuple[str, str]] = set()
    for index, member in enumerate(members):
        if member.get("record_type") != "grid_member" or member.get(
            "schema_version"
        ) != GRID_SCHEMA_VERSION or member.get("cell_index") != index:
            raise AgentDojoGridError("grid members are not canonical and contiguous")
        configuration = member.get("configuration")
        if not isinstance(configuration, dict):
            raise AgentDojoGridError("grid member lacks configuration")
        config = AgentDojoExperimentConfig.from_mapping(configuration)
        if config.scientific_dict() != configuration:
            raise AgentDojoGridError("grid member configuration is not canonical")
        for name in ("experiment_id", "tier2_track", "dataset_split"):
            if getattr(config, name) != metadata.get(name):
                raise AgentDojoGridError(
                    f"grid member {name} disagrees with grid metadata"
                )
        if dataset_split == "test":
            assert isinstance(heldout_binding, Mapping)
            suite = config.agentdojo_suite
            expected_count = heldout_binding["independent_unit_count_by_suite"].get(
                suite
            )
            expected_bundle = heldout_binding["selected_test_bundle_hash_by_suite"].get(
                suite
            )
            expected_groups = heldout_binding[
                "selected_structural_group_ids_by_suite"
            ].get(suite)
            if (
                config.sample_size_freeze_hash != heldout_binding.get("freeze_hash")
                or config.development_evidence_hash
                != heldout_binding.get("development_evidence_hash")
                or config.primary_contrast_id
                != heldout_binding.get("primary_contrast_id")
                or config.frozen_independent_unit_count != expected_count
                or config.selected_test_bundle_hash != expected_bundle
                or list(config.structural_group_ids) != expected_groups
            ):
                raise AgentDojoGridError(
                    "held-out grid member disagrees with metadata freeze semantics"
                )
        configuration_hash = str(member.get("configuration_hash", ""))
        if configuration_hash != config.configuration_hash:
            raise AgentDojoGridError("grid member configuration hash is invalid")
        shard_id = str(member.get("shard_id", ""))
        if shard_id != stable_hash(
            {
                "configuration_hash": configuration_hash,
                "scenario_bundle_hash": config.scenario_bundle_hash,
            }
        ):
            raise AgentDojoGridError("grid member shard ID is invalid")
        identity = (configuration_hash, shard_id)
        if identity in seen:
            raise AgentDojoGridError("grid repeats a member identity")
        seen.add(identity)
        task_id = member.get("task_id")
        if not isinstance(task_id, int) or task_id < 0 or task_id >= total_tasks:
            raise AgentDojoGridError("grid member task ID is invalid")
        if member.get("batch_offset") != len(task_members[task_id]):
            raise AgentDojoGridError("grid batch offsets are not contiguous")
        task_members[task_id].append(identity)
        current = (config.agentdojo_suite, config.scenario_bundle_hash, config.replicate)
        if task_meta[task_id] is None:
            task_meta[task_id] = current
        elif task_meta[task_id] != current:
            raise AgentDojoGridError("array task does not contain one matched scenario bundle")
    if any(not members_for_task for members_for_task in task_members):
        raise AgentDojoGridError("grid task IDs are incomplete")
    observed_suites = {item[0] for item in task_meta if item is not None}
    full_suite_coverage = observed_suites == set(AGENTDOJO_SUITES)
    estimation_only = is_estimation_only_protocol_disposition(
        protocol_disposition
    )
    expected_coverage_status = (
        "full_four_suite_estimation_only"
        if full_suite_coverage and estimation_only
        else "full_four_suite"
        if full_suite_coverage
        else "development_subset_nonconfirmatory"
    )
    expected_confirmatory_coverage = full_suite_coverage and not estimation_only
    if metadata.get("suite_coverage_status") != expected_coverage_status or metadata.get(
        "confirmatory_suite_coverage_eligible"
    ) is not expected_confirmatory_coverage:
        raise AgentDojoGridError("grid suite-coverage metadata is inconsistent")
    if dataset_split == "test" and not full_suite_coverage:
        raise AgentDojoGridError("held-out grid does not contain all four suites")
    grid_hash_payload = {
            "schema_version": GRID_SCHEMA_VERSION,
            "experiment_id": metadata["experiment_id"],
            "tier2_track": metadata["tier2_track"],
            "dataset_split": metadata["dataset_split"],
            "upstream_binding_hash": metadata["upstream_binding_hash"],
            "heldout_freeze_binding": heldout_binding,
            "tasks": [
                {
                    "task_id": task_id,
                    "suite": task_meta[task_id][0],
                    "scenario_bundle_hash": task_meta[task_id][1],
                    "replicate": task_meta[task_id][2],
                    "members": task_members[task_id],
                }
                for task_id in range(total_tasks)
            ],
        }
    if recorded_protocol_disposition is not None:
        grid_hash_payload["protocol_disposition"] = protocol_disposition
        grid_hash_payload["action_eligibility_manifest_hash"] = (
            action_eligibility_hash
        )
    calculated = stable_hash(grid_hash_payload)
    if metadata.get("grid_hash") != calculated:
        raise AgentDojoGridError("AgentDojo grid_hash is invalid")
    return {"metadata": metadata, "members": members}


def _grid_from_args(args: argparse.Namespace) -> AgentDojoGrid:
    inputs = load_frozen_inputs(
        catalog_path=args.catalog,
        splits_path=args.splits,
        strategy_catalog_path=args.strategy_catalog,
        pair_registry_path=args.pair_registry,
        analysis_plan_path=args.analysis_plan,
        dependency_lock_path=args.dependency_lock,
    )
    plan = load_json_object(args.grid_plan, label="AgentDojo grid plan")
    freeze = (
        load_json_object(args.sample_size_freeze, label="AgentDojo sample-size freeze")
        if args.sample_size_freeze is not None
        else None
    )
    return build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id=args.experiment,
        tier2_track=args.track,
        dataset_split=args.dataset_split,
        suites=args.suites or AGENTDOJO_SUITES,
        replicates=args.replicates or (0,),
        groups_per_bundle=args.groups_per_bundle,
        sample_size_freeze=freeze,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, choices=AGENTDOJO_EXPERIMENTS)
    parser.add_argument("--track", required=True, choices=("controlled", "ecological"))
    parser.add_argument("--dataset-split", required=True, choices=("train", "development", "test"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--strategy-catalog", type=Path, required=True)
    parser.add_argument("--pair-registry", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--grid-plan", type=Path, required=True)
    parser.add_argument("--sample-size-freeze", type=Path)
    parser.add_argument("--suite", dest="suites", action="append")
    parser.add_argument("--replicate", dest="replicates", type=int, action="append")
    parser.add_argument("--groups-per-bundle", type=int, default=8)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-free AgentDojo grid orchestration")
    children = parser.add_subparsers(dest="command", required=True)
    for command in ("print", "count", "member-count", "hash", "manifest", "select"):
        child = children.add_parser(command)
        _add_arguments(child)
        if command == "manifest":
            child.add_argument("--output", type=Path)
        if command == "select":
            child.add_argument("--task-id", type=int, required=True)
            child.add_argument("--format", choices=("jsonl", "env-nul"), default="jsonl")
    return parser


def _emit_env_nul(grid: AgentDojoGrid, task_id: int) -> None:
    output = sys.stdout.buffer
    task = grid.task(task_id)
    for offset, cell in enumerate(task.cells):
        values = {
            "grid_hash": grid.grid_hash,
            "grid_task_id": task_id,
            "grid_batch_hash": task.batch_hash,
            "batch_offset": offset,
            "configuration_hash": cell.configuration_hash,
            "shard_id": cell.shard_id,
            "configuration_json": canonical_json(cell.configuration),
            **cell.configuration,
        }
        for key, value in values.items():
            if isinstance(value, (list, dict)):
                value = canonical_json(value)
            elif value is None:
                value = ""
            output.write(str(key).encode() + b"\0" + str(value).encode() + b"\0")
        output.write(b"__CELL_END__\0")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        grid = _grid_from_args(args)
        if args.command == "count":
            print(len(grid.tasks))
        elif args.command == "member-count":
            print(len(grid.cells))
        elif args.command == "hash":
            print(grid.grid_hash)
        elif args.command == "print":
            for key, value in grid.metadata().items():
                if key not in {"record_type", "schema_version"}:
                    print(f"{key}={value}")
            for task in grid.tasks:
                for offset, cell in enumerate(task.cells):
                    factors = " ".join(
                        f"{key}={shlex.quote(str(value))}"
                        for key, value in cell.configuration.items()
                        if value is not None and not isinstance(value, (list, dict))
                    )
                    print(
                        f"task_id={task.task_id} batch_offset={offset} "
                        f"batch_hash={task.batch_hash} grid_hash={grid.grid_hash} "
                        f"configuration_hash={cell.configuration_hash} shard_id={cell.shard_id} "
                        f"{factors}"
                    )
        elif args.command == "manifest":
            if args.output is None:
                for record in grid.records():
                    print(canonical_json(record))
            else:
                write_manifest(grid, args.output)
                print(args.output)
        elif args.command == "select":
            if args.format == "env-nul":
                _emit_env_nul(grid, args.task_id)
            else:
                task = grid.task(args.task_id)
                for offset, cell in enumerate(task.cells):
                    print(canonical_json(cell.record(task_id=task.task_id, batch_offset=offset)))
    except (AgentDojoConfigError, AgentDojoGridError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "GRID_PLAN_SCHEMA_VERSION",
    "GRID_SCHEMA_VERSION",
    "PAIR_SCHEMA_VERSION",
    "SPLITS_SCHEMA_VERSION",
    "STRATEGY_SCHEMA_VERSION",
    "STRATEGY_SCHEMA_VERSION_V2",
    "STRATEGY_SCHEMA_VERSION_RECIPIENT_SEPARATION",
    "AgentDojoGrid",
    "AgentDojoGridError",
    "FrozenInputs",
    "GridCell",
    "GridTask",
    "ScenarioBundle",
    "build_grid",
    "is_estimation_only_protocol_disposition",
    "load_frozen_inputs",
    "load_grid_manifest",
    "scenario_bundles",
    "validate_grid_manifest_coverage",
    "validate_hashed_document",
    "validate_structural_splits",
    "write_manifest",
]
