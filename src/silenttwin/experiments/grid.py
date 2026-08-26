"""Deterministic, hash-bound experiment grids and pilot presets.

This module is deliberately model-free: inspecting a Tier-2 grid must never
load a checkpoint, contact a provider, or require a GPU.  The Bash launchers
use the command-line interface here as the single source of truth for grid
printing, array selection, batching, and the expected aggregate manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING, dataclass, fields
import itertools
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from silenttwin.config import DEFAULT_NUM_SAMPLES, ExperimentConfig, canonical_json, stable_hash
from silenttwin.metrics.power import validate_sample_size_freeze


GRID_SCHEMA_VERSION = "silenttwin.grid.v1"
PILOT_SCHEMA_VERSION = "silenttwin.pilot.v1"
EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5")
TIER2_PRESETS = ("pilot_c", "pilot_d")
_OPERATIONAL_CONFIG_FIELDS = {
    "output_dir",
    "overwrite",
    "grid_hash",
    "grid_task_id",
    "shard_id",
    "pilot_id",
}


class GridError(ValueError):
    """Raised when a grid or pilot preset is invalid."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dataclass_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field in fields(ExperimentConfig):
        if field.name in _OPERATIONAL_CONFIG_FIELDS:
            continue
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            defaults[field.name] = field.default_factory()  # type: ignore[misc]
    return defaults


def scientific_configuration(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical scientific configuration used for cell hashing.

    We ask :class:`ExperimentConfig` to perform its usual normalization,
    guaranteeing the printed hash matches the run manifest.  Constructing a
    Tier-2 configuration is model-free and therefore safe during grid
    inspection; checkpoint loading happens only in the run stage.
    """

    values = _dataclass_defaults()
    values.update({key: value for key, value in overrides.items() if value is not None})
    if values.get("num_samples") == -1:
        values["num_samples"] = DEFAULT_NUM_SAMPLES

    field_names = {field.name for field in fields(ExperimentConfig)}
    constructor_values = {
        key: value
        for key, value in values.items()
        if key in field_names and key not in _OPERATIONAL_CONFIG_FIELDS
    }
    # A canonical Tier-2 grid intentionally omits the operational cache path.
    # Supply a validation-only locator while normalizing an already-canonical
    # manifest; the run wrapper injects the operator's persistent cache later.
    if constructor_values.get("tier") == "tier2" and not constructor_values.get(
        "model_cache_dir"
    ):
        constructor_values["model_cache_dir"] = "/operational/cache/not-hashed"
    normalized = ExperimentConfig(
        **constructor_values,
        output_dir=Path("."),
        overwrite=False,
    ).scientific_dict()
    return _jsonable(normalized)


@dataclass(frozen=True, slots=True)
class GridCell:
    cell_index: int
    shard_index: int
    shard_count: int
    sample_start: int
    shard_id: str
    configuration: Mapping[str, Any]
    configuration_hash: str

    def member_record(self, *, task_id: int, batch_offset: int) -> dict[str, Any]:
        return {
            "record_type": "grid_member",
            "schema_version": GRID_SCHEMA_VERSION,
            "task_id": task_id,
            "batch_offset": batch_offset,
            "cell_index": self.cell_index,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "sample_start": self.sample_start,
            "shard_id": self.shard_id,
            "configuration_hash": self.configuration_hash,
            "configuration": dict(self.configuration),
        }


@dataclass(frozen=True, slots=True)
class GridTask:
    task_id: int
    cells: tuple[GridCell, ...]

    @property
    def batch_hash(self) -> str:
        return stable_hash(
            [
                {
                    "configuration_hash": cell.configuration_hash,
                    "shard_id": cell.shard_id,
                }
                for cell in self.cells
            ]
        )


@dataclass(frozen=True, slots=True)
class ExperimentGrid:
    experiment: str
    pilot_id: str | None
    factor_order: tuple[str, ...]
    cells_per_task: int
    tasks: tuple[GridTask, ...]

    @property
    def cells(self) -> tuple[GridCell, ...]:
        return tuple(cell for task in self.tasks for cell in task.cells)

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def total_configurations(self) -> int:
        return sum(len(task.cells) for task in self.tasks)

    @property
    def grid_hash(self) -> str:
        return _calculate_grid_hash(
            experiment=self.experiment,
            pilot_id=self.pilot_id,
            factor_order=self.factor_order,
            cells_per_task=self.cells_per_task,
            task_members=tuple(
                tuple((cell.configuration_hash, cell.shard_id) for cell in task.cells)
                for task in self.tasks
            ),
        )

    def metadata_record(self) -> dict[str, Any]:
        return {
            "record_type": "grid_metadata",
            "schema_version": GRID_SCHEMA_VERSION,
            "experiment_id": self.experiment,
            "pilot_id": self.pilot_id,
            "factor_order": list(self.factor_order),
            "cells_per_task": self.cells_per_task,
            "total_tasks": self.total_tasks,
            "valid_array_range": f"0-{self.total_tasks - 1}",
            "total_configurations": self.total_configurations,
            "grid_hash": self.grid_hash,
        }

    def manifest_records(self) -> list[dict[str, Any]]:
        records = [self.metadata_record()]
        for task in self.tasks:
            records.extend(
                cell.member_record(task_id=task.task_id, batch_offset=offset)
                for offset, cell in enumerate(task.cells)
            )
        return records

    def task(self, task_id: int) -> GridTask:
        if task_id < 0 or task_id >= self.total_tasks:
            raise GridError(
                f"array task id {task_id} is out of range; valid range is "
                f"0-{self.total_tasks - 1}"
            )
        return self.tasks[task_id]


def _validate_unique(name: str, values: Sequence[Any]) -> tuple[Any, ...]:
    if not values:
        raise GridError(f"{name} must contain at least one value")
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            raise GridError(f"{name} contains duplicate value {value!r}")
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise GridError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise GridError(f"{name} must be a positive integer") from error
    if integer <= 0:
        raise GridError(f"{name} must be a positive integer")
    return integer


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise GridError(f"{name} must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise GridError(f"{name} must be a non-negative integer") from error
    if integer < 0:
        raise GridError(f"{name} must be a non-negative integer")
    return integer


def _condition_budget_pairs(
    query_budgets: Sequence[int],
    conditions: Sequence[str],
    explicit: Mapping[str, Sequence[int]] | None = None,
) -> tuple[tuple[int, str], ...]:
    budgets = _validate_unique("query_budgets", tuple(int(value) for value in query_budgets))
    condition_values = _validate_unique("conditions", tuple(str(value) for value in conditions))
    pairs: list[tuple[int, str]] = []
    if explicit is not None:
        unknown = set(explicit) - set(condition_values)
        if unknown:
            raise GridError(f"condition_query_budgets has unknown conditions: {sorted(unknown)}")
        for budget in budgets:
            for condition in condition_values:
                allowed = tuple(int(value) for value in explicit.get(condition, ()))
                if budget in allowed:
                    pairs.append((budget, condition))
    else:
        for budget in budgets:
            for condition in condition_values:
                normalized = "adaptive" if condition == "genuine" else condition
                valid = (
                    (normalized in {"no_probe", "oracle"} and budget == 0)
                    or (normalized in {"adaptive", "shuffled", "random"} and budget > 0)
                )
                if valid:
                    pairs.append((budget, condition))
    if not pairs:
        raise GridError("the E2 condition/query-budget rules produce an empty grid")
    return tuple(pairs)


def _feedback_budget_pairs(
    query_budgets: Sequence[int],
    feedback_sources: Sequence[str],
    explicit: Mapping[str, Sequence[int]] | None = None,
) -> tuple[tuple[int, str], ...]:
    """Return the declared E1 feedback controls without meaningless crosses.

    Genuine feedback is meaningful at every supported budget.  Shuffled and
    constant controls require at least one callback, so a default custom grid
    omits their Q=0 duplicates.  Pilot presets use an explicit mapping to keep
    only their preregistered control budget.
    """

    budgets = _validate_unique("query_budgets", tuple(int(value) for value in query_budgets))
    source_values = _validate_unique(
        "feedback_sources", tuple(str(value) for value in feedback_sources)
    )
    pairs: list[tuple[int, str]] = []
    if explicit is not None:
        unknown = set(explicit) - set(source_values)
        if unknown:
            raise GridError(
                "feedback_source_query_budgets has unknown feedback sources: "
                f"{sorted(unknown)}"
            )
        for budget in budgets:
            for source in source_values:
                allowed = tuple(int(value) for value in explicit.get(source, ()))
                if budget in allowed:
                    pairs.append((budget, source))
    else:
        for budget in budgets:
            for source in source_values:
                if source == "genuine" or budget > 0:
                    pairs.append((budget, source))
    if not pairs:
        raise GridError("the E1 feedback-source/query-budget rules produce an empty grid")
    unrepresented = set(source_values) - {source for _, source in pairs}
    if unrepresented:
        raise GridError(
            "feedback-source/query-budget rules omit declared feedback sources: "
            f"{sorted(unrepresented)}"
        )
    return tuple(pairs)


def _split_shards(total_samples: int, episodes_per_shard: int | None) -> tuple[tuple[int, int], ...]:
    if total_samples == -1:
        total_samples = DEFAULT_NUM_SAMPLES
    total = _positive_integer("num_samples", total_samples)
    if episodes_per_shard is None or episodes_per_shard <= 0 or episodes_per_shard >= total:
        return ((0, total),)
    shard_size = _positive_integer("episodes_per_shard", episodes_per_shard)
    return tuple(
        (start, min(shard_size, total - start)) for start in range(0, total, shard_size)
    )


def _shard_id(
    *,
    configuration_hash: str,
    shard_index: int,
    shard_count: int,
    sample_start: int,
) -> str:
    return stable_hash(
        {
            "configuration_hash": configuration_hash,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "sample_start": sample_start,
        }
    )


def _calculate_grid_hash(
    *,
    experiment: str,
    pilot_id: str | None,
    factor_order: Sequence[str],
    cells_per_task: int,
    task_members: Sequence[Sequence[tuple[str, str]]],
) -> str:
    return stable_hash(
        {
            "schema_version": GRID_SCHEMA_VERSION,
            "experiment": experiment,
            "pilot_id": pilot_id,
            "factor_order": list(factor_order),
            "cells_per_task": cells_per_task,
            "tasks": [
                {
                    "task_id": task_id,
                    "members": [
                        {"configuration_hash": configuration_hash, "shard_id": shard_id}
                        for configuration_hash, shard_id in members
                    ],
                }
                for task_id, members in enumerate(task_members)
            ],
        }
    )


def build_grid(
    *,
    experiment: str,
    tiers: Sequence[str],
    world_suites: Sequence[str],
    runtimes: Sequence[str],
    attackers: Sequence[str],
    query_budgets: Sequence[int],
    seeds: Sequence[int],
    num_samples: int,
    conditions: Sequence[str] = (),
    condition_query_budgets: Mapping[str, Sequence[int]] | None = None,
    feedback_sources: Sequence[str] = ("genuine",),
    feedback_source_query_budgets: Mapping[str, Sequence[int]] | None = None,
    workflows: Sequence[str] = (),
    ablations: Sequence[str] = (),
    pair_families: Sequence[str | None] = (None,),
    template_ids: Sequence[str | None] = (None,),
    decoding_seeds: Sequence[int | None] = (None,),
    dataset_split: str | None = None,
    dataset_revision: str | None = None,
    model: Mapping[str, Any] | None = None,
    cells_per_task: int = 1,
    episodes_per_shard: int | None = None,
    e1_paired_instances: int | None = None,
    e1_public_instances_per_shard: int | None = None,
    pilot_id: str | None = None,
    sample_size_freeze: Mapping[str, Any] | None = None,
) -> ExperimentGrid:
    if experiment not in EXPERIMENTS:
        raise GridError(f"unknown experiment {experiment!r}")
    tier_values = _validate_unique("tiers", tuple(str(value) for value in tiers))
    world_values = _validate_unique("world_suites", tuple(str(value) for value in world_suites))
    runtime_values = _validate_unique("runtimes", tuple(str(value) for value in runtimes))
    attacker_values = _validate_unique("attackers", tuple(str(value) for value in attackers))
    budget_values = _validate_unique(
        "query_budgets", tuple(_nonnegative_integer("query budget", value) for value in query_budgets)
    )
    seed_values = _validate_unique(
        "seeds", tuple(_nonnegative_integer("seed", value) for value in seeds)
    )
    pair_values = _validate_unique("pair_families", tuple(pair_families))
    template_values = _validate_unique("template_ids", tuple(template_ids))
    decoding_values = _validate_unique(
        "decoding_seeds",
        tuple(
            None if value is None else _nonnegative_integer("decoding seed", value)
            for value in decoding_seeds
        ),
    )
    batch_size = _positive_integer("cells_per_task", cells_per_task)

    effective_split = dataset_split or "development"
    freeze_fields: dict[str, Any] = {}
    frozen_public_instances: int | None = None
    if effective_split == "test":
        if experiment not in {"e1", "e2"}:
            raise GridError("frozen held-out grids are currently defined only for E1/E2")
        if sample_size_freeze is None:
            raise GridError(
                "held-out grid generation requires --sample-size-freeze before any test inspection"
            )
        contrast_id = sample_size_freeze.get("contrast_id")
        development_hash = sample_size_freeze.get("development_manifest_hash")
        if not isinstance(contrast_id, str) or not isinstance(development_hash, str):
            raise GridError("sample-size freeze lacks contrast/development identifiers")
        try:
            frozen_public_instances = validate_sample_size_freeze(
                sample_size_freeze,
                experiment_id=experiment,
                dataset_revision=str(dataset_revision or ""),
                contrast_id=contrast_id,
                development_manifest_hash=development_hash,
            )
        except ValueError as error:
            raise GridError(f"invalid sample-size freeze: {error}") from error
        freeze_hash = sample_size_freeze.get("freeze_hash")
        if not isinstance(freeze_hash, str):
            raise GridError("sample-size freeze lacks a validated hash")
        freeze_fields = {
            "sample_size_freeze_hash": freeze_hash,
            "development_manifest_hash": development_hash,
            "frozen_public_instances": frozen_public_instances,
            "primary_contrast_id": contrast_id,
        }
    elif sample_size_freeze is not None:
        raise GridError("sample-size freezes are valid only for dataset_split=test")

    if experiment == "e2":
        condition_values = _validate_unique("conditions", tuple(str(value) for value in conditions))
        budget_condition_feedback_values: Sequence[tuple[int, str | None, str | None]] = tuple(
            (budget, condition, None)
            for budget, condition in _condition_budget_pairs(
                budget_values, condition_values, condition_query_budgets
            )
        )
    elif experiment == "e1":
        budget_condition_feedback_values = tuple(
            (budget, None, feedback_source)
            for budget, feedback_source in _feedback_budget_pairs(
                budget_values,
                feedback_sources,
                feedback_source_query_budgets,
            )
        )
    else:
        budget_condition_feedback_values = tuple(
            (budget, None, None) for budget in budget_values
        )

    workflow_values: tuple[str | None, ...]
    if experiment == "e4":
        workflow_values = _validate_unique("workflows", tuple(str(value) for value in workflows))
    else:
        workflow_values = (None,)
    ablation_values: tuple[str | None, ...]
    if experiment == "e5":
        ablation_values = _validate_unique("ablations", tuple(str(value) for value in ablations))
    else:
        ablation_values = (None,)

    if e1_paired_instances is not None and experiment != "e1":
        raise GridError("e1_paired_instances is only valid for E1")
    if e1_public_instances_per_shard is not None and e1_paired_instances is None:
        raise GridError(
            "e1_public_instances_per_shard requires e1_paired_instances"
        )
    e1_instance_count = (
        None
        if e1_paired_instances is None
        else _positive_integer("e1_paired_instances", e1_paired_instances)
    )
    if experiment == "e1" and frozen_public_instances is not None:
        if e1_instance_count is not None and e1_instance_count != frozen_public_instances:
            raise GridError("pilot/public-instance count disagrees with sample-size freeze")
        e1_instance_count = frozen_public_instances
    e1_shard_instance_count = (
        None
        if e1_public_instances_per_shard is None
        else _positive_integer(
            "e1_public_instances_per_shard", e1_public_instances_per_shard
        )
    )
    if (
        e1_instance_count is not None
        and e1_shard_instance_count is not None
        and e1_shard_instance_count > e1_instance_count
    ):
        raise GridError("e1_public_instances_per_shard cannot exceed e1_paired_instances")

    model_values = dict(model or {})
    base_cells: list[GridCell] = []
    factor_product = itertools.product(
        tier_values,
        world_values,
        runtime_values,
        attacker_values,
        budget_condition_feedback_values,
        workflow_values,
        ablation_values,
        pair_values,
        template_values,
        seed_values,
        decoding_values,
    )
    for (
        tier,
        world_suite,
        runtime,
        attacker,
        budget_condition_feedback,
        workflow,
        ablation,
        pair_family,
        template_id,
        seed,
        decoding_seed,
    ) in factor_product:
        query_budget, condition, feedback_source = budget_condition_feedback
        normalized_condition = "genuine" if condition == "adaptive" else condition
        # Opaque termination can retire the target session during adaptive
        # target-state probing, before E2's mandatory one-final-action slot.
        # Keep these known-invalid crosses out of the grid rather than making
        # a valid default grid fail midway through expansion.
        if (
            experiment == "e2"
            and runtime == "opaque_termination"
            and normalized_condition in {"genuine", "random"}
            and query_budget > 0
        ):
            continue
        base = {
            "experiment": experiment,
            "tier": tier,
            "world_suite": world_suite,
            "runtime": runtime,
            "attacker": attacker,
            "query_budget": query_budget,
            "seed": seed,
            "feedback_source": feedback_source,
            "condition": condition,
            "workflow": workflow,
            "ablation": ablation,
            "pair_family": pair_family,
            "template_id": template_id,
            "dataset_split": dataset_split,
            "dataset_revision": dataset_revision,
            "decoding_seed": decoding_seed,
            **freeze_fields,
            **model_values,
        }
        if e1_instance_count is not None:
            rows_per_instance = 4 if feedback_source == "shuffled" else 2
            source_num_samples = e1_instance_count * rows_per_instance
            source_episodes_per_shard = (
                episodes_per_shard
                if e1_shard_instance_count is None
                else None
                if e1_shard_instance_count == e1_instance_count
                # The preset's shard size is declared in ordinary E1
                # two-state rows.  Holding trial rows (and therefore model
                # work) fixed means shuffled shards contain half as many
                # public tasks, each with the required four-cell cross.
                else e1_shard_instance_count * 2
            )
        elif experiment == "e2" and frozen_public_instances is not None:
            source_num_samples = frozen_public_instances * 4
            source_episodes_per_shard = episodes_per_shard
        else:
            source_num_samples = num_samples
            source_episodes_per_shard = episodes_per_shard
        shard_ranges = _split_shards(
            source_num_samples, source_episodes_per_shard
        )
        shard_count = len(shard_ranges)
        for shard_index, (sample_start, shard_samples) in enumerate(shard_ranges):
            overrides = {
                **base,
                "num_samples": shard_samples,
                "sample_start": sample_start,
            }
            configuration = scientific_configuration(overrides)
            configuration_hash = stable_hash(configuration)
            shard_id = _shard_id(
                configuration_hash=configuration_hash,
                shard_index=shard_index,
                shard_count=shard_count,
                sample_start=sample_start,
            )
            base_cells.append(
                GridCell(
                    cell_index=len(base_cells),
                    shard_index=shard_index,
                    shard_count=shard_count,
                    sample_start=sample_start,
                    shard_id=shard_id,
                    configuration=configuration,
                    configuration_hash=configuration_hash,
                )
            )
    if not base_cells:
        raise GridError("grid expansion produced no configurations")
    tasks = tuple(
        GridTask(task_id=task_id, cells=tuple(base_cells[start : start + batch_size]))
        for task_id, start in enumerate(range(0, len(base_cells), batch_size))
    )
    factor_order = ["tier", "world_suite", "runtime", "attacker", "query_budget"]
    if experiment == "e1":
        factor_order.append("feedback_source")
    if experiment == "e2":
        factor_order.append("condition")
    if experiment == "e4":
        factor_order.append("workflow")
    if experiment == "e5":
        factor_order.append("ablation")
    if pair_values != (None,):
        factor_order.append("pair_family")
    if template_values != (None,):
        factor_order.append("template_id")
    factor_order.append("seed")
    if decoding_values != (None,):
        factor_order.append("decoding_seed")
    if any(cell.shard_count > 1 for cell in base_cells):
        factor_order.append("shard_index")
    if frozen_public_instances is not None:
        factor_order.append("sample_size_freeze_hash")
    return ExperimentGrid(
        experiment=experiment,
        pilot_id=pilot_id,
        factor_order=tuple(factor_order),
        cells_per_task=batch_size,
        tasks=tasks,
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def pilot_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()
    name = str(name_or_path)
    if not name.endswith(".json"):
        name += ".json"
    if "/" in name or "\\" in name or name.startswith("."):
        raise GridError(f"invalid pilot preset name: {name_or_path!r}")
    path = _repository_root() / "configs" / "silenttwin" / "pilots" / name
    if not path.is_file():
        raise GridError(f"pilot preset does not exist: {path}")
    return path


def load_pilot(name_or_path: str | Path) -> dict[str, Any]:
    path = pilot_path(name_or_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GridError(f"cannot load pilot preset {path}: {error}") from error
    if not isinstance(value, dict):
        raise GridError(f"pilot preset is not a JSON object: {path}")
    if value.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise GridError(
            f"pilot preset {path} has schema {value.get('schema_version')!r}; "
            f"expected {PILOT_SCHEMA_VERSION!r}"
        )
    if not isinstance(value.get("experiments"), dict):
        raise GridError(f"pilot preset has no experiments object: {path}")
    return value


def _sequence(spec: Mapping[str, Any], key: str, default: Sequence[Any] = ()) -> tuple[Any, ...]:
    if key not in spec:
        return tuple(default)
    value = spec[key]
    if not isinstance(value, list):
        raise GridError(f"pilot field {key!r} must be a JSON array")
    return tuple(value)


def grid_from_pilot(
    name_or_path: str | Path,
    *,
    experiment: str,
    model_overrides: Mapping[str, Any] | None = None,
    decoding_seeds_override: Sequence[int] | None = None,
    dataset_split_override: str | None = None,
    dataset_revision_override: str | None = None,
    sample_size_freeze: Mapping[str, Any] | None = None,
) -> ExperimentGrid:
    preset = load_pilot(name_or_path)
    pilot_id = str(preset.get("pilot_id", ""))
    if not pilot_id:
        raise GridError("pilot preset has no pilot_id")
    experiments = preset["experiments"]
    if experiment not in experiments:
        raise GridError(f"pilot {pilot_id!r} does not define experiment {experiment!r}")
    experiment_spec = experiments[experiment]
    if not isinstance(experiment_spec, dict):
        raise GridError(f"pilot experiment {experiment!r} must be an object")
    tier = str(experiment_spec.get("tier", preset.get("tier", "tier1")))
    dataset_split = experiment_spec.get("dataset_split", preset.get("dataset_split"))
    dataset_revision = experiment_spec.get(
        "dataset_revision", preset.get("dataset_revision")
    )
    if dataset_split_override is not None:
        dataset_split = dataset_split_override
    if dataset_revision_override is not None:
        dataset_revision = dataset_revision_override
    if bool(preset.get("development_only")) and dataset_split != "development":
        raise GridError(f"development-only pilot {pilot_id!r} must use the development split")
    if sample_size_freeze is not None:
        raise GridError("development pilot presets cannot consume a held-out sample-size freeze")

    model = dict(preset.get("model", {}))
    model.update({key: value for key, value in (model_overrides or {}).items() if value is not None})
    if tier == "tier2":
        for key in ("model_id", "model_revision", "model_cache_dir"):
            if not model.get(key):
                raise GridError(
                    f"Tier-2 pilot {pilot_id!r} requires {key}; supply the matching environment variable"
                )
        if not dataset_revision:
            raise GridError(
                f"Tier-2 pilot {pilot_id!r} requires an immutable DATASET_REVISION"
            )

    condition_query_budgets = experiment_spec.get("condition_query_budgets")
    if condition_query_budgets is not None and not isinstance(condition_query_budgets, dict):
        raise GridError("condition_query_budgets must be an object")
    feedback_source_query_budgets = experiment_spec.get(
        "feedback_source_query_budgets"
    )
    if feedback_source_query_budgets is not None and not isinstance(
        feedback_source_query_budgets, dict
    ):
        raise GridError("feedback_source_query_budgets must be an object")
    paired_instances_value = experiment_spec.get("paired_instances")
    e1_paired_instances: int | None = None
    e1_public_instances_per_shard: int | None = None
    if paired_instances_value is None:
        num_samples = _positive_integer(
            "num_samples_per_cell", experiment_spec.get("num_samples_per_cell")
        )
        episodes_per_shard = experiment_spec.get("episodes_per_shard")
    else:
        paired_instances = _positive_integer("paired_instances", paired_instances_value)
        expected_rows_per_instance = 4 if experiment == "e2" else 2 if experiment == "e1" else 1
        rows_per_instance = _positive_integer(
            "trial_rows_per_paired_instance",
            experiment_spec.get(
                "trial_rows_per_paired_instance", expected_rows_per_instance
            ),
        )
        if rows_per_instance != expected_rows_per_instance:
            raise GridError(
                f"pilot {pilot_id!r} {experiment} must use "
                f"trial_rows_per_paired_instance={expected_rows_per_instance}"
            )
        num_samples = paired_instances * rows_per_instance
        instances_per_shard_value = experiment_spec.get(
            "public_instances_per_shard", paired_instances
        )
        instances_per_shard = _positive_integer(
            "public_instances_per_shard", instances_per_shard_value
        )
        if instances_per_shard > paired_instances:
            raise GridError("public_instances_per_shard cannot exceed paired_instances")
        episodes_per_shard = instances_per_shard * rows_per_instance
        if experiment == "e1":
            # Genuine E1 uses two balanced target-state rows per public task;
            # shuffled E1 uses a four-row target/donor cross.  Passing public
            # instance counts lets build_grid preserve the same task cohort
            # while assigning the correct row count to each source.
            e1_paired_instances = paired_instances
            e1_public_instances_per_shard = instances_per_shard
    return build_grid(
        experiment=experiment,
        tiers=(tier,),
        world_suites=_sequence(experiment_spec, "world_suites"),
        runtimes=_sequence(experiment_spec, "runtimes"),
        attackers=_sequence(experiment_spec, "attackers"),
        query_budgets=_sequence(experiment_spec, "query_budgets"),
        seeds=_sequence(experiment_spec, "seeds", (42,)),
        num_samples=num_samples,
        conditions=_sequence(experiment_spec, "conditions"),
        condition_query_budgets=condition_query_budgets,
        feedback_sources=_sequence(
            experiment_spec, "feedback_sources", ("genuine",)
        ),
        feedback_source_query_budgets=feedback_source_query_budgets,
        workflows=_sequence(experiment_spec, "workflows"),
        ablations=_sequence(experiment_spec, "ablations"),
        pair_families=_sequence(experiment_spec, "pair_families", (None,)),
        template_ids=_sequence(experiment_spec, "template_ids", (None,)),
        decoding_seeds=(
            _sequence(experiment_spec, "decoding_seeds", (None,))
            if decoding_seeds_override is None
            else tuple(decoding_seeds_override)
        ),
        dataset_split=None if dataset_split is None else str(dataset_split),
        dataset_revision=None if dataset_revision is None else str(dataset_revision),
        model=model,
        cells_per_task=_positive_integer(
            "cells_per_task", experiment_spec.get("cells_per_task", 1)
        ),
        episodes_per_shard=episodes_per_shard,
        e1_paired_instances=e1_paired_instances,
        e1_public_instances_per_shard=e1_public_instances_per_shard,
        pilot_id=pilot_id,
        sample_size_freeze=None,
    )


def _load_sample_size_freeze(
    path: Path | str | None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    freeze_path = Path(path)
    try:
        value = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GridError(f"cannot load sample-size freeze {freeze_path}: {error}") from error
    if not isinstance(value, dict):
        raise GridError("sample-size freeze must be one JSON object")
    return value


def _parse_named_budget_mapping(
    values: Sequence[str], *, option_name: str, item_name: str
) -> dict[str, tuple[int, ...]] | None:
    if not values:
        return None
    result: dict[str, tuple[int, ...]] = {}
    for value in values:
        condition, separator, raw_budgets = value.partition(":")
        if not separator or not condition or not raw_budgets:
            raise GridError(
                f"{option_name} values must have the form {item_name}:q1,q2"
            )
        try:
            budgets = tuple(int(item) for item in raw_budgets.split(","))
        except ValueError as error:
            raise GridError(f"invalid {item_name} budget mapping: {value!r}") from error
        if condition in result:
            raise GridError(f"duplicate {item_name} budget mapping: {condition!r}")
        result[condition] = budgets
    return result


def _parse_condition_budget(values: Sequence[str]) -> dict[str, tuple[int, ...]] | None:
    return _parse_named_budget_mapping(
        values,
        option_name="--condition-query-budgets",
        item_name="condition",
    )


def _parse_feedback_source_budget(
    values: Sequence[str],
) -> dict[str, tuple[int, ...]] | None:
    return _parse_named_budget_mapping(
        values,
        option_name="--feedback-source-query-budgets",
        item_name="feedback-source",
    )


def _model_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_cache_dir": args.model_cache_dir,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
    }


def grid_from_args(args: argparse.Namespace) -> ExperimentGrid:
    sample_size_freeze = _load_sample_size_freeze(args.sample_size_freeze)
    if args.preset:
        return grid_from_pilot(
            args.preset,
            experiment=args.experiment,
            model_overrides=_model_overrides(args),
            decoding_seeds_override=args.decoding_seeds,
            dataset_split_override=args.dataset_split,
            dataset_revision_override=args.dataset_revision,
            sample_size_freeze=sample_size_freeze,
        )
    required_lists = {
        "--tier": args.tiers,
        "--world-suite": args.world_suites,
        "--runtime": args.runtimes,
        "--attacker": args.attackers,
        "--query-budget": args.query_budgets,
        "--seed": args.seeds,
    }
    missing = [name for name, value in required_lists.items() if not value]
    if missing:
        raise GridError("non-preset grids require " + ", ".join(missing))
    if args.num_samples is None and sample_size_freeze is None:
        raise GridError("non-preset grids require --num-samples")
    return build_grid(
        experiment=args.experiment,
        tiers=args.tiers,
        world_suites=args.world_suites,
        runtimes=args.runtimes,
        attackers=args.attackers,
        query_budgets=args.query_budgets,
        seeds=args.seeds,
        num_samples=(DEFAULT_NUM_SAMPLES if args.num_samples is None else args.num_samples),
        conditions=args.conditions or (),
        condition_query_budgets=_parse_condition_budget(args.condition_query_budgets),
        feedback_sources=args.feedback_sources or ("genuine",),
        feedback_source_query_budgets=_parse_feedback_source_budget(
            args.feedback_source_query_budgets
        ),
        workflows=args.workflows or (),
        ablations=args.ablations or (),
        pair_families=args.pair_families or (None,),
        template_ids=args.template_ids or (None,),
        decoding_seeds=args.decoding_seeds or (None,),
        dataset_split=args.dataset_split,
        dataset_revision=args.dataset_revision,
        model=_model_overrides(args),
        cells_per_task=args.cells_per_task,
        episodes_per_shard=args.episodes_per_shard,
        pilot_id=None,
        sample_size_freeze=sample_size_freeze,
    )


def _shell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (dict, list)):
        return canonical_json(_jsonable(value))
    return str(value)


def print_grid(grid: ExperimentGrid) -> None:
    metadata = grid.metadata_record()
    print(f"experiment={grid.experiment}")
    print(f"pilot_id={grid.pilot_id or 'custom'}")
    print(f"total_tasks={grid.total_tasks}")
    print(f"total_configurations={grid.total_configurations}")
    print(f"valid_array_range=0-{grid.total_tasks - 1}")
    print("ordering=" + ">".join(grid.factor_order) + " (rightmost factor varies fastest)")
    print(f"cells_per_task={grid.cells_per_task}")
    print(f"grid_hash={metadata['grid_hash']}")
    for task in grid.tasks:
        for offset, cell in enumerate(task.cells):
            factors = " ".join(
                f"{key}={shlex.quote(_shell_value(value))}"
                for key, value in cell.configuration.items()
                if value is not None
            )
            print(
                f"task_id={task.task_id} batch_offset={offset} "
                f"batch_hash={task.batch_hash} cell_index={cell.cell_index} "
                f"grid_hash={grid.grid_hash} shard_id={cell.shard_id} "
                f"configuration_hash={cell.configuration_hash} "
                f"{factors}"
            )


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(_jsonable(record)))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_manifest(grid: ExperimentGrid, path: Path) -> None:
    _atomic_write_jsonl(path, grid.manifest_records())


def load_grid_manifest(path: Path | str) -> dict[str, Any]:
    """Load and independently validate an exact expected-grid manifest.

    Validation covers canonical scientific configuration hashes, shard IDs,
    flattened member uniqueness, task/batch ordering, declared counts, and the
    deterministic overall grid hash.  The returned mapping has ``metadata``
    and ``members`` keys and is safe for the aggregator to consume directly.
    """

    manifest_path = Path(path)
    try:
        raw_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GridError(f"cannot read grid manifest {manifest_path}: {error}") from error
    if not raw_lines:
        raise GridError(f"grid manifest is empty: {manifest_path}")
    if any(not line.strip() for line in raw_lines):
        raise GridError(f"grid manifest contains blank lines: {manifest_path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise GridError(
                f"invalid grid manifest JSON at {manifest_path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise GridError(
                f"grid manifest record {line_number} is not an object: {manifest_path}"
            )
        records.append(record)

    metadata = records[0]
    if metadata.get("record_type") != "grid_metadata":
        raise GridError("the first grid manifest record must be grid_metadata")
    if metadata.get("schema_version") != GRID_SCHEMA_VERSION:
        raise GridError(
            f"unsupported grid manifest schema {metadata.get('schema_version')!r}"
        )
    experiment = metadata.get("experiment_id")
    if experiment not in EXPERIMENTS:
        raise GridError(f"invalid grid experiment_id: {experiment!r}")
    factor_order = metadata.get("factor_order")
    if (
        not isinstance(factor_order, list)
        or not factor_order
        or not all(isinstance(item, str) and item for item in factor_order)
        or len(set(factor_order)) != len(factor_order)
    ):
        raise GridError("grid factor_order must contain unique non-empty strings")
    cells_per_task = _positive_integer("cells_per_task", metadata.get("cells_per_task"))
    total_tasks = _positive_integer("total_tasks", metadata.get("total_tasks"))
    total_configurations = _positive_integer(
        "total_configurations", metadata.get("total_configurations")
    )
    if metadata.get("valid_array_range") != f"0-{total_tasks - 1}":
        raise GridError("grid valid_array_range does not match total_tasks")
    recorded_grid_hash = metadata.get("grid_hash")
    if (
        not isinstance(recorded_grid_hash, str)
        or len(recorded_grid_hash) != 64
        or any(character not in "0123456789abcdef" for character in recorded_grid_hash)
    ):
        raise GridError("grid metadata has no valid grid_hash")

    members = records[1:]
    if len(members) != total_configurations:
        raise GridError(
            f"grid declares {total_configurations} configurations but contains {len(members)}"
        )
    expected_task_count = (total_configurations + cells_per_task - 1) // cells_per_task
    if total_tasks != expected_task_count:
        raise GridError("grid total_tasks does not match total_configurations/cells_per_task")
    task_members: list[list[tuple[str, str]]] = [[] for _ in range(total_tasks)]
    identities: set[tuple[str, str]] = set()
    for expected_cell_index, member in enumerate(members):
        if member.get("record_type") != "grid_member":
            raise GridError(f"grid record {expected_cell_index + 2} is not grid_member")
        if member.get("schema_version") != GRID_SCHEMA_VERSION:
            raise GridError("grid member has an incompatible schema version")
        if member.get("cell_index") != expected_cell_index:
            raise GridError("grid member cell_index values are not contiguous and ordered")
        task_id = _nonnegative_integer("task_id", member.get("task_id"))
        if task_id >= total_tasks:
            raise GridError(f"grid member task_id {task_id} is out of range")
        if task_id != expected_cell_index // cells_per_task:
            raise GridError("grid member task_id values do not follow the declared batching")
        expected_batch_offset = len(task_members[task_id])
        if member.get("batch_offset") != expected_batch_offset:
            raise GridError(f"task {task_id} batch_offset values are not contiguous and ordered")
        configuration = member.get("configuration")
        if not isinstance(configuration, dict):
            raise GridError("grid member configuration must be an object")
        if configuration.get("experiment") != experiment:
            raise GridError("grid member experiment does not match grid metadata")
        if scientific_configuration(configuration) != configuration:
            raise GridError("grid member configuration is not canonical")
        configuration_hash = member.get("configuration_hash")
        if configuration_hash != stable_hash(configuration):
            raise GridError("grid member configuration_hash does not match configuration")
        shard_index = _nonnegative_integer("shard_index", member.get("shard_index"))
        shard_count = _positive_integer("shard_count", member.get("shard_count"))
        if shard_index >= shard_count:
            raise GridError("grid member shard_index is outside shard_count")
        sample_start = _nonnegative_integer("sample_start", member.get("sample_start"))
        if configuration.get("sample_start") != sample_start:
            raise GridError("grid member sample_start does not match configuration")
        shard_id = member.get("shard_id")
        expected_shard_id = _shard_id(
            configuration_hash=str(configuration_hash),
            shard_index=shard_index,
            shard_count=shard_count,
            sample_start=sample_start,
        )
        if shard_id != expected_shard_id:
            raise GridError("grid member shard_id does not match its member identity")
        identity = (str(configuration_hash), str(shard_id))
        if identity in identities:
            raise GridError(f"duplicate grid member identity: {identity}")
        identities.add(identity)
        task_members[task_id].append(identity)

    # A held-out manifest is valid only if every logical treatment cell covers
    # the entire preregistered interval exactly once. Individual shard configs
    # merely prove they stay within the bound; this cross-member check prevents
    # a partial frozen N from being presented as a complete evaluation grid.
    frozen_groups: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        configuration = member["configuration"]
        if configuration.get("dataset_split") != "test":
            continue
        reduced = {
            key: value
            for key, value in configuration.items()
            if key not in {"num_samples", "sample_start"}
        }
        frozen_groups.setdefault(stable_hash(reduced), []).append(member)
    for group in frozen_groups.values():
        configuration = group[0]["configuration"]
        frozen_count = _positive_integer(
            "frozen_public_instances",
            configuration.get("frozen_public_instances"),
        )
        rows_per_instance = (
            4
            if experiment == "e2"
            or (
                experiment == "e1"
                and configuration.get("feedback_source") == "shuffled"
            )
            else 2
        )
        expected_end = frozen_count * rows_per_instance
        ordered = sorted(
            (
                int(item["configuration"]["sample_start"]),
                int(item["configuration"]["num_samples"]),
                int(item["shard_index"]),
                int(item["shard_count"]),
            )
            for item in group
        )
        if {item[3] for item in ordered} != {len(ordered)} or {
            item[2] for item in ordered
        } != set(range(len(ordered))):
            raise GridError("held-out logical cell has incomplete shard identities")
        cursor = 0
        for start, count, _, _ in ordered:
            if start != cursor:
                raise GridError(
                    "held-out logical cell has a gap or overlap before the frozen N"
                )
            cursor += count
        if cursor != expected_end:
            raise GridError(
                "held-out logical cell does not cover exactly the frozen public-instance count"
            )

    if any(not task for task in task_members):
        raise GridError("grid task IDs are not contiguous or include an empty task")
    if any(len(task) > cells_per_task for task in task_members):
        raise GridError("a grid task exceeds cells_per_task")
    if any(len(task) != cells_per_task for task in task_members[:-1]):
        raise GridError("only the final grid task may be smaller than cells_per_task")
    calculated_grid_hash = _calculate_grid_hash(
        experiment=str(experiment),
        pilot_id=metadata.get("pilot_id"),
        factor_order=tuple(factor_order),
        cells_per_task=cells_per_task,
        task_members=tuple(tuple(task) for task in task_members),
    )
    if recorded_grid_hash != calculated_grid_hash:
        raise GridError("grid_hash does not match the ordered grid manifest")
    return {"metadata": dict(metadata), "members": [dict(member) for member in members]}


def _emit_selected_nul(grid: ExperimentGrid, task_id: int) -> None:
    output = sys.stdout.buffer
    task = grid.task(task_id)
    common = {
        "grid_hash": grid.grid_hash,
        "grid_task_id": task.task_id,
        "grid_batch_hash": task.batch_hash,
        "pilot_id": grid.pilot_id,
    }
    for offset, cell in enumerate(task.cells):
        values = {
            **common,
            "batch_offset": offset,
            "cell_index": cell.cell_index,
            "shard_index": cell.shard_index,
            "shard_count": cell.shard_count,
            "shard_id": cell.shard_id,
            "sample_start": cell.sample_start,
            "configuration_hash": cell.configuration_hash,
            "configuration_json": canonical_json(_jsonable(cell.configuration)),
            **cell.configuration,
        }
        for key, value in values.items():
            output.write(str(key).encode("utf-8"))
            output.write(b"\0")
            output.write(_shell_value(value).encode("utf-8"))
            output.write(b"\0")
        output.write(b"__CELL_END__\0")


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--preset")
    parser.add_argument("--tier", dest="tiers", action="append")
    parser.add_argument("--world-suite", dest="world_suites", action="append")
    parser.add_argument("--runtime", dest="runtimes", action="append")
    parser.add_argument("--attacker", dest="attackers", action="append")
    parser.add_argument("--query-budget", dest="query_budgets", type=int, action="append")
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--condition", dest="conditions", action="append")
    parser.add_argument(
        "--condition-query-budgets",
        action="append",
        default=[],
        metavar="CONDITION:Q1,Q2",
    )
    parser.add_argument(
        "--feedback-source", dest="feedback_sources", action="append"
    )
    parser.add_argument(
        "--feedback-source-query-budgets",
        action="append",
        default=[],
        metavar="SOURCE:Q1,Q2",
    )
    parser.add_argument("--workflow", dest="workflows", action="append")
    parser.add_argument("--ablation", dest="ablations", action="append")
    parser.add_argument("--pair-family", dest="pair_families", action="append")
    parser.add_argument("--template-id", dest="template_ids", action="append")
    parser.add_argument("--decoding-seed", dest="decoding_seeds", type=int, action="append")
    parser.add_argument("--dataset-split")
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--sample-size-freeze",
        type=Path,
        help="hash-bound freeze JSON required for dataset_split=test",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--dtype")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cells-per-task", type=int, default=1)
    parser.add_argument("--episodes-per-shard", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silenttwin.experiments.grid",
        description="Inspect and select deterministic hash-bound experiment grids.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("print", "print the complete ordered task-to-configuration mapping"),
        ("count", "print only the total array task count"),
        ("member-count", "print only the expected completed-run count"),
        ("hash", "print only the deterministic overall grid hash"),
        ("manifest", "write or print the exact expected grid JSONL"),
        ("select", "select one array task and emit its cells"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_grid_arguments(child)
        if command == "manifest":
            child.add_argument("--output", type=Path)
        if command == "select":
            child.add_argument("--task-id", required=True, type=int)
            child.add_argument("--format", choices=("jsonl", "env-nul"), default="jsonl")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        grid = grid_from_args(args)
        if args.command == "print":
            print_grid(grid)
        elif args.command == "count":
            print(grid.total_tasks)
        elif args.command == "member-count":
            print(grid.total_configurations)
        elif args.command == "hash":
            print(grid.grid_hash)
        elif args.command == "manifest":
            if args.output:
                write_manifest(grid, args.output)
                print(str(args.output))
            else:
                for record in grid.manifest_records():
                    print(canonical_json(record))
        elif args.command == "select":
            if args.format == "env-nul":
                _emit_selected_nul(grid, args.task_id)
            else:
                task = grid.task(args.task_id)
                for offset, cell in enumerate(task.cells):
                    print(canonical_json(cell.member_record(task_id=task.task_id, batch_offset=offset)))
        else:  # pragma: no cover - argparse enforces the command set.
            parser.error(f"unknown command {args.command!r}")
    except (GridError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRID_SCHEMA_VERSION",
    "PILOT_SCHEMA_VERSION",
    "ExperimentGrid",
    "GridCell",
    "GridError",
    "GridTask",
    "build_grid",
    "grid_from_pilot",
    "load_grid_manifest",
    "load_pilot",
    "pilot_path",
    "print_grid",
    "scientific_configuration",
    "write_manifest",
]
