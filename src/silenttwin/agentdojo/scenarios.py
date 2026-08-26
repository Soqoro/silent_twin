"""Dependency-free AgentDojo scenario identities and registry validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from silenttwin.schemas import stable_digest

from .config import AGENTDOJO_SUITES


SCENARIO_REGISTRY_REVISION = "silenttwin-agentdojo-exposure-v1"
CATALOG_SCHEMA_VERSION = "silenttwin.agentdojo.catalog.v1"
DATASET_SPLITS = ("train", "development", "test")
EXCLUDED_CROSS_SPLIT = "excluded_cross_split"
REQUIRED_SCENARIO_FIELDS = (
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
    "released_attack_name",
    "released_attack_target_pipeline",
    "released_attack_rendering_hash",
    "released_attack_initial_environment_hash",
    "structural_group_id",
    "dataset_split",
    "agentdojo_package_version",
    "agentdojo_source_revision",
    "agentdojo_benchmark_version",
)
HASH_FIELDS = (
    "scenario_id",
    "user_prompt_hash",
    "injection_goal_hash",
    "tool_schema_hash",
    "initial_environment_hash",
    "clean_initial_environment_hash",
    "released_attack_rendering_hash",
    "released_attack_initial_environment_hash",
    "structural_group_id",
)


class ScenarioRegistryError(ValueError):
    """A generated or frozen scenario registry is structurally invalid."""


def scenario_id(
    *,
    suite: str,
    user_task_id: str,
    injection_task_id: str,
    injection_vector_id: str,
    package_version: str,
    source_revision: str,
    benchmark_version: str,
) -> str:
    return stable_digest(
        {
            "identity_revision": SCENARIO_REGISTRY_REVISION,
            "suite": suite,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "injection_vector_id": injection_vector_id,
            "agentdojo_package_version": package_version,
            "agentdojo_source_revision": source_revision,
            "agentdojo_benchmark_version": benchmark_version,
        }
    )


def structural_group_id(*, suite: str, user_task_id: str) -> str:
    """Return the preregistered independent-unit identity.

    The user workflow, rather than a seed or injected rendering, is the
    independent unit.  Injection tasks are independently assigned to splits;
    the registry includes a combination only when both entity assignments
    agree.
    """

    return stable_digest(
        {
            "identity_revision": SCENARIO_REGISTRY_REVISION,
            "independent_unit": "user_task",
            "suite": suite,
            "user_task_id": user_task_id,
        }
    )


def scenario_registry_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return stable_digest([dict(row) for row in rows])


def validate_scenario_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_splits: Iterable[str] = DATASET_SPLITS,
    require_minimum_per_suite: int = 6,
) -> dict[str, int]:
    allowed = set(allowed_splits)
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    user_splits: dict[tuple[str, str], str] = {}
    injection_splits: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    distinct_groups: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_SCENARIO_FIELDS if field not in row]
        if missing:
            raise ScenarioRegistryError(f"scenario row {index} lacks {missing!r}")
        suite = str(row["suite"])
        split = str(row["dataset_split"])
        if suite not in AGENTDOJO_SUITES:
            raise ScenarioRegistryError(f"scenario row {index} has unknown suite {suite!r}")
        if split not in allowed:
            raise ScenarioRegistryError(f"scenario row {index} has invalid split {split!r}")
        for field in HASH_FIELDS:
            value = row[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value.lower() != value
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ScenarioRegistryError(f"scenario row {index} has invalid {field}")
        if row["released_attack_name"] != "direct":
            raise ScenarioRegistryError(
                f"scenario row {index} does not bind the frozen released direct attack"
            )
        if row["released_attack_target_pipeline"] != "silenttwin-local-tool-loop":
            raise ScenarioRegistryError(
                f"scenario row {index} has an invalid released attack target pipeline"
            )
        identifier = str(row["scenario_id"])
        if identifier in seen_ids:
            raise ScenarioRegistryError(f"duplicate scenario_id {identifier}")
        seen_ids.add(identifier)
        group = str(row["structural_group_id"])
        prior = group_splits.setdefault(group, split)
        if prior != split:
            raise ScenarioRegistryError(f"structural group {group} crosses splits")
        user_key = (suite, str(row["user_task_id"]))
        prior = user_splits.setdefault(user_key, split)
        if prior != split:
            raise ScenarioRegistryError(f"user task {user_key!r} crosses splits")
        injection_key = (suite, str(row["injection_task_id"]))
        prior = injection_splits.setdefault(injection_key, split)
        if prior != split:
            raise ScenarioRegistryError(f"injection task {injection_key!r} crosses splits")
        counts[suite] += 1
        distinct_groups[suite].add(group)
    if set(counts) != set(AGENTDOJO_SUITES):
        raise ScenarioRegistryError("scenario registry must cover all four AgentDojo suites")
    shortfalls = {
        suite: max(0, require_minimum_per_suite - len(distinct_groups[suite]))
        for suite in AGENTDOJO_SUITES
    }
    if any(shortfalls.values()):
        raise ScenarioRegistryError(
            "scenario registry has structural shortfalls: "
            + ", ".join(f"{suite}={value}" for suite, value in shortfalls.items() if value)
        )
    return {suite: counts[suite] for suite in AGENTDOJO_SUITES}


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DATASET_SPLITS",
    "EXCLUDED_CROSS_SPLIT",
    "HASH_FIELDS",
    "REQUIRED_SCENARIO_FIELDS",
    "SCENARIO_REGISTRY_REVISION",
    "ScenarioRegistryError",
    "scenario_id",
    "scenario_registry_hash",
    "structural_group_id",
    "validate_scenario_rows",
]
