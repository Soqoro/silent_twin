"""Pinned AgentDojo exposure census and split-safe scenario registry."""

from __future__ import annotations

from collections import Counter, defaultdict
from importlib import import_module
import re
from typing import Any, Mapping, Sequence

from silenttwin.schemas import stable_digest

from .config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
    ECOLOGICAL_ATTACK_TARGET_PIPELINE,
)
from .scenarios import (
    CATALOG_SCHEMA_VERSION,
    DATASET_SPLITS,
    EXCLUDED_CROSS_SPLIT,
    SCENARIO_REGISTRY_REVISION,
    scenario_id,
    scenario_registry_hash,
    structural_group_id,
    validate_scenario_rows,
)
from .splits import SPLIT_POLICY_REVISION, combination_split, entity_assignments


EXPECTED_EXPOSURE_COUNTS = {
    "workspace": 840,
    "banking": 144,
    "slack": 140,
    "travel": 343,
}
EXPECTED_EXPOSURE_TOTAL = 1467


class CatalogError(ValueError):
    """The live release census or a frozen catalog is invalid."""


def _compat_module(value: Any | None = None) -> Any:
    return value if value is not None else import_module("silenttwin.agentdojo.compat")


def _identifier_key(identifier: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.*?)(\d+)", identifier)
    if match is None:
        return identifier, -1, identifier
    return match.group(1), int(match.group(2)), identifier


def _report_value(report: Any, name: str) -> Any:
    if isinstance(report, Mapping):
        return report[name]
    return getattr(report, name)


def _metadata_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    result = {
        "name": value.name,
        "benchmark_version": list(value.benchmark_version),
        "user_task_count": value.user_task_count,
        "injection_task_count": value.injection_task_count,
        "tool_count": value.tool_count,
        "injection_vector_count": value.injection_vector_count,
    }
    return result


def _fresh_environment_hash(
    adapter: Any,
    suite: Any,
    user_task: Any,
    injections: Mapping[str, str],
) -> str:
    first = adapter.environment_hash(
        adapter.load_environment(suite, user_task, dict(injections))
    )
    second = adapter.environment_hash(
        adapter.load_environment(suite, user_task, dict(injections))
    )
    if first != second:
        raise CatalogError(
            f"non-deterministic initial environment for {suite.name}/{user_task.ID}: "
            f"{first} != {second}"
        )
    return first


def _build_row(
    *,
    suite_name: str,
    user_task: Any,
    injection_task: Any,
    vector_id: str,
    tool_schema_hash: str,
    initial_environment_hash: str,
    released_attack_rendering_hash: str,
    released_attack_initial_environment_hash: str,
    dataset_split: str,
    package_version: str,
    source_revision: str,
    benchmark_version: str,
) -> dict[str, Any]:
    identifier = scenario_id(
        suite=suite_name,
        user_task_id=str(user_task.ID),
        injection_task_id=str(injection_task.ID),
        injection_vector_id=vector_id,
        package_version=package_version,
        source_revision=source_revision,
        benchmark_version=benchmark_version,
    )
    return {
        "scenario_id": identifier,
        "suite": suite_name,
        "user_task_id": str(user_task.ID),
        "injection_task_id": str(injection_task.ID),
        "injection_vector_id": vector_id,
        "user_prompt_hash": stable_digest(str(user_task.PROMPT)),
        "injection_goal_hash": stable_digest(str(injection_task.GOAL)),
        "tool_schema_hash": tool_schema_hash,
        "initial_environment_hash": initial_environment_hash,
        # ``initial_environment_hash`` above deliberately remains the
        # controlled-track raw-goal initialization.  A released attack can
        # render different text, so ecological initialization is separately
        # frozen and must never be inferred from that controlled hash.
        "released_attack_name": "direct",
        "released_attack_target_pipeline": ECOLOGICAL_ATTACK_TARGET_PIPELINE,
        "released_attack_rendering_hash": released_attack_rendering_hash,
        "released_attack_initial_environment_hash": (
            released_attack_initial_environment_hash
        ),
        "structural_group_id": structural_group_id(
            suite=suite_name, user_task_id=str(user_task.ID)
        ),
        "dataset_split": dataset_split,
        "agentdojo_package_version": package_version,
        "agentdojo_source_revision": source_revision,
        "agentdojo_benchmark_version": benchmark_version,
        # A row cannot literally contain the final catalog hash without a
        # circular digest.  The top-level catalog_hash transitively binds this
        # complete row and the exposure_census_hash binds the census alone.
        "catalog_binding": "top_level_catalog_hash",
    }


def build_catalog(
    *,
    deployment_source_revision: str = AGENTDOJO_SOURCE_REVISION,
    benchmark_version: str = AGENTDOJO_BENCHMARK_VERSION,
    validate_release_drift: bool = True,
    compat: Any | None = None,
) -> dict[str, Any]:
    """Introspect all released suites and return one hash-bound catalog.

    ``eligible_combinations`` is the complete released exposure census.
    ``scenarios`` is the subset whose independently assigned user and
    injection entities land in the same split.  Keeping the layers separate
    is necessary: the full exposure graph connects all task entities and
    therefore cannot also be a leakage-free three-way partition.
    """

    adapter = _compat_module(compat)
    report = adapter.assert_compatible(deployment_source_revision, benchmark_version)
    package_version = str(_report_value(report, "package_version"))
    source_revision = str(_report_value(report, "source_revision"))
    resolved_benchmark = str(_report_value(report, "benchmark_version"))
    if package_version != AGENTDOJO_PACKAGE_VERSION:
        raise CatalogError("compatibility report returned another package version")

    eligible_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    suite_records: dict[str, Any] = {}
    assignments_by_suite: dict[str, Any] = {}
    observed_counts: Counter[str] = Counter()

    for suite_name in AGENTDOJO_SUITES:
        suite = adapter.load_suite(
            suite_name,
            deployment_source_revision=deployment_source_revision,
            benchmark_version=benchmark_version,
        )
        user_ids = sorted((str(item) for item in suite.user_tasks), key=_identifier_key)
        injection_ids = sorted(
            (str(item) for item in suite.injection_tasks), key=_identifier_key
        )
        assignments = entity_assignments(
            suite=suite_name,
            user_task_ids=user_ids,
            injection_task_ids=injection_ids,
        )
        assignments_by_suite[suite_name] = assignments

        tool_schemas = adapter.canonical_tool_schemas(suite)
        tool_schema_hash = stable_digest(tool_schemas)
        if tool_schema_hash != stable_digest(adapter.canonical_tool_schemas(suite)):
            raise CatalogError(f"non-deterministic tool schema for suite {suite_name}")
        defaults = suite.get_injection_vector_defaults()
        exposed_vectors: set[str] = set()
        user_vector_edge_count = 0
        initial_hashes: dict[str, str] = {}
        candidate_vectors_by_user: dict[str, list[str]] = {}

        for user_id in user_ids:
            user_task = adapter.get_user_task(suite, user_id)
            clean_initial_hash = _fresh_environment_hash(
                adapter, suite, user_task, {}
            )
            initial_hashes[user_id] = clean_initial_hash
            candidates = sorted(
                {str(item) for item in adapter.get_injection_candidates(suite, user_task)}
            )
            if not candidates:
                raise CatalogError(
                    "released candidate discovery found no vector for "
                    f"{suite_name}/{user_id}"
                )
            if not set(candidates) <= set(defaults):
                raise CatalogError(
                    "candidate discovery returned an unknown vector for "
                    f"{suite_name}/{user_id}"
                )
            candidate_vectors_by_user[user_id] = candidates
            exposed_vectors.update(candidates)
            user_vector_edge_count += len(candidates)
            for injection_id in injection_ids:
                injection_task = adapter.get_injection_task(suite, injection_id)
                released_injections = adapter.generate_attack_injections(
                    suite,
                    user_task,
                    injection_task,
                    attack_name="direct",
                    target_pipeline_name=ECOLOGICAL_ATTACK_TARGET_PIPELINE,
                )
                assigned = combination_split(
                    assignments,
                    user_task_id=user_id,
                    injection_task_id=injection_id,
                )
                split = assigned if assigned is not None else EXCLUDED_CROSS_SPLIT
                for vector_id in candidates:
                    # The primary controlled scenario starts with this exact
                    # benchmark-local indirect injection already materialized.
                    # Hashing the clean task here would bind a different state
                    # from the one the controlled backend actually executes.
                    initial_hash = _fresh_environment_hash(
                        adapter,
                        suite,
                        user_task,
                        {vector_id: str(injection_task.GOAL)},
                    )
                    if vector_id not in released_injections:
                        raise CatalogError(
                            "released direct attack omitted an exposed vector for "
                            f"{suite_name}/{user_id}/{injection_id}/{vector_id}"
                        )
                    rendered_injection = {
                        vector_id: str(released_injections[vector_id])
                    }
                    released_initial_hash = _fresh_environment_hash(
                        adapter,
                        suite,
                        user_task,
                        rendered_injection,
                    )
                    row = _build_row(
                        suite_name=suite_name,
                        user_task=user_task,
                        injection_task=injection_task,
                        vector_id=vector_id,
                        tool_schema_hash=tool_schema_hash,
                        initial_environment_hash=initial_hash,
                        released_attack_rendering_hash=stable_digest(
                            rendered_injection
                        ),
                        released_attack_initial_environment_hash=(
                            released_initial_hash
                        ),
                        dataset_split=split,
                        package_version=package_version,
                        source_revision=source_revision,
                        benchmark_version=resolved_benchmark,
                    )
                    row["clean_initial_environment_hash"] = clean_initial_hash
                    eligible_rows.append(row)
                    observed_counts[suite_name] += 1
                    if assigned is not None:
                        scenario_rows.append(row)

        metadata_record = _metadata_dict(adapter.suite_metadata(suite))
        suite_records[suite_name] = {
            **metadata_record,
            "tool_schema_hash": tool_schema_hash,
            "injection_vector_defaults_hash": stable_digest(defaults),
            "exposed_user_vector_edge_count": user_vector_edge_count,
            "eligible_combination_count": observed_counts[suite_name],
            "candidate_vectors_by_user_task": candidate_vectors_by_user,
            "clean_initial_environment_hash_by_user_task": initial_hashes,
            "unexposed_default_vectors": sorted(set(defaults) - exposed_vectors),
        }

    # Sorting is explicit even though suite/task iteration is already frozen.
    row_key = lambda row: (
        AGENTDOJO_SUITES.index(str(row["suite"])),
        _identifier_key(str(row["user_task_id"])),
        _identifier_key(str(row["injection_task_id"])),
        str(row["injection_vector_id"]),
    )
    eligible_rows.sort(key=row_key)
    scenario_rows.sort(key=row_key)
    validate_scenario_rows(scenario_rows)
    if validate_release_drift:
        observed = {suite: observed_counts[suite] for suite in AGENTDOJO_SUITES}
        if observed != EXPECTED_EXPOSURE_COUNTS:
            raise CatalogError(
                f"pinned release exposure counts drifted: expected {EXPECTED_EXPOSURE_COUNTS!r}, "
                f"got {observed!r}"
            )
        if len(eligible_rows) != EXPECTED_EXPOSURE_TOTAL:
            raise CatalogError(
                f"pinned release exposure total drifted: expected {EXPECTED_EXPOSURE_TOTAL}, "
                f"got {len(eligible_rows)}"
            )

    structural_counts = {
        suite: len(
            {
                row["structural_group_id"]
                for row in scenario_rows
                if row["suite"] == suite
            }
        )
        for suite in AGENTDOJO_SUITES
    }
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "environment_backend": "agentdojo",
        "agentdojo_package_version": package_version,
        "agentdojo_source_revision": source_revision,
        "agentdojo_source_revision_verification": _report_value(
            report, "source_revision_verification"
        ),
        "agentdojo_benchmark_version": resolved_benchmark,
        "agentdojo_wheel_sha256": _report_value(report, "wheel_sha256"),
        "suite_order": list(AGENTDOJO_SUITES),
        "resolved_suites": suite_records,
        "split_policy_revision": SPLIT_POLICY_REVISION,
        "entity_split_assignments": assignments_by_suite,
        "scenario_registry_revision": SCENARIO_REGISTRY_REVISION,
        "scenario_registry_hash": scenario_registry_hash(scenario_rows),
        "exposure_census_hash": stable_digest(eligible_rows),
        "eligible_combination_count": len(eligible_rows),
        "eligible_combination_count_by_suite": {
            suite: observed_counts[suite] for suite in AGENTDOJO_SUITES
        },
        "scenario_count": len(scenario_rows),
        "scenario_count_by_suite": {
            suite: sum(1 for row in scenario_rows if row["suite"] == suite)
            for suite in AGENTDOJO_SUITES
        },
        "structural_group_count_by_suite": structural_counts,
        "minimum_structural_scenarios_per_suite": 6,
        "preferred_structural_scenarios_per_suite": 8,
        "structural_shortfall_by_suite": {
            suite: max(0, 6 - structural_counts[suite]) for suite in AGENTDOJO_SUITES
        },
        "cross_split_policy": "full_census_retained; only same-entity-split rows enter scenarios",
        "eligible_combinations": eligible_rows,
        "scenarios": scenario_rows,
    }
    document = {**payload, "catalog_hash": stable_digest(payload)}
    validate_catalog(document, validate_release_drift=validate_release_drift)
    return document


def validate_catalog(
    document: Mapping[str, Any], *, validate_release_drift: bool = True
) -> None:
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise CatalogError("unsupported AgentDojo catalog schema")
    payload = dict(document)
    recorded_hash = payload.pop("catalog_hash", None)
    if recorded_hash != stable_digest(payload):
        raise CatalogError("catalog_hash is invalid")
    if document.get("agentdojo_package_version") != AGENTDOJO_PACKAGE_VERSION:
        raise CatalogError("catalog package version is not pinned")
    if document.get("agentdojo_source_revision") != AGENTDOJO_SOURCE_REVISION:
        raise CatalogError("catalog source revision is not pinned")
    if document.get("agentdojo_benchmark_version") != AGENTDOJO_BENCHMARK_VERSION:
        raise CatalogError("catalog benchmark version is not pinned")
    eligible = document.get("eligible_combinations")
    scenarios = document.get("scenarios")
    if not isinstance(eligible, list) or not isinstance(scenarios, list):
        raise CatalogError("catalog census and scenarios must be lists")
    if document.get("exposure_census_hash") != stable_digest(eligible):
        raise CatalogError("exposure_census_hash is invalid")
    for index, row in enumerate(eligible):
        if not isinstance(row, Mapping):
            raise CatalogError(f"eligible scenario row {index} is not an object")
        if row.get("released_attack_name") != "direct":
            raise CatalogError(
                f"eligible scenario row {index} lacks the frozen released direct attack"
            )
        if (
            row.get("released_attack_target_pipeline")
            != ECOLOGICAL_ATTACK_TARGET_PIPELINE
        ):
            raise CatalogError(
                f"eligible scenario row {index} has another attack target pipeline"
            )
        for field in (
            "clean_initial_environment_hash",
            "initial_environment_hash",
            "released_attack_rendering_hash",
            "released_attack_initial_environment_hash",
        ):
            value = row.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value.lower() != value
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise CatalogError(
                    f"eligible scenario row {index} has an invalid {field}"
                )
    if document.get("scenario_registry_hash") != scenario_registry_hash(scenarios):
        raise CatalogError("scenario_registry_hash is invalid")
    validate_scenario_rows(scenarios)
    eligible_by_id = {row["scenario_id"]: row for row in eligible}
    scenario_ids = {row["scenario_id"] for row in scenarios}
    eligible_ids = set(eligible_by_id)
    if not scenario_ids <= eligible_ids:
        raise CatalogError("scenario registry is not a subset of the exposure census")
    if len(eligible_ids) != len(eligible):
        raise CatalogError("exposure census repeats scenario identities")
    if any(eligible_by_id[row["scenario_id"]] != row for row in scenarios):
        raise CatalogError("scenario registry row differs from its exposure-census row")
    if any(row.get("dataset_split") not in DATASET_SPLITS for row in scenarios):
        raise CatalogError("split-safe scenario registry contains excluded rows")
    eligible_counts = Counter(str(row.get("suite")) for row in eligible)
    scenario_counts = Counter(str(row.get("suite")) for row in scenarios)
    if document.get("eligible_combination_count") != len(eligible):
        raise CatalogError("eligible_combination_count does not match the census")
    if document.get("scenario_count") != len(scenarios):
        raise CatalogError("scenario_count does not match the registry")
    if document.get("eligible_combination_count_by_suite") != {
        suite: eligible_counts[suite] for suite in AGENTDOJO_SUITES
    }:
        raise CatalogError("eligible suite counts do not match the census")
    if document.get("scenario_count_by_suite") != {
        suite: scenario_counts[suite] for suite in AGENTDOJO_SUITES
    }:
        raise CatalogError("scenario suite counts do not match the registry")
    structural_counts = {
        suite: len(
            {
                row["structural_group_id"]
                for row in scenarios
                if row.get("suite") == suite
            }
        )
        for suite in AGENTDOJO_SUITES
    }
    if document.get("structural_group_count_by_suite") != structural_counts:
        raise CatalogError("structural suite counts do not match the registry")
    if document.get("suite_order") != list(AGENTDOJO_SUITES):
        raise CatalogError("catalog suite order is not the pinned explicit order")
    resolved = document.get("resolved_suites")
    if not isinstance(resolved, Mapping) or set(resolved) != set(AGENTDOJO_SUITES):
        raise CatalogError("catalog resolved suites are incomplete")
    if validate_release_drift:
        observed = {suite: eligible_counts[suite] for suite in AGENTDOJO_SUITES}
        if observed != EXPECTED_EXPOSURE_COUNTS or len(eligible) != EXPECTED_EXPOSURE_TOTAL:
            raise CatalogError("frozen catalog does not match pinned release exposure counts")


def catalog_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    validate_catalog(document)
    return {
        "catalog_hash": document["catalog_hash"],
        "scenario_registry_hash": document["scenario_registry_hash"],
        "eligible_combination_count": document["eligible_combination_count"],
        "eligible_combination_count_by_suite": document[
            "eligible_combination_count_by_suite"
        ],
        "scenario_count": document["scenario_count"],
        "scenario_count_by_suite": document["scenario_count_by_suite"],
        "structural_group_count_by_suite": document[
            "structural_group_count_by_suite"
        ],
    }


__all__ = [
    "CatalogError",
    "EXPECTED_EXPOSURE_COUNTS",
    "EXPECTED_EXPOSURE_TOTAL",
    "build_catalog",
    "catalog_summary",
    "validate_catalog",
]
