"""Deterministic, entity-disjoint AgentDojo structural split manifests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from silenttwin.schemas import stable_digest

from .config import AGENTDOJO_SUITES
from .scenarios import DATASET_SPLITS, validate_scenario_rows


SPLITS_SCHEMA_VERSION = "silenttwin.agentdojo.splits.v1"
SPLIT_POLICY_REVISION = "silenttwin-agentdojo-entity-split-v1"


class SplitManifestError(ValueError):
    """A structural split assignment or manifest is invalid."""


def _allocation_counts(size: int) -> tuple[int, int, int]:
    if size < 3:
        raise SplitManifestError("at least three entity IDs are required for three-way splitting")
    train = (size + 1) // 2
    remainder = size - train
    development = (remainder + 1) // 2
    test = remainder - development
    if min(train, development, test) < 1:
        raise SplitManifestError("split allocation produced an empty partition")
    return train, development, test


def assign_entity_ids(
    identifiers: Sequence[str],
    *,
    suite: str,
    entity_kind: str,
) -> dict[str, str]:
    """Assign IDs with a frozen, order-independent 50/25/25 policy."""

    if suite not in AGENTDOJO_SUITES:
        raise SplitManifestError(f"unknown suite {suite!r}")
    values = tuple(str(item) for item in identifiers)
    if len(values) != len(set(values)):
        raise SplitManifestError(f"duplicate {entity_kind} IDs in {suite}")
    ordered = sorted(
        values,
        key=lambda item: (
            stable_digest(
                {
                    "split_policy_revision": SPLIT_POLICY_REVISION,
                    "suite": suite,
                    "entity_kind": entity_kind,
                    "entity_id": item,
                }
            ),
            item,
        ),
    )
    train_n, development_n, _ = _allocation_counts(len(ordered))
    boundaries = (train_n, train_n + development_n)
    return {
        item: (
            "train"
            if index < boundaries[0]
            else "development"
            if index < boundaries[1]
            else "test"
        )
        for index, item in enumerate(ordered)
    }


def entity_assignments(
    *,
    suite: str,
    user_task_ids: Sequence[str],
    injection_task_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    return {
        "user_tasks": assign_entity_ids(
            user_task_ids, suite=suite, entity_kind="user_task"
        ),
        "injection_tasks": assign_entity_ids(
            injection_task_ids, suite=suite, entity_kind="injection_task"
        ),
    }


def combination_split(
    assignments: Mapping[str, Mapping[str, str]],
    *,
    user_task_id: str,
    injection_task_id: str,
) -> str | None:
    user_split = assignments["user_tasks"][user_task_id]
    injection_split = assignments["injection_tasks"][injection_task_id]
    return user_split if user_split == injection_split else None


def build_split_manifest(catalog: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = catalog.get("scenarios")
    if not isinstance(scenarios, list):
        raise SplitManifestError("catalog scenarios must be a list")
    validate_scenario_rows(scenarios)
    catalog_hash = catalog.get("catalog_hash")
    if not isinstance(catalog_hash, str) or len(catalog_hash) != 64:
        raise SplitManifestError("catalog must have a valid catalog_hash")

    split_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    user_entities: dict[str, dict[str, set[str]]] = {
        split: defaultdict(set) for split in DATASET_SPLITS
    }
    injection_entities: dict[str, dict[str, set[str]]] = {
        split: defaultdict(set) for split in DATASET_SPLITS
    }
    for row in scenarios:
        split = str(row["dataset_split"])
        split_rows[split].append(row)
        suite = str(row["suite"])
        user_entities[split][suite].add(str(row["user_task_id"]))
        injection_entities[split][suite].add(str(row["injection_task_id"]))

    entries: dict[str, Any] = {}
    for split in DATASET_SPLITS:
        rows = split_rows[split]
        entries[split] = {
            "scenario_ids": sorted(str(row["scenario_id"]) for row in rows),
            "structural_group_ids": sorted(
                {str(row["structural_group_id"]) for row in rows}
            ),
            "user_task_ids_by_suite": {
                suite: sorted(user_entities[split][suite]) for suite in AGENTDOJO_SUITES
            },
            "injection_task_ids_by_suite": {
                suite: sorted(injection_entities[split][suite])
                for suite in AGENTDOJO_SUITES
            },
            "scenario_count_by_suite": {
                suite: sum(1 for row in rows if row["suite"] == suite)
                for suite in AGENTDOJO_SUITES
            },
        }
    payload = {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "split_policy_revision": SPLIT_POLICY_REVISION,
        "catalog_hash": catalog_hash,
        "independent_unit": "user_task",
        "entity_isolation": ["user_task_id", "injection_task_id", "attack_template_family"],
        "assignment_policy": "sha256_ordered_50_25_25_then_same_split_pairs_only",
        "cross_split_combinations": "retained_in_exposure_census_but_excluded_from_scenarios",
        "splits": entries,
    }
    document = {**payload, "split_manifest_hash": stable_digest(payload)}
    validate_split_manifest(document, catalog=catalog)
    return document


def validate_split_manifest(
    manifest: Mapping[str, Any], *, catalog: Mapping[str, Any] | None = None
) -> None:
    if manifest.get("schema_version") != SPLITS_SCHEMA_VERSION:
        raise SplitManifestError("unsupported split manifest schema")
    payload = dict(manifest)
    recorded_hash = payload.pop("split_manifest_hash", None)
    if recorded_hash != stable_digest(payload):
        raise SplitManifestError("split_manifest_hash is invalid")
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(DATASET_SPLITS):
        raise SplitManifestError("manifest must contain train/development/test")
    seen_groups: set[str] = set()
    seen_scenarios: set[str] = set()
    seen_users: set[tuple[str, str]] = set()
    seen_injections: set[tuple[str, str]] = set()
    for split in DATASET_SPLITS:
        entry = raw_splits[split]
        if not isinstance(entry, Mapping):
            raise SplitManifestError(f"{split} entry must be an object")
        groups = {str(item) for item in entry.get("structural_group_ids", ())}
        scenarios = {str(item) for item in entry.get("scenario_ids", ())}
        if seen_groups & groups:
            raise SplitManifestError("structural groups cross split boundaries")
        if seen_scenarios & scenarios:
            raise SplitManifestError("scenario IDs cross split boundaries")
        seen_groups.update(groups)
        seen_scenarios.update(scenarios)
        for suite, identifiers in entry.get("user_task_ids_by_suite", {}).items():
            entities = {(str(suite), str(item)) for item in identifiers}
            if seen_users & entities:
                raise SplitManifestError("user tasks cross split boundaries")
            seen_users.update(entities)
        for suite, identifiers in entry.get("injection_task_ids_by_suite", {}).items():
            entities = {(str(suite), str(item)) for item in identifiers}
            if seen_injections & entities:
                raise SplitManifestError("injection tasks cross split boundaries")
            seen_injections.update(entities)
    if catalog is not None:
        if manifest.get("catalog_hash") != catalog.get("catalog_hash"):
            raise SplitManifestError("split manifest is bound to another catalog")
        rows = catalog.get("scenarios")
        if not isinstance(rows, list):
            raise SplitManifestError("catalog scenarios must be a list")
        expected_groups = {str(row["structural_group_id"]) for row in rows}
        expected_scenarios = {str(row["scenario_id"]) for row in rows}
        if seen_groups != expected_groups or seen_scenarios != expected_scenarios:
            raise SplitManifestError("split manifest is not an exhaustive scenario partition")


__all__ = [
    "SPLITS_SCHEMA_VERSION",
    "SPLIT_POLICY_REVISION",
    "SplitManifestError",
    "assign_entity_ids",
    "build_split_manifest",
    "combination_split",
    "entity_assignments",
    "validate_split_manifest",
]
