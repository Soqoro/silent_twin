from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from silenttwin.agentdojo.config import AGENTDOJO_SUITES
from silenttwin.agentdojo.scenarios import scenario_id, structural_group_id
from silenttwin.agentdojo.splits import (
    SplitManifestError,
    assign_entity_ids,
    build_split_manifest,
    validate_split_manifest,
)
from silenttwin.schemas import stable_digest


def _catalog_fixture() -> dict[str, object]:
    rows = []
    for suite in AGENTDOJO_SUITES:
        for index in range(6):
            split = ("train", "development", "test")[index % 3]
            user_id = f"user_task_{index}"
            injection_id = f"injection_task_{split}"
            rows.append(
                {
                    "scenario_id": scenario_id(
                        suite=suite,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        injection_vector_id="vector_0",
                        package_version="0.1.35",
                        source_revision="a75aba7631d3ca5fb7ab938965c97ead2f9ff84b",
                        benchmark_version="v1.2.2",
                    ),
                    "suite": suite,
                    "user_task_id": user_id,
                    "injection_task_id": injection_id,
                    "injection_vector_id": "vector_0",
                    "user_prompt_hash": stable_digest([suite, user_id, "prompt"]),
                    "injection_goal_hash": stable_digest([suite, injection_id, "goal"]),
                    "tool_schema_hash": stable_digest([suite, "tools"]),
                    "initial_environment_hash": stable_digest([suite, user_id, "env"]),
                    "clean_initial_environment_hash": stable_digest(
                        [suite, user_id, "clean-env"]
                    ),
                    "released_attack_name": "direct",
                    "released_attack_target_pipeline": "silenttwin-local-tool-loop",
                    "released_attack_rendering_hash": stable_digest(
                        [suite, user_id, injection_id, "direct-rendering"]
                    ),
                    "released_attack_initial_environment_hash": stable_digest(
                        [suite, user_id, injection_id, "direct-env"]
                    ),
                    "structural_group_id": structural_group_id(
                        suite=suite, user_task_id=user_id
                    ),
                    "dataset_split": split,
                    "agentdojo_package_version": "0.1.35",
                    "agentdojo_source_revision": "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b",
                    "agentdojo_benchmark_version": "v1.2.2",
                }
            )
    return {"catalog_hash": "a" * 64, "scenarios": rows}


def test_entity_assignment_is_order_independent_and_three_way() -> None:
    identifiers = [f"user_task_{index}" for index in range(12)]
    forward = assign_entity_ids(identifiers, suite="workspace", entity_kind="user_task")
    reverse = assign_entity_ids(
        list(reversed(identifiers)), suite="workspace", entity_kind="user_task"
    )
    assert forward == reverse
    assert set(forward.values()) == {"train", "development", "test"}


def test_manifest_is_exhaustive_and_tamper_evident() -> None:
    catalog = _catalog_fixture()
    manifest = build_split_manifest(catalog)
    validate_split_manifest(manifest, catalog=catalog)
    all_groups = {
        group
        for split in manifest["splits"].values()
        for group in split["structural_group_ids"]
    }
    assert all_groups == {row["structural_group_id"] for row in catalog["scenarios"]}

    tampered = copy.deepcopy(manifest)
    tampered["splits"]["test"]["scenario_ids"].append("not-a-scenario")
    with pytest.raises(SplitManifestError, match="split_manifest_hash"):
        validate_split_manifest(tampered, catalog=catalog)


def test_committed_split_manifest_binds_committed_catalog() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads(
        (root / "configs/silenttwin/agentdojo/catalog-v1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "configs/silenttwin/agentdojo/splits-v1.json").read_text(encoding="utf-8")
    )
    validate_split_manifest(manifest, catalog=catalog)
