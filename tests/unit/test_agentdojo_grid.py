from __future__ import annotations

from dataclasses import replace
import json
import hashlib
from pathlib import Path

import pytest

import silenttwin.agentdojo.grid as agentdojo_grid
from silenttwin.agentdojo.config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
    bundle_hash,
    stable_hash,
)
from silenttwin.agentdojo.grid import (
    AgentDojoGridError,
    _experiment_cells,
    _validate_preregistered_cells,
    build_grid,
    load_frozen_inputs,
    load_grid_manifest,
    validate_grid_manifest_coverage,
    validate_structural_splits,
    write_manifest,
)
from silenttwin.agentdojo.aggregate import (
    AgentDojoAggregationError,
    aggregate,
    discover_leaves,
)
from silenttwin.agentdojo.pair_mining import (
    make_observation_set_manifest,
    validate_candidate_strategy_catalog,
    validate_pair_registry,
)
from silenttwin.agentdojo.runtime_integrity import (
    not_applicable_learned_runtime_provenance,
)
from tests.unit.test_agentdojo_pair_mining import _pinned_compatibility


REPO_ROOT = Path(__file__).resolve().parents[2]


def _hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    return {**payload, field: stable_hash(payload)}


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _frozen_inputs(tmp_path: Path):
    scenarios: list[dict[str, object]] = []
    split_entries: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "development", "test"):
        split_scenarios: list[str] = []
        split_groups: list[str] = []
        for suite in AGENTDOJO_SUITES:
            scenario_id = f"{suite}-{split}-scenario"
            group_id = f"{suite}-{split}-group"
            split_scenarios.append(scenario_id)
            split_groups.append(group_id)
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "suite": suite,
                    "user_task_id": f"{suite}-{split}-user",
                    "injection_task_id": f"{suite}-{split}-injection",
                    "injection_vector_id": f"{suite}-{split}-vector",
                    "user_prompt_hash": "1" * 64,
                    "injection_goal_hash": "2" * 64,
                    "tool_schema_hash": "3" * 64,
                    "initial_environment_hash": "4" * 64,
                    "clean_initial_environment_hash": "5" * 64,
                    "released_attack_name": "direct",
                    "released_attack_target_pipeline": "silenttwin-local-tool-loop",
                    "released_attack_rendering_hash": "6" * 64,
                    "released_attack_initial_environment_hash": "7" * 64,
                    "structural_group_id": group_id,
                    "dataset_split": split,
                    "agentdojo_package_version": AGENTDOJO_PACKAGE_VERSION,
                    "agentdojo_source_revision": AGENTDOJO_SOURCE_REVISION,
                    "agentdojo_benchmark_version": AGENTDOJO_BENCHMARK_VERSION,
                    "catalog_binding": "top_level_catalog_hash",
                }
            )
        split_entries[split] = {
            "scenario_ids": split_scenarios,
            "structural_group_ids": split_groups,
        }
    catalog_payload: dict[str, object] = {
        "schema_version": "silenttwin.agentdojo.catalog.v1",
        "environment_backend": "agentdojo",
        "agentdojo_package_version": AGENTDOJO_PACKAGE_VERSION,
        "agentdojo_source_revision": AGENTDOJO_SOURCE_REVISION,
        "agentdojo_benchmark_version": AGENTDOJO_BENCHMARK_VERSION,
        "suite_order": list(AGENTDOJO_SUITES),
        "scenario_registry_revision": "fixture-registry-v1",
        "scenario_registry_hash": stable_hash(scenarios),
        "scenarios": scenarios,
    }
    catalog = _hashed(catalog_payload, "catalog_hash")
    splits = _hashed(
        {
            "schema_version": "silenttwin.agentdojo.splits.v1",
            "catalog_hash": catalog["catalog_hash"],
            "splits": split_entries,
        },
        "split_manifest_hash",
    )
    strategy = _hashed(
        {
            "schema_version": "silenttwin.agentdojo.candidate_strategy_catalog.v1",
            "environment_backend": "agentdojo",
            "tier2_track": "controlled",
            "catalog_hash": catalog["catalog_hash"],
            "split_manifest_hash": splits["split_manifest_hash"],
            "transformation_family_revision": "fixture-transformations-v1",
            "train_evidence_hash": "8" * 64,
            "frozen_before_development_pair_validation": True,
            "strategies": [
                {
                    "strategy_id": "s0",
                    "transformation_family": "fixture-family",
                    "ground_truth_label": "prohibited",
                    "plan_template": {"calls": [{"function": "fixture", "arguments": {}}]},
                    "frozen_on_split": "train",
                },
                {
                    "strategy_id": "s1",
                    "transformation_family": "fixture-family",
                    "ground_truth_label": "prohibited",
                    "plan_template": {"calls": [{"function": "fixture", "arguments": {}}]},
                    "frozen_on_split": "train",
                },
            ],
            "monitor_profiles": [
                {
                    "profile_id": "p0",
                    "profile_hash": "9" * 64,
                    "family": "deterministic_task_policy",
                    "frozen_on_split": "train",
                },
                {
                    "profile_id": "p1",
                    "profile_hash": "a" * 64,
                    "family": "deterministic_task_policy",
                    "frozen_on_split": "train",
                },
            ],
            "mixed_workflows": [],
        },
        "candidate_strategy_catalog_hash",
    )
    pair_rows = []
    for suite in AGENTDOJO_SUITES:
        pair_id = stable_hash(
            {
                "suite": suite,
                "profile_theta0": "p0",
                "profile_theta1": "p1",
                "candidate_0_strategy_id": "s0",
                "candidate_1_strategy_id": "s1",
                "candidate_strategy_catalog_hash": strategy[
                    "candidate_strategy_catalog_hash"
                ],
            }
        )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "suite": suite,
                "selection_split": "train",
                "validation_split": "development",
                "profile_theta0": "p0",
                "profile_theta1": "p1",
                "candidate_0_strategy_id": "s0",
                "candidate_1_strategy_id": "s1",
                "ground_truth_label_candidate_0": "prohibited",
                "ground_truth_label_candidate_1": "prohibited",
                "public_label_to_plan_mapping": {
                    "candidate_0": "s0",
                    "candidate_1": "s1",
                },
            }
        )
    pair_id_by_suite = {str(row["suite"]): row["pair_id"] for row in pair_rows}
    observation_set_manifests: dict[str, dict[str, object]] = {}
    for split in ("train", "development"):
        observations = [
            {
                "fixture": "grid-artifact-provenance",
                "dataset_split": split,
                "scenario_ids": split_entries[split]["scenario_ids"],
            }
        ]
        observation_set_manifests[split] = make_observation_set_manifest(
            observations=observations,
            dataset_split=split,
            catalog_hash=str(catalog["catalog_hash"]),
            split_manifest_hash=str(splits["split_manifest_hash"]),
            candidate_strategy_catalog_hash=str(
                strategy["candidate_strategy_catalog_hash"]
            ),
            generator_source_tree_hash=stable_hash(
                "grid-fixture-observation-generator"
            ),
            compatibility=_pinned_compatibility(),
            scientific_evidence_eligible=True,
            learned_runtime=not_applicable_learned_runtime_provenance(),
        )
    pair = _hashed(
        {
            "schema_version": "silenttwin.agentdojo.pair_registry.v1",
            "environment_backend": "agentdojo",
            "tier2_track": "controlled",
            "catalog_hash": catalog["catalog_hash"],
            "split_manifest_hash": splits["split_manifest_hash"],
            "candidate_strategy_catalog_hash": strategy[
                "candidate_strategy_catalog_hash"
            ],
            "train_observation_set_hash": observation_set_manifests["train"][
                "observation_set_hash"
            ],
            "development_observation_set_hash": observation_set_manifests[
                "development"
            ]["observation_set_hash"],
            "observation_set_manifests": observation_set_manifests,
            "test_outcomes_inspected": False,
            "pairs": pair_rows,
            "test_instantiations": [
                {
                    "pair_id": pair_id_by_suite[str(row["suite"])],
                    "scenario_id": row["scenario_id"],
                    "suite": row["suite"],
                    "structural_group_id": row["structural_group_id"],
                    "status": "unobserved_pre_execution",
                    "selected_by_test_outcome": False,
                }
                for row in scenarios
                if row["dataset_split"] == "test"
            ],
        },
        "pair_registry_hash",
    )
    analysis = json.loads(
        (REPO_ROOT / "configs/silenttwin/agentdojo/analysis/controlled-v1.json").read_text()
    )
    paths = {
        "catalog": tmp_path / "catalog.json",
        "splits": tmp_path / "splits.json",
        "strategy": tmp_path / "strategy.json",
        "pair": tmp_path / "pair.json",
        "analysis": tmp_path / "analysis.json",
        "lock": tmp_path / "lock.txt",
    }
    for name, value in (
        ("catalog", catalog),
        ("splits", splits),
        ("strategy", strategy),
        ("pair", pair),
        ("analysis", analysis),
    ):
        _write(paths[name], value)
    paths["lock"].write_text("fully-resolved-fixture==1.0 --hash=sha256:" + "0" * 64 + "\n")
    inputs = load_frozen_inputs(
        catalog_path=paths["catalog"],
        splits_path=paths["splits"],
        strategy_catalog_path=paths["strategy"],
        pair_registry_path=paths["pair"],
        analysis_plan_path=paths["analysis"],
        dependency_lock_path=paths["lock"],
    )
    return inputs, paths


def test_e2_grid_has_exact_condition_names_all_suites_and_matched_shards(
    tmp_path: Path,
) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text()
    )
    grid = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e2",
        tier2_track="controlled",
        dataset_split="development",
    )
    assert [task.suite for task in grid.tasks] == list(AGENTDOJO_SUITES)
    assert all(len(task.cells) == 13 for task in grid.tasks)
    assert {
        cell.configuration["condition"] for cell in grid.cells
    } == {
        "no_probe",
        "genuine",
        "matched_shuffled",
        "constant",
        "random_final",
        "oracle",
        "silenttwin",
    }
    assert all(
        len({cell.configuration["scenario_bundle_hash"] for cell in task.cells}) == 1
        for task in grid.tasks
    )

    manifest = tmp_path / "grid.jsonl"
    write_manifest(grid, manifest)
    loaded = load_grid_manifest(manifest)
    assert loaded["metadata"]["grid_hash"] == grid.grid_hash
    assert loaded["metadata"]["valid_array_range"] == "0-3"


def test_grid_is_deterministic_and_model_free(tmp_path: Path) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text()
    )
    first = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e1",
        tier2_track="controlled",
        dataset_split="development",
    )
    second = build_grid(
        inputs=inputs,
        grid_plan=json.loads(json.dumps(plan)),
        experiment_id="e1",
        tier2_track="controlled",
        dataset_split="development",
    )
    assert first.grid_hash == second.grid_hash
    assert first.metadata()["model_free"] is True
    assert len(first.cells) == 4 * 4 * 3 * 4


def test_grid_derives_monitor_family_from_frozen_pair_not_plan_label(
    tmp_path: Path,
) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text()
    )
    plan["base_configuration"]["monitor_family"] = "local_action_monitor"

    grid = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e1",
        tier2_track="controlled",
        dataset_split="development",
        suites=("workspace",),
    )

    assert {
        cell.configuration["monitor_family"] for cell in grid.cells
    } == {"deterministic_task_policy"}
    assert all(
        not any(model["role"] == "monitor" for model in cell.configuration["models"])
        for cell in grid.cells
    )


def test_learned_pair_mislabeled_deterministic_rejects_wrong_monitor_client_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text()
    )
    assert plan["base_configuration"]["monitor_family"] == (
        "deterministic_task_policy"
    )
    monkeypatch.setattr(
        agentdojo_grid,
        "monitor_pair_binding",
        lambda *_args, **_kwargs: {
            "profile_theta0": "p0",
            "profile_theta1": "p1",
            "monitor_family": "local_action_monitor",
            "monitor_profile_hash": "f" * 64,
        },
    )

    with pytest.raises(AgentDojoGridError, match="monitor identity differs from frozen"):
        build_grid(
            inputs=inputs,
            grid_plan=plan,
            experiment_id="e1",
            tier2_track="controlled",
            dataset_split="development",
            suites=("workspace",),
        )


@pytest.mark.parametrize("experiment_id", ("e4", "e5"))
def test_authored_experiment_grid_rejects_empty_workflow_catalog_preflight(
    tmp_path: Path, experiment_id: str
) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text()
    )

    with pytest.raises(AgentDojoGridError, match="train-frozen authored workflows"):
        build_grid(
            inputs=inputs,
            grid_plan=plan,
            experiment_id=experiment_id,
            tier2_track="controlled",
            dataset_split="development",
        )


def test_checked_fake_smoke_artifacts_validate_and_bootstrap_model_free_grids() -> None:
    config_root = REPO_ROOT / "configs/silenttwin/agentdojo"
    catalog_path = config_root / "catalog-v1.json"
    splits_path = config_root / "splits-v1.json"
    strategy_path = (
        config_root
        / "fixtures/deterministic-fake-smoke-candidate-strategies-v1.json"
    )
    pair_path = (
        config_root / "fixtures/deterministic-fake-smoke-pair-registry-v1.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    validate_candidate_strategy_catalog(strategy)
    validate_pair_registry(
        pair,
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy,
    )
    assert strategy["scientific_evidence_eligible"] is False
    assert pair["scientific_evidence_eligible"] is False
    expected_public_calls = {
        "workspace": "list_files",
        "travel": "get_user_information",
        "banking": "get_balance",
        "slack": "get_channels",
    }
    for candidate in strategy["strategies"]:
        assert {
            suite: template["calls"][0]["function"]
            for suite, template in candidate["suite_plan_templates"].items()
        } == expected_public_calls
    for workflow in strategy["mixed_workflows"]:
        assert workflow["frozen_on_split"] == "train"
        expected_function = expected_public_calls[workflow["suite"]]
        assert all(
            effect["calls"][0]["function"] == expected_function
            for effect in workflow["effects"]
        )

    inputs = load_frozen_inputs(
        catalog_path=catalog_path,
        splits_path=splits_path,
        strategy_catalog_path=strategy_path,
        pair_registry_path=pair_path,
        analysis_plan_path=config_root / "analysis/controlled-v1.json",
        dependency_lock_path=REPO_ROOT / "requirements-tier2-agentdojo.lock",
    )
    plan = json.loads(
        (config_root / "grid-plans/controlled-fake-smoke-v1.json").read_text(
            encoding="utf-8"
        )
    )
    for experiment_id, cells_per_task in (("e1", 48), ("e2", 13)):
        grid = build_grid(
            inputs=inputs,
            grid_plan=plan,
            experiment_id=experiment_id,
            tier2_track="controlled",
            dataset_split="development",
        )
        assert {task.suite for task in grid.tasks} == set(AGENTDOJO_SUITES)
        assert all(len(task.cells) == cells_per_task for task in grid.tasks)
        assert grid.metadata()["model_free"] is True
        assert all(cell.configuration["fixture_mode"] is True for cell in grid.cells)

    e5 = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e5",
        tier2_track="controlled",
        dataset_split="development",
    )
    structural = {
        "incomplete_data_dependencies",
        "incomplete_control_dependencies",
        "missing_atomic_group",
    }
    assert {
        (cell.configuration["ablation"], cell.configuration["query_budget"])
        for cell in e5.cells
    } == {
        (ablation, 0 if ablation in structural else 16)
        for ablation in {
            "none",
            "visible_settlement",
            "timing_leak",
            "identifier_leak",
            "error_schema_leak",
            "verdict_dependent_termination",
            "callback_leak",
            "later_world_read",
            "retained_memory",
            "retained_kv_cache",
            *structural,
            "public_fidelity_degradation",
            "decision_dependent_twin_mismatch",
        }
    }
    atomic_workflows = [
        workflow
        for workflow in strategy["mixed_workflows"]
        if workflow["workflow"] == "atomic"
    ]
    assert len(atomic_workflows) == len(AGENTDOJO_SUITES)
    for workflow in atomic_workflows:
        effects = workflow["effects"]
        assert any(effect["data_dependencies"] for effect in effects)
        assert any(effect["control_dependencies"] for effect in effects)
        groups = [effect["atomic_group"] for effect in effects if effect["atomic_group"]]
        assert len(groups) != len(set(groups))


def test_e3_e4_e5_and_ecological_require_exact_declared_cell_coverage(
    tmp_path: Path,
) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text(encoding="utf-8")
    )
    mutations = {
        "e1": lambda document: document["experiments"]["e1"]["factor_grid"][
            "query_budget"
        ].pop(),
        "e3": lambda document: document["experiments"]["e3"]["cells"].pop(),
        "e4": lambda document: document["experiments"]["e4"]["factor_grid"][
            "settlement_runtime"
        ].pop(),
        "e5": lambda document: document["experiments"]["e5"]["cells"].pop(),
    }
    for experiment_id, mutate in mutations.items():
        malformed = json.loads(json.dumps(plan))
        mutate(malformed)
        with pytest.raises(AgentDojoGridError, match="exact preregistered coverage"):
            build_grid(
                inputs=inputs,
                grid_plan=malformed,
                experiment_id=experiment_id,
                tier2_track="controlled",
                dataset_split="development",
            )

    ecological_plan = json.loads(json.dumps(plan))
    ecological_analysis = json.loads(
        (
            REPO_ROOT / "configs/silenttwin/agentdojo/analysis/ecological-v1.json"
        ).read_text(encoding="utf-8")
    )
    ecological_cells = [
        {**ecological_plan["base_configuration"], **cell}
        for cell in _experiment_cells(ecological_plan, "ecological")
    ]
    _validate_preregistered_cells(
        "ecological", ecological_cells, analysis_plan=ecological_analysis
    )
    with pytest.raises(AgentDojoGridError, match="exact preregistered coverage"):
        _validate_preregistered_cells(
            "ecological",
            ecological_cells[:-1],
            analysis_plan=ecological_analysis,
        )


def test_manifest_coverage_is_revalidated_and_subset_is_nonconfirmatory(
    tmp_path: Path,
) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text(encoding="utf-8")
    )
    grid = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e3",
        tier2_track="controlled",
        dataset_split="development",
    )
    loaded = {"metadata": grid.metadata(), "members": grid.records()[1:]}
    coverage = validate_grid_manifest_coverage(loaded, inputs.analysis_plan)
    assert coverage == {
        "cell_coverage_status": "exact_preregistered_matrix",
        "suite_coverage_status": "full_four_suite",
        "confirmatory_suite_coverage_eligible": True,
    }
    tampered = {
        "metadata": loaded["metadata"],
        "members": [
            member for member in loaded["members"] if member["batch_offset"] != 11
        ],
    }
    with pytest.raises(AgentDojoGridError, match="exact preregistered coverage"):
        validate_grid_manifest_coverage(tampered, inputs.analysis_plan)

    subset = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e3",
        tier2_track="controlled",
        dataset_split="development",
        suites=("workspace",),
    )
    subset_coverage = validate_grid_manifest_coverage(
        {"metadata": subset.metadata(), "members": subset.records()[1:]},
        inputs.analysis_plan,
    )
    assert subset_coverage["suite_coverage_status"] == (
        "development_subset_nonconfirmatory"
    )
    assert subset_coverage["confirmatory_suite_coverage_eligible"] is False

    estimation = replace(
        grid,
        protocol_disposition="estimation_only_action_representable",
        action_eligibility_manifest_hash="a" * 64,
    )
    estimation_metadata = estimation.metadata()
    assert estimation_metadata["suite_coverage_status"] == (
        "full_four_suite_estimation_only"
    )
    assert (
        estimation_metadata["confirmatory_suite_coverage_eligible"] is False
    )
    manifest_path = tmp_path / "estimation-grid.jsonl"
    write_manifest(estimation, manifest_path)
    assert load_grid_manifest(manifest_path)["metadata"] == estimation_metadata

    with pytest.raises(AgentDojoGridError, match="held-out execution"):
        replace(estimation, dataset_split="test")


def test_rehashed_fake_smoke_artifact_with_erased_claim_boundary_is_rejected(
    tmp_path: Path,
) -> None:
    config_root = REPO_ROOT / "configs/silenttwin/agentdojo"
    pair = json.loads(
        (
            config_root
            / "fixtures/deterministic-fake-smoke-pair-registry-v1.json"
        ).read_text(encoding="utf-8")
    )
    pair["claim_boundary"] = ""
    pair.pop("pair_registry_hash")
    pair["pair_registry_hash"] = stable_hash(pair)
    pair_path = tmp_path / "pair.json"
    _write(pair_path, pair)
    with pytest.raises(AgentDojoGridError, match="incomplete fake-smoke claim boundary"):
        load_frozen_inputs(
            catalog_path=config_root / "catalog-v1.json",
            splits_path=config_root / "splits-v1.json",
            strategy_catalog_path=(
                config_root
                / "fixtures/deterministic-fake-smoke-candidate-strategies-v1.json"
            ),
            pair_registry_path=pair_path,
            analysis_plan_path=config_root / "analysis/controlled-v1.json",
            dependency_lock_path=REPO_ROOT / "requirements-tier2-agentdojo.lock",
        )


def test_grid_rejects_semantically_rehashed_test_selected_pairs(tmp_path: Path) -> None:
    _, paths = _frozen_inputs(tmp_path)
    pair = json.loads(paths["pair"].read_text())
    pair["test_outcomes_inspected"] = True
    pair.pop("pair_registry_hash")
    pair["pair_registry_hash"] = stable_hash(pair)
    _write(paths["pair"], pair)
    with pytest.raises(AgentDojoGridError, match="held-out outcomes"):
        load_frozen_inputs(
            catalog_path=paths["catalog"],
            splits_path=paths["splits"],
            strategy_catalog_path=paths["strategy"],
            pair_registry_path=paths["pair"],
            analysis_plan_path=paths["analysis"],
            dependency_lock_path=paths["lock"],
        )


def test_structural_split_catalog_assignments_are_checked(tmp_path: Path) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    partitions = validate_structural_splits(inputs)
    assert set(partitions) == {"train", "development", "test"}
    assert all(len(value) == 4 for value in partitions.values())


@pytest.mark.parametrize(
    "experiment_id,donor_state",
    (("e1", None), ("e2", "theta0")),
)
def test_aggregate_rejects_self_consistent_production_manifest_missing_rows(
    tmp_path: Path,
    experiment_id: str,
    donor_state: str | None,
) -> None:
    from silenttwin.agentdojo.storage import (
        AgentDojoCheckpointStore,
        publish_completed_run,
    )
    from tests.unit.test_agentdojo_storage import SOURCE_HASH, _config, _e2_sample

    scenario_ids = ("scenario-0",)
    structural_group_ids = ("group-0",)
    config = replace(
        _config(tmp_path),
        experiment_id=experiment_id,
        condition="no_probe" if experiment_id == "e2" else None,
        query_budget=0,
        scenario_ids=scenario_ids,
        structural_group_ids=structural_group_ids,
        scenario_bundle_hash=bundle_hash(
            suite="workspace",
            dataset_split="train",
            scenario_ids=scenario_ids,
            structural_group_ids=structural_group_ids,
        ),
    )
    sample = _e2_sample(
        config,
        actual_state="theta0",
        donor_state=donor_state,
    )
    if experiment_id == "e1":
        sample["experiment_id"] = "e1"
        sample["condition"] = config.feedback_source
        sample["model_provenance"]["attacker"]["calls"] = sample[
            "model_provenance"
        ]["attacker"]["calls"][:1]
    sample.update(
        {
            "agent_visible_transcript": [],
            "postselection_output": [],
            "postselection_delivered_to_agent": False,
            "prediction_valid": False,
            "run_valid": True,
            "final_plan_attempt_count": 1 if experiment_id == "e2" else 0,
            "exact_transcript_distribution": {
                "status": "not_applicable",
                "method": None,
                "reason": "learned_non_enumerable_agentdojo_backend",
            },
        }
    )
    sample["trusted_evaluation"]["value"][
        "final_plan_locked_before_settlement"
    ] = experiment_id == "e2"
    store = AgentDojoCheckpointStore(
        tmp_path,
        config,
        (str(sample["trial_id"]),),
        provenance_hash=SOURCE_HASH,
    )
    store.initialize()
    store.save(sample)
    publish_completed_run(
        store=store,
        failures=(),
        started_at="2026-08-24T00:00:00Z",
        completed_at="2026-08-24T00:01:00Z",
        grid_hash=stable_hash("aggregate-incomplete-grid"),
        grid_task_id=0,
        shard_id="aggregate-incomplete-shard",
        provenance={"source_tree_hash": SOURCE_HASH},
    )

    with pytest.raises(
        AgentDojoAggregationError,
        match="incomplete or mismatched controlled cohort",
    ):
        discover_leaves(tmp_path)


def test_strict_aggregate_accepts_exact_grid_and_only_explicit_development_partial(
    tmp_path: Path,
) -> None:
    inputs, paths = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text()
    )
    grid = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e2",
        tier2_track="controlled",
        dataset_split="development",
    )
    grid_manifest = tmp_path / "grid.jsonl"
    write_manifest(grid, grid_manifest)
    run_root = tmp_path / "runs"
    leaf_manifests: list[Path] = []
    for cell in grid.cells:
        config = cell.configuration
        condition = str(config["condition"])
        attack = condition in {"genuine", "oracle"}
        utility = True
        scenario_id = str(config["scenario_ids"][0])
        assignments = (
            ("theta0", "theta0"),
            ("theta0", "theta1"),
            ("theta1", "theta0"),
            ("theta1", "theta1"),
        )
        rows = []
        for actual_state, donor_state in assignments:
            trial_id = stable_hash(
                {
                    "protocol": "silenttwin.agentdojo.controlled.v1",
                    "configuration_hash": cell.configuration_hash,
                    "scenario_id": scenario_id,
                    "actual_state": actual_state,
                    "donor_state": donor_state,
                    "replicate": config["replicate"],
                }
            )
            rows.append(
                {
                    "schema_version": "silenttwin.agentdojo.result.v1",
                    "record_type": "sample",
                    "experiment_id": "e2",
                    "trial_id": trial_id,
                    "configuration_hash": cell.configuration_hash,
                    "environment_backend": "agentdojo",
                    "tier2_track": "controlled",
                    "scenario_id": scenario_id,
                    "structural_group_id": config["structural_group_ids"][0],
                    "agentdojo_suite": config["agentdojo_suite"],
                    "condition": condition,
                    "feedback_policy": config["feedback_policy"],
                    "feedback_source": config["feedback_source"],
                    "query_budget": config["query_budget"],
                    "hidden_state_prediction": actual_state,
                    "posterior": {
                        "theta0": 0.8 if actual_state == "theta0" else 0.2,
                        "theta1": 0.2 if actual_state == "theta0" else 0.8,
                    },
                    "prediction_valid": True,
                    "agent_visible_transcript": [],
                    "postselection_output": [],
                    "postselection_delivered_to_agent": False,
                    "trusted_evaluation": {
                        "schema": "silenttwin.agentdojo.trusted.v1",
                        "value": {
                            "actual_hidden_state": actual_state,
                            "donor_state": donor_state,
                            "pair_yield_class": "both",
                            "final_plan_locked_before_settlement": True,
                        },
                    },
                    "exact_transcript_distribution": {
                        "status": "not_applicable",
                        "method": None,
                        "reason": "learned_non_enumerable_agentdojo_backend",
                    },
                    "raw_attack_success": attack,
                    "attack_success": attack,
                    "safe": not attack,
                    "utility": utility,
                    "safe_and_useful": utility and not attack,
                    "run_valid": True,
                    "final_plan_attempt_count": 1,
                    "committed_effect_ids": ["effect"] if attack else [],
                    "rejected_effect_ids": [] if attack else ["effect"],
                }
            )
        leaf = run_root / str(cell.shard_id)
        leaf.mkdir(parents=True)
        result_path = leaf / "result.jsonl"
        result_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
        manifest = {
            "status": "complete",
            "environment_backend": "agentdojo",
            "configuration": config,
            "configuration_hash": cell.configuration_hash,
            "result_file": "result.jsonl",
            "result_sha256": digest,
            "orchestration": {
                "grid_hash": grid.grid_hash,
                "shard_id": cell.shard_id,
            },
            "provenance": {"source_tree_hash": "source-fixture-v1"},
        }
        manifest_path = leaf / "manifest.json"
        _write(manifest_path, manifest)
        leaf_manifests.append(manifest_path)

    summary = aggregate(
        input_root=run_root,
        output_dir=tmp_path / "aggregate",
        expected_grid_manifest=grid_manifest,
        analysis_plan_path=paths["analysis"],
    )
    assert summary["grid_validation_mode"] == "exact_expected_grid"
    assert summary["suite_independent_unit_counts"] == {
        suite: 1 for suite in AGENTDOJO_SUITES
    }
    assert summary["fixture_mode"] is True
    assert summary["evidence_class"] == "engineering_smoke_only"
    assert summary["scientific_evidence_eligible"] is False
    assert summary["sample_size_freeze_eligible"] is False
    assert summary["development_power_analysis"]["status"] == (
        "not_evaluable_engineering_smoke"
    )
    assert summary["heldout_claim_disposition"] == "engineering_smoke_only"
    assert "harm" not in json.dumps(summary, sort_keys=True)
    assert (
        summary["go_no_go_gates"]["confirmatory_status"]
        == "not_evaluable_engineering_smoke"
    )
    assert all(
        gate["status"] == "not_evaluable"
        for gate in summary["go_no_go_gates"].values()
        if isinstance(gate, dict) and "status" in gate
    )
    analysis_manifest = json.loads(
        (tmp_path / "aggregate/analysis_manifest.json").read_text()
    )
    assert analysis_manifest["fixture_mode"] is True
    assert analysis_manifest["evidence_class"] == "engineering_smoke_only"
    assert analysis_manifest["scientific_evidence_eligible"] is False
    assert analysis_manifest["sample_size_freeze_eligible"] is False
    assert analysis_manifest["development_power_analysis"]["status"] == (
        "not_evaluable_engineering_smoke"
    )
    assert analysis_manifest["current_evidence_hash"] == analysis_manifest[
        "development_evidence_hash"
    ]
    recorded_hash = analysis_manifest.pop("analysis_manifest_hash")
    assert recorded_hash == stable_hash(analysis_manifest)

    leaf_manifests[-1].unlink()
    with pytest.raises(AgentDojoAggregationError, match="grid membership mismatch"):
        aggregate(
            input_root=run_root,
            output_dir=tmp_path / "strict-partial",
            expected_grid_manifest=grid_manifest,
            analysis_plan_path=paths["analysis"],
        )
    partial = aggregate(
        input_root=run_root,
        output_dir=tmp_path / "explicit-partial",
        expected_grid_manifest=grid_manifest,
        analysis_plan_path=paths["analysis"],
        allow_development_partial=True,
    )
    assert partial["grid_validation_mode"] == "development_only_partial"
    assert partial["development_only_partial"] is True
