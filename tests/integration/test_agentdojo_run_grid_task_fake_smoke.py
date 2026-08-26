from __future__ import annotations

from importlib import metadata
from importlib.util import find_spec
import json
from pathlib import Path
import sys

import pytest

from silenttwin.agentdojo.config import (
    AGENTDOJO_SUITES,
    AgentDojoExperimentConfig,
    bundle_hash,
    stable_hash,
)
from silenttwin.agentdojo.grid import (
    AgentDojoGrid,
    GridCell,
    GridTask,
    build_grid,
    load_frozen_inputs,
    write_manifest,
)
from silenttwin.agentdojo.storage import (
    CHECKPOINT_MANIFEST_FILENAME,
    FAILURES_FILENAME,
    RESULT_FILENAME,
    validate_completed_run,
)
from silenttwin.io.jsonl import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/silenttwin/agentdojo"
CATALOG = CONFIG / "catalog-v1.json"
SPLITS = CONFIG / "splits-v1.json"
STRATEGY = CONFIG / "fixtures/deterministic-fake-smoke-candidate-strategies-v1.json"
PAIR = CONFIG / "fixtures/deterministic-fake-smoke-pair-registry-v1.json"
ANALYSIS = CONFIG / "analysis/controlled-v1.json"
PLAN = CONFIG / "grid-plans/controlled-fake-smoke-v1.json"
LOCK = ROOT / "requirements-tier2-agentdojo.lock"


def _has_pinned_agentdojo() -> bool:
    if sys.version_info[:2] != (3, 11) or find_spec("agentdojo") is None:
        return False
    try:
        return metadata.version("agentdojo") == "0.1.35"
    except metadata.PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_pinned_agentdojo(),
    reason="requires Python 3.11 and AgentDojo 0.1.35",
)


def _checked_single_scenario_grid(experiment_id: str) -> AgentDojoGrid:
    inputs = load_frozen_inputs(
        catalog_path=CATALOG,
        splits_path=SPLITS,
        strategy_catalog_path=STRATEGY,
        pair_registry_path=PAIR,
        analysis_plan_path=ANALYSIS,
        dependency_lock_path=LOCK,
    )
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    expanded = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id=experiment_id,
        tier2_track="controlled",
        dataset_split="development",
        groups_per_bundle=1,
    )
    scenario_index = {
        str(row["scenario_id"]): row for row in inputs.scenarios
    }
    tasks: list[GridTask] = []
    cell_index = 0
    for task_id, suite in enumerate(AGENTDOJO_SUITES):
        source = min(
            (task for task in expanded.tasks if task.suite == suite),
            key=lambda task: (
                len(task.cells[0].configuration["scenario_ids"]),
                task.scenario_bundle_hash,
            ),
        )
        scenario_id = str(source.cells[0].configuration["scenario_ids"][0])
        structural_group_id = str(
            scenario_index[scenario_id]["structural_group_id"]
        )
        scenario_bundle_hash = bundle_hash(
            suite=suite,
            dataset_split="development",
            scenario_ids=(scenario_id,),
            structural_group_ids=(structural_group_id,),
        )
        task_cells: list[GridCell] = []
        for source_cell in source.cells:
            scientific = {
                **dict(source_cell.configuration),
                "scenario_ids": [scenario_id],
                "structural_group_ids": [structural_group_id],
                "scenario_bundle_hash": scenario_bundle_hash,
            }
            config = AgentDojoExperimentConfig.from_mapping(scientific)
            configuration = config.scientific_dict()
            configuration_hash = config.configuration_hash
            shard_id = stable_hash(
                {
                    "configuration_hash": configuration_hash,
                    "scenario_bundle_hash": scenario_bundle_hash,
                }
            )
            task_cells.append(
                GridCell(
                    cell_index=cell_index,
                    configuration=configuration,
                    configuration_hash=configuration_hash,
                    shard_id=shard_id,
                )
            )
            cell_index += 1
        tasks.append(
            GridTask(
                task_id=task_id,
                suite=suite,
                scenario_bundle_hash=scenario_bundle_hash,
                replicate=source.replicate,
                cells=tuple(task_cells),
            )
        )
    return AgentDojoGrid(
        experiment_id=experiment_id,
        tier2_track="controlled",
        dataset_split="development",
        tasks=tuple(tasks),
        upstream_binding_hash=inputs.upstream.binding_hash,
    )


def _offline_fake_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_root: Path,
) -> None:
    for name in (
        "AGENTDOJO_MODEL_CACHE",
        "AGENTDOJO_ATTACKER_CHECKPOINT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_TMPDIR",
        "PBS_JOBID",
        "PBS_JOBDIR",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("AGENTDOJO_FAKE_MODEL", "1")
    monkeypatch.setenv("AGENTDOJO_REQUIRES_GPU", "0")
    monkeypatch.setenv("AGENTDOJO_CATALOG", str(CATALOG))
    monkeypatch.setenv("AGENTDOJO_SPLITS", str(SPLITS))
    monkeypatch.setenv("AGENTDOJO_STRATEGY_CATALOG", str(STRATEGY))
    monkeypatch.setenv("AGENTDOJO_PAIR_REGISTRY", str(PAIR))
    monkeypatch.setenv("AGENTDOJO_ANALYSIS_PLAN", str(ANALYSIS))
    monkeypatch.setenv("AGENTDOJO_DEPENDENCY_LOCK", str(LOCK))
    monkeypatch.setenv("AGENTDOJO_TASK_OUTPUT_DIR", str(output_root))


@pytest.mark.parametrize("experiment_id", ("e1", "e2"))
@pytest.mark.parametrize("suite", AGENTDOJO_SUITES)
def test_checked_fake_grid_runs_to_atomic_completion_without_model_or_gpu_path(
    experiment_id: str,
    suite: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from silenttwin.agentdojo import runner

    grid = _checked_single_scenario_grid(experiment_id)
    grid_path = tmp_path / f"{experiment_id}-grid.jsonl"
    write_manifest(grid, grid_path)
    task = next(task for task in grid.tasks if task.suite == suite)
    output_root = tmp_path / f"{experiment_id}-{suite}-runs"
    _offline_fake_environment(monkeypatch, output_root=output_root)

    def forbidden_local_model(*args, **kwargs):
        raise AssertionError("fake smoke must not construct a local/API model client")

    monkeypatch.setattr(runner, "model_client_from_identity", forbidden_local_model)
    source_tree_hash = stable_hash(
        ["checked-fake-run-grid-task", experiment_id, suite]
    )
    monkeypatch.setattr(
        runner,
        "collect_provenance",
        lambda: {
            "source_tree_hash": source_tree_hash,
            "code_revision": "unit-test-fixture",
            "scheduler": {},
        },
    )
    manifests = runner.run_grid_task(grid_manifest=grid_path, task_id=task.task_id)
    assert len(manifests) == len(task.cells)
    assert all(
        model["implementation"] == "deterministic_fake"
        for cell in task.cells
        for model in cell.configuration["models"]
        if model["role"] == "attacker"
    )

    for batch_offset, manifest in enumerate(manifests):
        shard_id = str(manifest["orchestration"]["shard_id"])
        leaf = output_root / f"batch-{batch_offset:03d}-{shard_id[:16]}"
        expected_trial_count = (
            4
            if experiment_id == "e2"
            or manifest["configuration"]["feedback_source"]
            == "matched_shuffled"
            else 2
        )
        assert manifest["status"] == "complete"
        assert manifest["fixture_mode"] is True
        assert manifest["evidence_class"] == "engineering_smoke_only"
        assert manifest["scientific_evidence_eligible"] is False
        assert manifest["expected_trial_count"] == expected_trial_count
        assert manifest["actual_trial_count"] == expected_trial_count
        assert manifest["failure_count"] == 0
        assert manifest["provenance"]["learned_runtime"]["status"] == (
            "not_applicable"
        )
        assert validate_completed_run(
            leaf,
            expected_grid_hash=grid.grid_hash,
            expected_shard_id=shard_id,
        ) == manifest

        rows = read_jsonl(leaf / RESULT_FILENAME)
        assert len(rows) == expected_trial_count
        assert all(row["agentdojo_suite"] == suite for row in rows)
        assert all(row["fixture_mode"] is True for row in rows)
        assert all(
            row["evidence_class"] == "engineering_smoke_only" for row in rows
        )
        if experiment_id == "e2":
            assert all(row["final_plan_attempt_count"] == 1 for row in rows)
        assert read_jsonl(leaf / FAILURES_FILENAME) == []

        checkpoint = json.loads(
            (leaf / CHECKPOINT_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert checkpoint["status"] == "complete"
        assert len(checkpoint["completed_checkpoint_ids"]) == expected_trial_count
        assert len(tuple((leaf / "checkpoints").glob("*.json"))) == (
            expected_trial_count
        )

    # The exact same task must reuse its complete, collision-validated leaves.
    assert runner.run_grid_task(
        grid_manifest=grid_path,
        task_id=task.task_id,
    ) == manifests
