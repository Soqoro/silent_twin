from __future__ import annotations

import json
from pathlib import Path

import pytest

from silenttwin.agentdojo.grid import build_grid, load_frozen_inputs, write_manifest
from silenttwin.agentdojo.runtime_validation import (
    RuntimeArtifactError,
    validate_runtime_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/silenttwin/agentdojo"
CATALOG = CONFIG / "catalog-v1.json"
SPLITS = CONFIG / "splits-v1.json"
STRATEGY = (
    CONFIG
    / "fixtures/deterministic-fake-smoke-candidate-strategies-v1.json"
)
PAIR = CONFIG / "fixtures/deterministic-fake-smoke-pair-registry-v1.json"
ANALYSIS = CONFIG / "analysis/controlled-v1.json"
LOCK = ROOT / "requirements-tier2-agentdojo.lock"
PLAN = CONFIG / "grid-plans/controlled-fake-smoke-v1.json"


def _grid():
    inputs = load_frozen_inputs(
        catalog_path=CATALOG,
        splits_path=SPLITS,
        strategy_catalog_path=STRATEGY,
        pair_registry_path=PAIR,
        analysis_plan_path=ANALYSIS,
        dependency_lock_path=LOCK,
    )
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    return build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e1",
        tier2_track="controlled",
        dataset_split="development",
    )


def _selected(grid):
    records = grid.records()
    return records[0], [
        row for row in records[1:] if row["task_id"] == 0
    ]


def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTDOJO_FAKE_MODEL", "1")
    monkeypatch.setenv("AGENTDOJO_REQUIRES_GPU", "0")


def test_runtime_preflight_accepts_exact_checked_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch)
    metadata, members = _selected(_grid())
    frozen = validate_runtime_artifacts(
        catalog_path=CATALOG,
        splits_path=SPLITS,
        strategy_catalog_path=STRATEGY,
        pair_registry_path=PAIR,
        analysis_plan_path=ANALYSIS,
        dependency_lock_path=LOCK,
        grid_metadata=metadata,
        selected_members=members,
    )
    assert frozen.catalog["catalog_hash"] == members[0]["configuration"][
        "agentdojo_catalog_hash"
    ]
    assert {model["role"] for model in members[0]["configuration"]["models"]} == {
        "attacker"
    }


def test_runtime_preflight_rejects_tampered_artifact_and_fake_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch)
    metadata, members = _selected(_grid())
    tampered = json.loads(CATALOG.read_text(encoding="utf-8"))
    tampered["scenarios"][0]["user_task_id"] = "tampered"
    tampered_path = tmp_path / "catalog.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeArtifactError, match="invalid frozen"):
        validate_runtime_artifacts(
            catalog_path=tampered_path,
            splits_path=SPLITS,
            strategy_catalog_path=STRATEGY,
            pair_registry_path=PAIR,
            analysis_plan_path=ANALYSIS,
            dependency_lock_path=LOCK,
            grid_metadata=metadata,
            selected_members=members,
        )

    monkeypatch.setenv("AGENTDOJO_FAKE_MODEL", "0")
    with pytest.raises(RuntimeArtifactError, match="explicitly match"):
        validate_runtime_artifacts(
            catalog_path=CATALOG,
            splits_path=SPLITS,
            strategy_catalog_path=STRATEGY,
            pair_registry_path=PAIR,
            analysis_plan_path=ANALYSIS,
            dependency_lock_path=LOCK,
            grid_metadata=metadata,
            selected_members=members,
        )


def test_grid_task_tamper_fails_before_model_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from silenttwin.agentdojo import runner

    _environment(monkeypatch)
    grid = _grid()
    manifest = tmp_path / "grid.jsonl"
    write_manifest(grid, manifest)
    tampered = json.loads(CATALOG.read_text(encoding="utf-8"))
    tampered["scenarios"][0]["user_task_id"] = "tampered"
    tampered_path = tmp_path / "catalog.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setenv("AGENTDOJO_CATALOG", str(tampered_path))
    monkeypatch.setenv("AGENTDOJO_SPLITS", str(SPLITS))
    monkeypatch.setenv("AGENTDOJO_STRATEGY_CATALOG", str(STRATEGY))
    monkeypatch.setenv("AGENTDOJO_PAIR_REGISTRY", str(PAIR))
    monkeypatch.setenv("AGENTDOJO_ANALYSIS_PLAN", str(ANALYSIS))
    monkeypatch.setenv("AGENTDOJO_DEPENDENCY_LOCK", str(LOCK))
    monkeypatch.setenv("AGENTDOJO_TASK_OUTPUT_DIR", str(tmp_path / "out"))
    constructed = False

    def forbidden_client(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("model construction must not be reached")

    monkeypatch.setattr(runner, "_client_for_identity", forbidden_client)
    with pytest.raises(RuntimeArtifactError, match="invalid frozen"):
        runner.run_grid_task(grid_manifest=manifest, task_id=0)
    assert constructed is False


@pytest.mark.parametrize(
    "ephemeral_variable",
    (
        "AGENTDOJO_MONITOR_CHECKPOINT",
        "AGENTDOJO_MONITOR_CHECKPOINT_MONITOR_A",
        "AGENTDOJO_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ),
)
def test_pair_observation_preflight_rejects_every_ephemeral_model_path(
    ephemeral_variable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from silenttwin.agentdojo import runner

    scratch = tmp_path / "slurm-tmp"
    scratch.mkdir()
    ephemeral = scratch / ephemeral_variable.lower()
    ephemeral.mkdir()
    persistent_checkpoint = tmp_path / "persistent-monitor"
    persistent_checkpoint.mkdir()
    for variable in (
        "AGENTDOJO_MONITOR_CHECKPOINT",
        "AGENTDOJO_MONITOR_CHECKPOINT_MONITOR_A",
        "AGENTDOJO_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    monkeypatch.setenv(
        "AGENTDOJO_MONITOR_CHECKPOINT", str(persistent_checkpoint)
    )
    monkeypatch.setenv(ephemeral_variable, str(ephemeral))

    def forbidden_environment_check(**kwargs):
        raise AssertionError(
            "package validation and model construction must follow path preflight"
        )

    monkeypatch.setattr(
        runner, "validate_environment_integrity", forbidden_environment_check
    )
    with pytest.raises(
        RuntimeArtifactError, match="ephemeral scheduler scratch SLURM_TMPDIR"
    ):
        runner._preflight_pair_observation_environment(
            strategy_catalog={
                "monitor_profiles": [
                    {
                        "profile_id": "monitor-a",
                        "implementation": "local_transformers",
                        "runtime_fingerprint": "sha256:" + "a" * 64,
                    }
                ]
            },
            dependency_lock_path=LOCK,
        )


@pytest.mark.parametrize("scratch_variable", ("PBS_JOBDIR", "TMPDIR"))
def test_pair_observation_preflight_rejects_pbs_ephemeral_model_paths(
    scratch_variable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from silenttwin.agentdojo import runner

    scratch = tmp_path / scratch_variable.lower()
    scratch.mkdir()
    ephemeral = scratch / "model-cache"
    ephemeral.mkdir()
    persistent_checkpoint = tmp_path / "persistent-monitor"
    persistent_checkpoint.mkdir()
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv("PBS_JOBID", "123.gaas")
    monkeypatch.setenv(scratch_variable, str(scratch))
    monkeypatch.setenv(
        "AGENTDOJO_MONITOR_CHECKPOINT", str(persistent_checkpoint)
    )
    monkeypatch.setenv("AGENTDOJO_MODEL_CACHE", str(ephemeral))

    monkeypatch.setattr(
        runner,
        "validate_environment_integrity",
        lambda **_: pytest.fail("runtime validation must follow path preflight"),
    )
    with pytest.raises(RuntimeArtifactError, match=scratch_variable):
        runner._preflight_pair_observation_environment(
            strategy_catalog={
                "monitor_profiles": [
                    {
                        "profile_id": "monitor-a",
                        "implementation": "local_transformers",
                        "runtime_fingerprint": "sha256:" + "a" * 64,
                    }
                ]
            },
            dependency_lock_path=LOCK,
        )


def test_pair_observation_preflight_allows_checkpoint_in_pbs_home_jobdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from silenttwin.agentdojo import runner

    home = tmp_path / "home"
    checkpoint = home / "persistent-monitor"
    model_cache = home / "persistent-model-cache"
    checkpoint.mkdir(parents=True)
    model_cache.mkdir()
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setenv("PBS_JOBID", "123.gaas")
    monkeypatch.setenv("PBS_JOBDIR", str(home))
    monkeypatch.setenv("PBS_O_HOME", str(home))
    monkeypatch.setenv("AGENTDOJO_MONITOR_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("AGENTDOJO_MODEL_CACHE", str(model_cache))
    monkeypatch.setattr(
        runner,
        "validate_environment_integrity",
        lambda **_: None,
    )

    runner._preflight_pair_observation_environment(
        strategy_catalog={
            "monitor_profiles": [
                {
                    "profile_id": "monitor-a",
                    "implementation": "local_transformers",
                    "runtime_fingerprint": "sha256:" + "a" * 64,
                }
            ]
        },
        dependency_lock_path=LOCK,
    )


def test_pair_observation_preflight_requires_checkpoint_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from silenttwin.agentdojo import runner

    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.delenv("AGENTDOJO_MONITOR_CHECKPOINT", raising=False)
    monkeypatch.delenv(
        "AGENTDOJO_MONITOR_CHECKPOINT_MONITOR_A", raising=False
    )
    monkeypatch.setattr(
        runner,
        "validate_environment_integrity",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("environment check must not precede checkpoint validation")
        ),
    )
    with pytest.raises(RuntimeArtifactError, match="persistent local checkpoint"):
        runner._preflight_pair_observation_environment(
            strategy_catalog={
                "monitor_profiles": [
                    {
                        "profile_id": "monitor-a",
                        "implementation": "local_transformers",
                        "runtime_fingerprint": "sha256:" + "a" * 64,
                    }
                ]
            },
            dependency_lock_path=LOCK,
        )


def test_generate_pair_observations_cli_forwards_action_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from silenttwin.agentdojo import (
        action_eligibility,
        catalog,
        compat,
        pair_mining,
        runner,
        splits,
    )

    documents = {
        "catalog.json": {
            "agentdojo_source_revision": "pinned-source",
            "agentdojo_benchmark_version": "pinned-benchmark",
        },
        "splits.json": {},
        "strategies.json": {"monitor_profiles": []},
        "eligibility.json": {"sentinel": "checked-eligibility"},
    }
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        path = tmp_path / name
        path.write_text(json.dumps(document), encoding="utf-8")
        paths[name] = path

    monkeypatch.setattr(catalog, "validate_catalog", lambda _: None)
    monkeypatch.setattr(splits, "validate_split_manifest", lambda *_, **__: None)
    monkeypatch.setattr(
        action_eligibility,
        "validate_action_eligibility_manifest",
        lambda *_, **__: "a" * 64,
    )
    monkeypatch.setattr(
        pair_mining, "validate_candidate_strategy_catalog", lambda _: None
    )
    monkeypatch.setattr(
        pair_mining, "validate_estimation_strategy_coverage", lambda *_, **__: None
    )
    monkeypatch.setattr(compat, "assert_compatible", lambda *_, **__: None)
    monkeypatch.setattr(
        runner,
        "_preflight_pair_observation_environment",
        lambda **_: {"status": "captured"},
    )
    monkeypatch.setattr(
        runner,
        "collect_provenance",
        lambda: {"source_tree_hash": "b" * 64},
    )

    captured: dict[str, object] = {}
    published: dict[str, object] = {}

    def checked_generator(
        *, action_eligibility_manifest: object, **kwargs: object
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        captured["action_eligibility_manifest"] = action_eligibility_manifest
        captured.update(kwargs)
        return [], {
            "observation_set_hash": "c" * 64,
            "protocol_disposition": "estimation_only_action_representable",
            "action_eligibility_manifest_hash": "a" * 64,
        }

    monkeypatch.setattr(runner, "generate_pair_observation_set", checked_generator)
    monkeypatch.setattr(
        runner,
        "atomic_write_jsonl",
        lambda path, rows: published.update(
            {"observations_path": path, "observations": rows}
        ),
    )
    monkeypatch.setattr(
        runner,
        "atomic_write_json",
        lambda path, document: published.update(
            {"manifest_path": path, "manifest": document}
        ),
    )
    observations = tmp_path / "train.jsonl"
    manifest = tmp_path / "train.manifest.json"
    assert runner.main(
        [
            "generate-pair-observations",
            "--catalog",
            str(paths["catalog.json"]),
            "--splits",
            str(paths["splits.json"]),
            "--strategy-catalog",
            str(paths["strategies.json"]),
            "--action-eligibility",
            str(paths["eligibility.json"]),
            "--dependency-lock",
            str(tmp_path / "requirements.lock"),
            "--dataset-split",
            "train",
            "--observations-output",
            str(observations),
            "--observation-manifest-output",
            str(manifest),
        ]
    ) == 0

    assert captured["action_eligibility_manifest"] == documents[
        "eligibility.json"
    ]
    assert captured["dataset_split"] == "train"
    assert published["observations_path"] == observations
    assert published["observations"] == []
    assert published["manifest_path"] == manifest
    assert published["manifest"] == {
        "observation_set_hash": "c" * 64,
        "protocol_disposition": "estimation_only_action_representable",
        "action_eligibility_manifest_hash": "a" * 64,
    }
    assert json.loads(capsys.readouterr().out)["observation_count"] == 0
