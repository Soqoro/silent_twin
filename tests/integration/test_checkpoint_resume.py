from __future__ import annotations

from pathlib import Path

import pytest

from silenttwin.config import ExperimentConfig, SCHEMA_VERSION
from silenttwin.experiments import common
from silenttwin.io.checkpoints import CheckpointStore, episode_id
from silenttwin.io.jsonl import read_jsonl
from silenttwin.io.jsonl import atomic_write_json


def _sample(config: ExperimentConfig, index: int) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "sample",
        "experiment_id": "e1",
        "sample_id": f"sample-{index:06d}",
        "paired_world_id": f"pair-{index}",
        "public_instance_hash": f"{index:064x}",
        "configuration_hash": config.configuration_hash,
        "code_revision": None,
        "trusted_evaluation": {"private_state": index % 2},
        "secret_prediction_correct": 1,
    }


def _summary(config: ExperimentConfig, samples: object) -> dict[str, object]:
    materialized = list(samples)  # type: ignore[arg-type]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "summary",
        "experiment_id": "e1",
        "tier": config.tier,
        "sample_count": len(materialized),
        "configuration_hash": config.configuration_hash,
        "configuration": config.as_manifest_config(),
        "metrics": {"hidden_state_inference_accuracy": 1.0},
    }


def test_interrupted_run_resumes_without_duplicate_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(
        experiment="e1",
        attacker="mock_llm",
        query_budget=4,
        num_samples=4,
        output_dir=tmp_path / "run",
    )
    first_calls: list[int] = []

    def interrupted_runner(selected: ExperimentConfig, index: int) -> dict[str, object]:
        first_calls.append(index)
        if index == 1:
            raise RuntimeError("synthetic interruption")
        return _sample(selected, index)

    monkeypatch.setattr(common, "_load_experiment", lambda _: (interrupted_runner, _summary))
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        common.run_experiment(config)
    assert first_calls == [0, 1]
    failures = read_jsonl(config.output_dir / "failures.jsonl")
    assert len(failures) == 1
    assert failures[0]["sample_index"] == 1

    resumed_calls: list[int] = []

    def resumed_runner(selected: ExperimentConfig, index: int) -> dict[str, object]:
        resumed_calls.append(index)
        return _sample(selected, index)

    monkeypatch.setattr(common, "_load_experiment", lambda _: (resumed_runner, _summary))
    outcome = common.run_experiment(config)
    assert not outcome.reused
    assert resumed_calls == [1, 2, 3]
    result = read_jsonl(config.output_dir / "result.jsonl")
    samples = result[:-1]
    assert [sample["sample_id"] for sample in samples] == [
        "sample-000000",
        "sample-000001",
        "sample-000002",
        "sample-000003",
    ]
    assert len({sample["episode_id"] for sample in samples}) == 4

    monkeypatch.setattr(
        common,
        "_load_experiment",
        lambda _: (_ for _ in ()).throw(AssertionError("reused run executed")),
    )
    assert common.run_experiment(config).reused


def test_resume_repairs_manifest_after_episode_publish_crash(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment="e1",
        attacker="mock_llm",
        query_budget=0,
        num_samples=2,
        output_dir=tmp_path / "run",
    )
    store = CheckpointStore(config.output_dir, config, (0, 1), "source-hash")
    store.initialize()

    # Reproduce the exact crash window: the episode's atomic rename completed,
    # but the manifest rewrite did not begin.
    identifier = episode_id(config, 0)
    atomic_write_json(
        store.directory / f"{identifier}.json",
        {
            "checkpoint_schema_version": "silenttwin.checkpoint.v1",
            "configuration_hash": config.configuration_hash,
            "episode_id": identifier,
            "sample_index": 0,
            "sample": _sample(config, 0),
        },
    )

    loaded = store.load()
    assert list(loaded) == [0]
    import json

    repaired = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert repaired["status"] == "running"
    assert repaired["completed_episode_ids"] == [identifier]
