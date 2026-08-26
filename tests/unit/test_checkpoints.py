from __future__ import annotations

from pathlib import Path

import pytest

from silenttwin.config import ExperimentConfig
from silenttwin.io.checkpoints import CheckpointStore, episode_id
from silenttwin.io.jsonl import ResultValidationError


def _config(path: Path, **overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "experiment": "e1",
        "attacker": "mock_llm",
        "query_budget": 4,
        "num_samples": 2,
        "output_dir": path,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def _sample(config: ExperimentConfig, index: int) -> dict[str, object]:
    return {
        "record_type": "sample",
        "sample_id": f"sample-{index:06d}",
        "configuration_hash": config.configuration_hash,
    }


def test_checkpoint_resume_loads_each_episode_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = CheckpointStore(tmp_path, config, (0, 1), provenance_hash="source-a")
    store.initialize()
    store.save(0, _sample(config, 0))

    resumed = CheckpointStore(tmp_path, config, (0, 1), provenance_hash="source-a")
    resumed.initialize()
    assert list(resumed.load()) == [0]
    resumed.save(1, _sample(config, 1))
    resumed.mark_complete()
    assert list(resumed.load()) == [0, 1]
    assert episode_id(config, 0) != episode_id(config, 1)


def test_checkpoint_refuses_duplicate_episode(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = CheckpointStore(tmp_path, config, (0, 1), provenance_hash="source-a")
    store.initialize()
    store.save(0, _sample(config, 0))
    with pytest.raises(ResultValidationError, match="already exists"):
        store.save(0, _sample(config, 0))


def test_checkpoint_refuses_changed_configuration_or_source(tmp_path: Path) -> None:
    config = _config(tmp_path)
    CheckpointStore(tmp_path, config, (0, 1), provenance_hash="source-a").initialize()
    with pytest.raises(ResultValidationError, match="source tree"):
        CheckpointStore(
            tmp_path, config, (0, 1), provenance_hash="source-b"
        ).initialize()
    changed = _config(tmp_path, seed=43)
    with pytest.raises(ResultValidationError, match="configuration"):
        CheckpointStore(
            tmp_path, changed, (0, 1), provenance_hash="source-a"
        ).initialize()
