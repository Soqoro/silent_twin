from __future__ import annotations

from pathlib import Path

import pytest

from silenttwin.config import ExperimentConfig


def test_negative_seed_and_sample_start_fail() -> None:
    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig(experiment="e1", seed=-1)
    with pytest.raises(ValueError, match="sample start"):
        ExperimentConfig(experiment="e1", sample_start=-1)


@pytest.mark.parametrize(
    ("condition", "query_budget"),
    [
        ("no_probe", 0),
        ("oracle", 0),
        ("genuine", 4),
        ("shuffled", 16),
        ("random", 32),
    ],
)
def test_valid_e2_condition_budget_cells(condition: str, query_budget: int) -> None:
    config = ExperimentConfig(
        experiment="e2", condition=condition, query_budget=query_budget
    )
    assert config.condition == condition


@pytest.mark.parametrize(
    ("condition", "query_budget"),
    [("no_probe", 4), ("oracle", 4), ("genuine", 0), ("shuffled", 0), ("random", 0)],
)
def test_invalid_e2_condition_budget_cells_fail(
    condition: str, query_budget: int
) -> None:
    with pytest.raises(ValueError, match="requires query budget"):
        ExperimentConfig(
            experiment="e2", condition=condition, query_budget=query_budget
        )


def test_adaptive_and_pair_family_aliases_are_canonicalized() -> None:
    config = ExperimentConfig(
        experiment="e2",
        condition="adaptive",
        query_budget=4,
        pair_family="blind_spot",
    )
    assert config.condition == "genuine"
    assert config.pair_family == "monitor_blind_spot"


def test_e2_rejects_target_feedback_that_can_end_before_final_action() -> None:
    with pytest.raises(ValueError, match="mandatory final action"):
        ExperimentConfig(
            experiment="e2",
            condition="genuine",
            query_budget=4,
            runtime="opaque_termination",
        )
    # A donor may terminate without retiring the untouched target session.
    shuffled = ExperimentConfig(
        experiment="e2",
        condition="shuffled",
        query_budget=4,
        runtime="opaque_termination",
    )
    assert shuffled.feedback_source == "shuffled"


def test_scientific_shards_preserve_complete_counterbalance_blocks() -> None:
    with pytest.raises(ValueError, match="balanced"):
        ExperimentConfig(experiment="e1", num_samples=3)
    with pytest.raises(ValueError, match="four-cell"):
        ExperimentConfig(
            experiment="e2", condition="genuine", query_budget=4, num_samples=2
        )
    with pytest.raises(ValueError, match="four-cell"):
        ExperimentConfig(
            experiment="e2",
            condition="genuine",
            query_budget=4,
            num_samples=4,
            sample_start=2,
        )


def test_tier2_requires_named_local_llm_and_development_split() -> None:
    with pytest.raises(ValueError, match="attacker='llm'"):
        ExperimentConfig(experiment="e1", tier="tier2")
    with pytest.raises(ValueError, match="model-id"):
        ExperimentConfig(experiment="e1", tier="tier2", attacker="llm")
    config = ExperimentConfig(
        experiment="e1",
        tier="tier2",
        attacker="llm",
        model_id="/approved/local/checkpoint",
        model_revision="a" * 40,
        model_cache_dir="/approved/persistent/cache",
        dataset_split="development",
    )
    assert config.model_id == "/approved/local/checkpoint"
    with pytest.raises(ValueError, match="batch-size 1"):
        ExperimentConfig(
            experiment="e1",
            tier="tier2",
            attacker="llm",
            model_id="/approved/local/checkpoint",
            model_revision="a" * 40,
            model_cache_dir="/approved/persistent/cache",
            batch_size=2,
        )
    with pytest.raises(ValueError, match="sample-size freeze"):
        ExperimentConfig(
            experiment="e1",
            tier="tier2",
            attacker="llm",
            model_id="/approved/local/checkpoint",
            model_revision="a" * 40,
            model_cache_dir="/approved/persistent/cache",
            dataset_split="test",
        )


def test_heldout_execution_requires_exact_freeze_and_bounds_shards() -> None:
    fields = {
        "sample_size_freeze_hash": "a" * 64,
        "development_manifest_hash": "b" * 64,
        "frozen_public_instances": 3,
        "primary_contrast_id": "e1_ordinary_q16_minus_q0",
    }
    config = ExperimentConfig(
        experiment="e1",
        attacker="bayesian",
        query_budget=16,
        num_samples=2,
        sample_start=4,
        dataset_split="test",
        **fields,
    )
    assert config.sample_start + config.num_samples == 6
    with pytest.raises(ValueError, match="exceeds"):
        ExperimentConfig(
            experiment="e1",
            attacker="bayesian",
            query_budget=16,
            num_samples=2,
            sample_start=6,
            dataset_split="test",
            **fields,
        )
    with pytest.raises(ValueError, match="only for"):
        ExperimentConfig(
            experiment="e1",
            num_samples=2,
            dataset_split="development",
            **fields,
        )


def test_operational_location_and_grid_binding_do_not_change_scientific_hash() -> None:
    left = ExperimentConfig(
        experiment="e1",
        output_dir=Path("one"),
        grid_hash="a" * 64,
        grid_task_id=1,
        shard_id="shard-a",
        pilot_id="pilot-a",
    )
    right = ExperimentConfig(
        experiment="e1",
        output_dir=Path("two"),
        grid_hash="b" * 64,
        grid_task_id=9,
        shard_id="shard-b",
        pilot_id="pilot-b",
    )
    assert left.configuration_hash == right.configuration_hash


def test_model_cache_location_does_not_change_scientific_hash() -> None:
    common = {
        "experiment": "e1",
        "tier": "tier2",
        "attacker": "llm",
        "model_id": "org/model",
        "model_revision": "a" * 40,
    }
    left = ExperimentConfig(**common, model_cache_dir="/persistent/cache-a")
    right = ExperimentConfig(**common, model_cache_dir="/persistent/cache-b")
    assert left.configuration_hash == right.configuration_hash
    assert "model_cache_dir" not in left.scientific_dict()
    assert left.operational_dict()["model_cache_dir"] == "/persistent/cache-a"


def test_dataset_split_and_sample_start_are_scientific_factors() -> None:
    base = ExperimentConfig(experiment="e1")
    shifted = ExperimentConfig(experiment="e1", sample_start=16)
    held_out = ExperimentConfig(
        experiment="e1",
        dataset_split="test",
        sample_size_freeze_hash="a" * 64,
        development_manifest_hash="b" * 64,
        frozen_public_instances=8,
        primary_contrast_id="e1_ordinary_q16_minus_q0",
    )
    assert len({base.configuration_hash, shifted.configuration_hash, held_out.configuration_hash}) == 3
