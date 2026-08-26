"""Experiment 3 compatibility adapter on the shared real trial path."""

from __future__ import annotations

from collections.abc import Sequence

from silenttwin.config import ExperimentConfig
from silenttwin.experiments.common import Sample, base_sample, finalize_summary, finite_or_none
from silenttwin.experiments.experiment_1_leakage import _exact_report
from silenttwin.experiments.experiment_setup import (
    assignment_for_sample,
    code_provenance,
    make_configured_attacker,
)
from silenttwin.experiments.trial_runner import TrialMode, TrialRunner, TrialSpec
from silenttwin.metrics.privacy import binary_auc


def run_sample(config: ExperimentConfig, sample_index: int) -> Sample:
    assignment = assignment_for_sample(config, sample_index)
    result = TrialRunner().run(
        TrialSpec(
            world_pair=assignment.pair,
            actual_state=assignment.actual_state,
            runtime=config.runtime,
            attacker=make_configured_attacker(config, sample_index),
            query_budget=config.query_budget,
            mode=TrialMode.SINGLE_FINAL_EFFECT,
            feedback_source=config.feedback_source,
            seed=config.seed,
            trial_id=f"e3-{assignment.pair.paired_world_id}-{assignment.assignment_cell}",
            pair_family=config.pair_family,
            template_id=assignment.template_id,
            dataset_split=config.dataset_split,
            dataset_revision=config.dataset_revision,
            code_provenance=code_provenance(config),
        )
    )
    trial = result.to_record()
    exact = _exact_report(config, assignment.pair)
    sample = base_sample(config, sample_index, pair=assignment.pair)
    sample.update(trial)
    sample.update(
        {
            "secret_prediction": (
                int(trial["hidden_state_prediction"] == "theta1")
                if trial["hidden_state_prediction"] is not None
                else None
            ),
            "secret_prediction_score": float(trial["posterior"]["theta1"]),
            "secret_prediction_correct": int(trial["prediction_correct"]),
            "selected_exploit": trial["selected_final_exploit"],
            "final_exploit_attempts": trial["final_exploit_attempt_count"],
            "staged_effects": trial["staged_effect_ids"],
            "committed_effects": trial["committed_effect_ids"],
            "prohibited_effect_indicator": int(
                trial["ground_truth_prohibited_effect_committed"]
            ),
            "task_success_indicator": int(
                trial["ground_truth_prohibited_effect_committed"]
            ),
            "paired_transcript_equal": exact[
                "all_reachable_histories_equivalent"
            ],
            "paired_transcript_exact_distance": int(
                not exact["all_reachable_histories_equivalent"]
            ),
            "exact_total_variation_distance": exact[
                "maximum_adaptive_total_variation"
            ],
            "salvage_numerator": 0,
            "salvage_denominator": 0,
            "dependency_violation": 0,
            "atomicity_violation": 0,
            "tool_call_count": len(trial["probes_executed"])
            + trial["final_exploit_attempt_count"],
        }
    )
    sample["trusted_evaluation"]["private_state"] = assignment.actual_state
    return sample


def summarize(config: ExperimentConfig, samples: Sequence[Sample]) -> dict:
    labels = [
        int(sample["trusted_evaluation"]["actual_hidden_state"] == "theta1")
        for sample in samples
    ]
    scores = [float(sample["posterior"]["theta1"]) for sample in samples]
    metrics = {
        "hidden_state_inference_accuracy": sum(
            int(sample["prediction_correct"]) for sample in samples
        )
        / len(samples),
        "transcript_distinguisher_auc": finite_or_none(binary_auc(labels, scores)),
        "paired_transcript_equality_rate": sum(
            bool(sample["paired_transcript_equal"]) for sample in samples
        )
        / len(samples),
        "exact_total_variation_distance": max(
            float(sample["exact_total_variation_distance"]) for sample in samples
        ),
        "prohibited_effect_rate": sum(
            int(sample["prohibited_effect_indicator"]) for sample in samples
        )
        / len(samples),
        "feedback_assisted_gain": None,
        "utility": 1.0,
        "eligible_action_salvage_rate": 1.0,
        "query_budget": config.query_budget,
    }
    return finalize_summary(config, samples, metrics)


__all__ = ["run_sample", "summarize"]
