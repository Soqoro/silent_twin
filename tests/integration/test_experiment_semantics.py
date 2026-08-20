from __future__ import annotations

import unittest

from silenttwin.config import ExperimentConfig
from silenttwin.experiments import (
    experiment_1_leakage,
    experiment_2_bypass,
    experiment_4_utility,
    experiment_5_ablations,
)


def config(experiment: str, **overrides) -> ExperimentConfig:
    values = {
        "experiment": experiment,
        "world_suite": "email",
        "runtime": "silenttwin",
        "attacker": "bayesian",
        "query_budget": 4,
        "seed": 42,
        "num_samples": 2,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


class ExperimentSemanticsTests(unittest.TestCase):
    def test_e1_positive_and_negative_controls(self) -> None:
        generic = config("e1", runtime="generic_failure")
        generic_samples = [experiment_1_leakage.run_sample(generic, i) for i in range(2)]
        generic_summary = experiment_1_leakage.summarize(generic, generic_samples)
        self.assertEqual(
            1.0, generic_summary["metrics"]["hidden_state_inference_accuracy"]
        )
        self.assertGreater(
            generic_summary["metrics"]["mean_paired_transcript_exact_distance"], 0
        )

        exact = config("e1", runtime="silenttwin")
        exact_samples = [experiment_1_leakage.run_sample(exact, i) for i in range(2)]
        exact_summary = experiment_1_leakage.summarize(exact, exact_samples)
        self.assertEqual(
            0.0, exact_summary["metrics"]["mean_paired_transcript_exact_distance"]
        )
        self.assertEqual(1.0, exact_summary["metrics"]["paired_transcript_equality_rate"])

    def test_e2_has_one_final_attempt_and_feedback_gain(self) -> None:
        adaptive = config(
            "e2", runtime="generic_failure", condition="adaptive", query_budget=4
        )
        samples = [experiment_2_bypass.run_sample(adaptive, i) for i in range(2)]
        self.assertTrue(all(sample["final_exploit_attempts"] == 1 for sample in samples))
        summary = experiment_2_bypass.summarize(adaptive, samples)
        self.assertEqual(1.0, summary["metrics"]["prohibited_effect_rate"])
        self.assertEqual(0.5, summary["metrics"]["feedback_assisted_gain"])

    def test_e2_all_controls_keep_the_same_final_budget(self) -> None:
        for condition in ("adaptive", "random", "oracle", "shuffled"):
            with self.subTest(condition=condition):
                cfg = config("e2", runtime="generic_failure", condition=condition)
                sample = experiment_2_bypass.run_sample(cfg, 0)
                self.assertEqual(1, sample["final_exploit_attempts"])

    def test_e4_preserves_independent_work_and_discards_descendants(self) -> None:
        independent = config("e4", workflow="independent", query_budget=0)
        independent_sample = experiment_4_utility.run_sample(independent, 0)
        self.assertEqual(2, independent_sample["salvage_numerator"])
        self.assertEqual(2, independent_sample["salvage_denominator"])
        self.assertEqual(0, independent_sample["prohibited_effect_indicator"])

        dependent = config("e4", workflow="rejection_dependent", query_budget=0)
        dependent_sample = experiment_4_utility.run_sample(dependent, 0)
        self.assertEqual(0.0, dependent_sample["rejected_descendant_commit_rate"])
        self.assertEqual(0, dependent_sample["dependency_violation"])

    def test_state_independent_fidelity_ablation_does_not_leak(self) -> None:
        cfg = config("e5", ablation="public_fidelity_degradation")
        samples = [experiment_5_ablations.run_sample(cfg, i) for i in range(2)]
        summary = experiment_5_ablations.summarize(cfg, samples)
        self.assertEqual(1.0, summary["metrics"]["paired_transcript_equality_rate"])
        self.assertLess(summary["metrics"]["utility"], 1.0)


if __name__ == "__main__":
    unittest.main()
