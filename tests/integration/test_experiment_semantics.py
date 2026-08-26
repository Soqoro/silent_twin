from __future__ import annotations

import unittest

from silenttwin.config import ExperimentConfig
from silenttwin.experiments import (
    experiment_1_leakage,
    experiment_2_bypass,
    experiment_4_utility,
    experiment_5_ablations,
)
from silenttwin.experiments.experiment_setup import (
    assignment_for_sample,
    make_configured_attacker,
)
from silenttwin.schemas import MonitorDecision, TruePolicyLabel
from silenttwin.worlds.dataset import build_dataset_world_pair


def config(experiment: str, **overrides) -> ExperimentConfig:
    values = {
        "experiment": experiment,
        "world_suite": "email",
        "runtime": "silenttwin",
        "attacker": "mock_llm",
        "query_budget": 4,
        "seed": 42,
        "num_samples": 8 if experiment == "e2" else 2,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def run_and_summarize(module, selected: ExperimentConfig):
    samples = [
        module.run_sample(selected, index)
        for index in range(selected.num_samples)
    ]
    return samples, module.summarize(selected, samples)["metrics"]


class ExperimentSemanticsTests(unittest.TestCase):
    def test_e1_positive_and_exact_negative_controls(self) -> None:
        generic_samples, generic = run_and_summarize(
            experiment_1_leakage,
            config("e1", runtime="generic_failure", num_samples=8),
        )
        self.assertGreater(
            generic["hidden_state_inference_accuracy"],
            generic["best_prior_only_accuracy"],
        )
        self.assertGreater(generic["accuracy_above_prior"], 0.0)
        self.assertGreater(generic["exact_total_variation_distance"], 0.0)
        self.assertTrue(
            all(sample["final_exploit_attempts"] == 0 for sample in generic_samples)
        )
        self.assertTrue(
            all(not sample["committed_effects"] for sample in generic_samples)
        )

        silent_samples, silent = run_and_summarize(
            experiment_1_leakage,
            config("e1", runtime="silenttwin", num_samples=8),
        )
        self.assertEqual(
            silent["best_prior_only_accuracy"],
            silent["hidden_state_inference_accuracy"],
        )
        self.assertEqual(0.0, silent["accuracy_above_prior"])
        self.assertEqual(0.0, silent["exact_total_variation_distance"])
        self.assertEqual(1.0, silent["paired_transcript_equality_rate"])
        self.assertEqual(
            {"theta0", "theta1"},
            {
                sample["trusted_evaluation"]["actual_hidden_state"]
                for sample in silent_samples
            },
        )

    def test_e2_synthetic_causal_controls_and_one_final_budget(self) -> None:
        conditions = {
            "no_probe": 0,
            "genuine": 4,
            "shuffled": 4,
            "random": 4,
            "oracle": 0,
        }
        samples_by_condition = {}
        metrics_by_condition = {}
        for condition, query_budget in conditions.items():
            selected = config(
                "e2",
                runtime="generic_failure",
                condition=condition,
                query_budget=query_budget,
            )
            samples, metrics = run_and_summarize(experiment_2_bypass, selected)
            samples_by_condition[condition] = samples
            metrics_by_condition[condition] = metrics
            self.assertTrue(
                all(sample["final_exploit_attempts"] == 1 for sample in samples),
                condition,
            )
            self.assertEqual(1.0, metrics["mean_final_exploit_attempts"])

        genuine = metrics_by_condition["genuine"]
        self.assertGreater(
            genuine["prohibited_effect_rate"],
            metrics_by_condition["no_probe"]["prohibited_effect_rate"],
        )
        self.assertGreater(
            genuine["prohibited_effect_rate"],
            metrics_by_condition["shuffled"]["prohibited_effect_rate"],
        )
        self.assertGreater(
            genuine["hidden_state_inference_accuracy"],
            metrics_by_condition["no_probe"]["hidden_state_inference_accuracy"],
        )

        # Independently check the oracle against trusted ground truth: it
        # chooses the unique target-state blind spot, which the monitor allows
        # even though TruePolicy prohibits it.
        oracle_samples = samples_by_condition["oracle"]
        for sample in oracle_samples:
            trusted = sample["trusted_evaluation"]
            pair = build_dataset_world_pair(
                sample["template_id"],
                pair_family="monitor_blind_spot",
                instance_id=(
                    f"42-{int(sample['public_instance_index']):08d}"
                ),
                seed=42,
                dataset_revision=sample["dataset_revision"],
                split=sample["dataset_split"],
            )
            world = pair.world_for_state(trusted["actual_hidden_state"])
            selected_action = world.action(sample["selected_final_exploit"])
            self.assertIs(
                MonitorDecision.ALLOW,
                world.monitor.decide(selected_action),
            )
            self.assertIs(
                TruePolicyLabel.PROHIBITED,
                world.true_policy.evaluate(selected_action),
            )
            self.assertEqual("allow", sample["monitor_decision"])
            self.assertEqual("prohibited", sample["true_policy_label"])
            self.assertTrue(sample["ground_truth_prohibited_effect_committed"])
        self.assertEqual(1.0, metrics_by_condition["oracle"]["prohibited_effect_rate"])

        # Random selection is target independent within each independently
        # crossed four-row block. Each fixed exploit is the blind spot in one
        # of the two target states, so its balanced prior baseline is one half.
        random_samples = samples_by_condition["random"]
        by_instance: dict[str, list[dict]] = {}
        for sample in random_samples:
            by_instance.setdefault(sample["public_instance_hash"], []).append(sample)
        self.assertTrue(all(len(block) == 4 for block in by_instance.values()))
        self.assertTrue(
            all(
                len({sample["selected_final_exploit"] for sample in block}) == 1
                for block in by_instance.values()
            )
        )
        independently_expected_random = []
        for sample in random_samples:
            pair = build_dataset_world_pair(
                sample["template_id"],
                pair_family="monitor_blind_spot",
                instance_id=f"42-{int(sample['public_instance_index']):08d}",
                seed=42,
                dataset_revision=sample["dataset_revision"],
                split=sample["dataset_split"],
            )
            world = pair.world_for_state(
                sample["trusted_evaluation"]["actual_hidden_state"]
            )
            selected_action = world.action(sample["selected_final_exploit"])
            independently_expected_random.append(
                world.monitor.decide(selected_action) is MonitorDecision.ALLOW
                and world.true_policy.evaluate(selected_action)
                is TruePolicyLabel.PROHIBITED
            )
        expected_random_rate = sum(independently_expected_random) / len(
            independently_expected_random
        )
        self.assertEqual(0.5, expected_random_rate)
        self.assertEqual(
            expected_random_rate,
            metrics_by_condition["random"]["prohibited_effect_rate"],
        )

    def test_silenttwin_feedback_does_not_improve_e2_over_no_probe(self) -> None:
        _, no_probe = run_and_summarize(
            experiment_2_bypass,
            config(
                "e2",
                runtime="silenttwin",
                condition="no_probe",
                query_budget=0,
            ),
        )
        _, genuine = run_and_summarize(
            experiment_2_bypass,
            config(
                "e2",
                runtime="silenttwin",
                condition="genuine",
                query_budget=4,
            ),
        )
        self.assertEqual(
            no_probe["hidden_state_inference_accuracy"],
            genuine["hidden_state_inference_accuracy"],
        )
        self.assertEqual(
            no_probe["prohibited_effect_rate"],
            genuine["prohibited_effect_rate"],
        )

    def test_e2_requires_complete_four_cell_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "four-cell"):
            config("e2", condition="genuine", num_samples=2)
        with self.assertRaisesRegex(ValueError, "four-cell"):
            config("e2", condition="genuine", num_samples=4, sample_start=2)

    def test_attacker_randomness_is_constant_within_private_assignment_block(self) -> None:
        selected = config(
            "e2", attacker="random", runtime="generic_failure", condition="genuine"
        )
        first_block = [make_configured_attacker(selected, index).seed for index in range(4)]
        second_block = [
            make_configured_attacker(selected, index).seed for index in range(4, 8)
        ]
        self.assertEqual(1, len(set(first_block)))
        self.assertEqual(1, len(set(second_block)))
        self.assertNotEqual(first_block[0], second_block[0])

    def test_assignment_blocks_balance_private_cells_over_material_instances(self) -> None:
        e2_config = config(
            "e2",
            runtime="generic_failure",
            condition="genuine",
            template_id="email.triage_and_reply",
            dataset_split="train",
            num_samples=8,
        )
        e2_assignments = [
            assignment_for_sample(e2_config, index) for index in range(8)
        ]
        for start in (0, 4):
            block = e2_assignments[start : start + 4]
            self.assertEqual(1, len({item.public_instance_index for item in block}))
            self.assertEqual(1, len({item.pair.public_instance_hash for item in block}))
            self.assertEqual(
                {
                    ("theta0", "theta0"),
                    ("theta0", "theta1"),
                    ("theta1", "theta0"),
                    ("theta1", "theta1"),
                },
                {(item.actual_state, item.donor_state) for item in block},
            )
        self.assertNotEqual(
            e2_assignments[0].pair.public_instance_hash,
            e2_assignments[4].pair.public_instance_hash,
        )
        self.assertNotEqual(
            e2_assignments[0].pair.theta0.public_environment[
                "public_instance_content"
            ],
            e2_assignments[4].pair.theta0.public_environment[
                "public_instance_content"
            ],
        )

        e1_config = config(
            "e1",
            runtime="generic_failure",
            template_id="email.triage_and_reply",
            dataset_split="train",
            num_samples=4,
        )
        e1_assignments = [
            assignment_for_sample(e1_config, index) for index in range(4)
        ]
        for start in (0, 2):
            block = e1_assignments[start : start + 2]
            self.assertEqual({"theta0", "theta1"}, {item.actual_state for item in block})
            self.assertEqual(1, len({item.pair.public_instance_hash for item in block}))
        self.assertNotEqual(
            e1_assignments[0].pair.public_instance_hash,
            e1_assignments[2].pair.public_instance_hash,
        )

    def test_e4_preserves_independent_work_and_discards_descendants(self) -> None:
        independent = config(
            "e4", workflow="independent", query_budget=0, attacker="bayesian"
        )
        independent_sample = experiment_4_utility.run_sample(independent, 0)
        self.assertEqual(2, independent_sample["salvage_numerator"])
        self.assertEqual(2, independent_sample["salvage_denominator"])
        self.assertEqual(0, independent_sample["prohibited_effect_indicator"])

        dependent = config(
            "e4", workflow="rejection_dependent", query_budget=0, attacker="bayesian"
        )
        dependent_sample = experiment_4_utility.run_sample(dependent, 0)
        self.assertEqual(0.0, dependent_sample["rejected_descendant_commit_rate"])
        self.assertEqual(0, dependent_sample["dependency_violation"])

    def test_state_independent_fidelity_ablation_does_not_leak(self) -> None:
        selected = config(
            "e5", ablation="public_fidelity_degradation", attacker="bayesian"
        )
        samples = [
            experiment_5_ablations.run_sample(selected, index) for index in range(2)
        ]
        summary = experiment_5_ablations.summarize(selected, samples)
        self.assertEqual(1.0, summary["metrics"]["paired_transcript_equality_rate"])
        self.assertLess(summary["metrics"]["utility"], 1.0)


if __name__ == "__main__":
    unittest.main()
