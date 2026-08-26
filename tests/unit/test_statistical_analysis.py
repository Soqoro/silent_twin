from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from silenttwin.cli import main as cli_main
from silenttwin.config import stable_hash
from silenttwin.experiments.aggregate import (
    RunData,
    _model_environment_summary,
    _paired_task_power_evidence,
    _validate_dataset_analysis_scope,
)
from silenttwin.io.jsonl import ResultValidationError, atomic_write_json
from silenttwin.metrics.calibration import (
    binary_brier_score,
    binary_log_loss,
    expected_calibration_error,
    mean_entropy_reduction,
    posterior_entropy,
)
from silenttwin.metrics.confidence_intervals import (
    cluster_bootstrap_statistic_ci,
    paired_cluster_permutation_p_value,
    paired_task_cluster_bootstrap_ci,
    task_cluster_bootstrap_ci,
)
from silenttwin.metrics.power import (
    find_required_sample_size,
    make_sample_size_freeze,
    paired_discordance_rate,
    simulate_paired_binary_power,
    validate_sample_size_freeze,
)
from silenttwin.metrics.privacy import binary_auc
from silenttwin.worlds.dataset import DATASET_REVISION


class CalibrationMetricTests(unittest.TestCase):
    def test_entropy_and_proper_scores_have_independent_oracles(self) -> None:
        self.assertEqual(1.0, posterior_entropy({"theta0": 0.5, "theta1": 0.5}))
        self.assertEqual(
            1.0,
            mean_entropy_reduction(
                {"theta0": 0.5, "theta1": 0.5},
                (
                    {"theta0": 1.0, "theta1": 0.0},
                    {"theta0": 0.0, "theta1": 1.0},
                ),
            ),
        )
        self.assertAlmostEqual(0.025, binary_brier_score([0, 1], [0.1, 0.8]))
        self.assertAlmostEqual(
            -(0.5 * (math.log(0.9) + math.log(0.8))),
            binary_log_loss([0, 1], [0.1, 0.8]),
        )
        self.assertAlmostEqual(0.15, expected_calibration_error([0, 1], [0.1, 0.8], bins=2))


class ClusteredInferenceTests(unittest.TestCase):
    def test_task_cluster_mean_does_not_treat_decoding_seeds_as_units(self) -> None:
        values = [0.0, 0.0, 1.0, 1.0, 1.0]
        task_ids = ["task-a", "task-a", "task-b", "task-b", "task-b"]
        first = task_cluster_bootstrap_ci(values, task_ids, resamples=500, seed=11)
        second = task_cluster_bootstrap_ci(values, task_ids, resamples=500, seed=11)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.5)
        self.assertGreaterEqual(first[1], 0.5)

    def test_paired_task_ci_and_permutation_keep_pairs_together(self) -> None:
        left = [1, 1, 1, 1, 1, 1, 1, 1]
        right = [0, 0, 0, 0, 0, 0, 0, 0]
        task_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
        interval = paired_task_cluster_bootstrap_ci(
            left, right, task_ids, resamples=200, seed=7
        )
        self.assertEqual((1.0, 1.0), interval)
        self.assertEqual(
            0.125,
            paired_cluster_permutation_p_value(left, right, task_ids),
        )

    def test_cluster_bootstrap_supports_auc_as_a_nonlinear_statistic(self) -> None:
        observations = [(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]
        task_ids = ["n1", "n2", "p1", "p2"]
        interval = cluster_bootstrap_statistic_ci(
            observations,
            task_ids,
            lambda rows: binary_auc(
                [label for label, _ in rows], [score for _, score in rows]
            ),
            resamples=500,
            seed=3,
        )
        self.assertEqual((1.0, 1.0), interval)


class PowerAndFreezeTests(unittest.TestCase):
    @staticmethod
    def _power_run(name: str, values: list[int]) -> RunData:
        samples = tuple(
            {
                "sample_id": f"sample-{index:06d}",
                "public_instance_hash": "task-a" if index < 2 else "task-b",
                "prediction_correct": value,
            }
            for index, value in enumerate(values)
        )
        return RunData(
            directory=Path(name),
            manifest={
                "configuration_hash": name,
                "configuration": {"query_budget": 16 if name == "target" else 0},
            },
            samples=samples,
            summary={"metrics": {}},
        )

    def test_power_discordance_uses_one_complete_block_binary_outcome_per_task(self) -> None:
        target = self._power_run("target", [1, 0, 1, 1])
        reference = self._power_run("reference", [0, 0, 0, 0])
        evidence = _paired_task_power_evidence(
            target,
            reference,
            "hidden_state_inference_accuracy",
            expected_rows_per_task=2,
        )
        # Row-level discordance would be 3/4.  The conservative complete-block
        # outcomes are target=[0,1], reference=[0,0], hence one discordant task.
        self.assertEqual(0.5, evidence["paired_discordance"])
        self.assertEqual(2, evidence["public_task_count"])
        self.assertEqual(64, len(evidence["paired_task_outcomes_hash"]))
        self.assertEqual(
            {
                "target0_reference0": 1,
                "target0_reference1": 0,
                "target1_reference0": 1,
                "target1_reference1": 0,
            },
            evidence["paired_binary_cell_counts"],
        )

    def test_exact_heldout_analysis_rejects_a_partial_logical_cell(self) -> None:
        partial = RunData(
            directory=Path("partial-test-cell"),
            manifest={
                "configuration_hash": "partial",
                "configuration": {
                    "dataset_split": "test",
                    "dataset_revision": DATASET_REVISION,
                    "feedback_source": "genuine",
                    "sample_start": 0,
                    "num_samples": 2,
                    "frozen_public_instances": 2,
                    "sample_size_freeze_hash": "a" * 64,
                    "development_manifest_hash": "b" * 64,
                    "primary_contrast_id": "e1_ordinary_q16_minus_q0",
                },
            },
            samples=({}, {}),
            summary={"metrics": {}},
        )
        with self.assertRaisesRegex(
            ResultValidationError, "every combined logical cell to cover"
        ):
            _validate_dataset_analysis_scope(
                [partial],
                experiment="e1",
                grid_validation_mode="exact_expected_grid",
            )

    def test_discordance_and_power_simulation_are_deterministic(self) -> None:
        self.assertEqual(0.5, paired_discordance_rate([1, 0, 1, 0], [0, 0, 0, 0]))
        first = simulate_paired_binary_power(
            80, effect=0.2, discordance=0.3, simulations=400, seed=19
        )
        second = simulate_paired_binary_power(
            80, effect=0.2, discordance=0.3, simulations=400, seed=19
        )
        null = simulate_paired_binary_power(
            80, effect=0.0, discordance=0.3, simulations=400, seed=19
        )
        self.assertEqual(first, second)
        self.assertGreater(first.power, null.power)

    def test_required_size_and_freeze_are_hash_bound(self) -> None:
        analysis = find_required_sample_size(
            (20, 40, 80),
            effect=0.25,
            discordance=0.35,
            target_power=0.5,
            simulations=300,
            seed=5,
        )
        self.assertTrue(analysis.estimates)
        selected = analysis.selected_sample_size or analysis.estimates[-1].sample_size
        estimate = next(
            item for item in analysis.estimates if item.sample_size == selected
        )
        freeze = make_sample_size_freeze(
            experiment_id="e2",
            dataset_revision=DATASET_REVISION,
            development_manifest_hash="d" * 64,
            contrast_id="e2_genuine_q16_minus_shuffled_q16",
            sample_size=selected,
            power_estimate=estimate,
        )
        self.assertEqual(
            selected,
            validate_sample_size_freeze(
                freeze,
                experiment_id="e2",
                dataset_revision=DATASET_REVISION,
                contrast_id="e2_genuine_q16_minus_shuffled_q16",
                development_manifest_hash="d" * 64,
            ),
        )
        tampered = {**freeze, "sample_size": selected + 1}
        with self.assertRaisesRegex(ValueError, "freeze hash"):
            validate_sample_size_freeze(
                tampered,
                experiment_id="e2",
                dataset_revision=DATASET_REVISION,
                contrast_id="e2_genuine_q16_minus_shuffled_q16",
            )

    def test_cli_freezes_exact_development_manifest_recommendation(self) -> None:
        manifest = {
            "schema_version": "silenttwin.analysis-manifest.v1",
            "experiment_id": "e2",
            "dataset_split": "development",
            "dataset_revision": DATASET_REVISION,
            "development_power_analysis": {
                "status": "estimated_not_frozen",
                "strata": [
                    {"contrast": "e2_genuine_q16_minus_shuffled_q16"}
                ],
                "simulation_power": {
                    "selected_sample_size": 50,
                    "estimates": [
                        {
                            "sample_size": 50,
                            "effect": 0.1,
                            "discordance": 0.2,
                            "alpha": 0.05,
                            "simulations": 10000,
                            "seed": 7,
                            "power": 0.83,
                        }
                    ],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path = root / "analysis_manifest.json"
            freeze_path = root / "freeze.json"
            atomic_write_json(analysis_path, manifest)
            self.assertEqual(
                0,
                cli_main(
                    [
                        "freeze-sample-size",
                        "--analysis-manifest",
                        str(analysis_path),
                        "--output-file",
                        str(freeze_path),
                    ]
                ),
            )
            import json

            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        self.assertEqual(50, freeze["sample_size"])
        self.assertEqual(stable_hash(manifest), freeze["development_manifest_hash"])
        self.assertEqual(
            50,
            validate_sample_size_freeze(
                freeze,
                experiment_id="e2",
                dataset_revision=DATASET_REVISION,
                contrast_id="e2_genuine_q16_minus_shuffled_q16",
                development_manifest_hash=stable_hash(manifest),
            ),
        )
        with self.assertRaisesRegex(ValueError, "different experiment"):
            validate_sample_size_freeze(
                freeze,
                experiment_id="e1",
                dataset_revision=DATASET_REVISION,
                contrast_id="e2_genuine_q16_minus_shuffled_q16",
            )


class ModelEnvironmentReportingTests(unittest.TestCase):
    def test_requested_and_resolved_local_identity_and_cost_are_separate(self) -> None:
        run = RunData(
            directory=Path("tier2"),
            manifest={
                "configuration_hash": "tier2-cell",
                "configuration": {
                    "tier": "tier2",
                    "model_id": "/requested/model",
                    "model_revision": "a" * 40,
                    "model_cache_dir": "/cache",
                    "dtype": "bfloat16",
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "batch_size": 2,
                    "decoding_seed": 7,
                },
            },
            samples=(
                {
                    "latency_ms": 20.0,
                    "model_provenance": {
                        "model_id": "/resolved/model",
                        "model_revision": "b" * 40,
                        "tokenizer_revision": "c" * 40,
                        "input_tokens": 101,
                        "output_tokens": 11,
                        "latency_ms": 12.5,
                        "retries": 1,
                        "failures": [],
                        "metadata": {
                            "client": "local_transformers",
                            "requested_model_revision": "a" * 40,
                            "requested_tokenizer_revision": "d" * 40,
                            "local_checkpoint_fingerprint": "sha256:" + "e" * 64,
                            "local_checkpoint_verification_mode": "full_tree_sha256_audit",
                            "local_checkpoint_manifest_hash": "f" * 64,
                            "chat_template_hash": "1" * 64,
                            "dtype": "bfloat16",
                            "device": "cuda:0",
                            "decoding_seed": 99,
                            "torch_version": "2.test",
                            "transformers_version": "5.test",
                            "cuda_version": "13.test",
                            "gpu_name": "Test GPU",
                            "local_files_only": True,
                            "external_api_calls": 0,
                        },
                    }
                },
            ),
            summary={"metrics": {}},
        )
        records = _model_environment_summary([run])
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("a" * 40, record["requested"]["model_revision"])
        self.assertEqual("b" * 40, record["resolved"]["model_revision"])
        self.assertEqual("d" * 40, record["requested"]["tokenizer_revision"])
        self.assertEqual("c" * 40, record["resolved"]["tokenizer_revision"])
        self.assertTrue(record["execution_environment"]["local_files_only"])
        self.assertEqual([7], record["requested_decoding_seeds"])
        self.assertEqual([99], record["observed_generation_seeds"])
        self.assertEqual(101, record["cost_accounting"]["input_tokens"])
        self.assertEqual(11, record["cost_accounting"]["output_tokens"])
        self.assertEqual(12.5, record["cost_accounting"]["latency_ms"])
        self.assertEqual(20.0, record["cost_accounting"]["trial_wall_time_ms"])
        self.assertAlmostEqual(
            12.5 / 3_600_000.0,
            record["cost_accounting"]["single_device_model_call_time_proxy_hours"],
        )
        self.assertFalse(record["cost_accounting"]["monetary_cost_reported"])


if __name__ == "__main__":
    unittest.main()
