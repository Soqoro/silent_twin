from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from silenttwin.cli import main as cli_main
from silenttwin.config import ExperimentConfig
from silenttwin.experiments.aggregate import aggregate_experiment
from silenttwin.experiments.common import run_experiment
from silenttwin.experiments.grid import build_grid, write_manifest
from silenttwin.io.jsonl import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from silenttwin.metrics.power import PowerEstimate, RequiredSampleSize


def run_config(output_dir: Path, *, runtime: str = "generic_failure", samples: int = 2):
    return ExperimentConfig(
        experiment="e1",
        world_suite="email",
        runtime=runtime,
        attacker="mock_llm",
        query_budget=4,
        seed=42,
        num_samples=samples,
        output_dir=output_dir,
    )


class ResultContractTests(unittest.TestCase):
    def test_atomic_result_has_one_final_summary_and_reuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            first = run_experiment(run_config(output))
            self.assertFalse(first.reused)
            self.assertEqual(
                {
                    "checkpoint_manifest.json",
                    "checkpoints",
                    "failures.jsonl",
                    "manifest.json",
                    "result.jsonl",
                    "run.log",
                },
                {path.name for path in output.iterdir()},
            )
            self.assertEqual(
                2,
                len(list((output / "checkpoints").glob("*.json"))),
            )
            self.assertEqual([], read_jsonl(output / "failures.jsonl"))
            records = read_jsonl(output / "result.jsonl")
            self.assertEqual(3, len(records))
            self.assertEqual("summary", records[-1]["record_type"])
            self.assertEqual(
                1,
                sum(record["record_type"] == "summary" for record in records),
            )

            second = run_experiment(run_config(output))
            self.assertTrue(second.reused)

    def test_all_visible_transcript_aliases_are_validated_and_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            run_experiment(run_config(output))
            records = read_jsonl(output / "result.jsonl")

            private_postselection = json.loads(json.dumps(records))
            private_postselection[0]["postselection_visible_transcript"] = [
                {"value": {"private_state": "theta0"}}
            ]
            with self.assertRaisesRegex(ResultValidationError, "private field"):
                atomic_write_jsonl(root / "private.jsonl", private_postselection)

            inconsistent = json.loads(json.dumps(records))
            inconsistent[0]["decision_visible_transcript"] = []
            with self.assertRaisesRegex(ResultValidationError, "inconsistent"):
                atomic_write_jsonl(root / "inconsistent.jsonl", inconsistent)

    def test_incompatible_existing_output_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_experiment(run_config(output, samples=2))
            with self.assertRaisesRegex(
                ResultValidationError, "incomplete or incompatible"
            ):
                run_experiment(run_config(output, samples=4))

    def test_missing_log_makes_result_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_experiment(run_config(output))
            (output / "run.log").unlink()
            with self.assertRaisesRegex(ResultValidationError, "log file"):
                run_experiment(run_config(output))

    def test_aggregation_keeps_configurations_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            run_experiment(run_config(root / "generic", runtime="generic_failure"))
            run_experiment(run_config(root / "exact", runtime="silenttwin"))
            aggregate = aggregate_experiment(
                experiment="e1",
                input_root=root,
                output_dir=root / "aggregate",
                expected_runs=2,
            )
            self.assertEqual(2, aggregate["run_count"])
            self.assertEqual(2, len(aggregate["configuration_groups"]))
            expected_artifacts = {
                "accuracy_vs_q.csv",
                "analysis_manifest.json",
                "auc_vs_q.csv",
                "entropy_reduction_vs_q.csv",
                "grid_manifest.jsonl",
                "heldout_monitor_fidelity_vs_q.csv",
                "paired_comparisons.csv",
                "summary.csv",
                "summary.json",
                "validated_run_index.json",
            }
            self.assertTrue(
                expected_artifacts.issubset(
                    {path.name for path in (root / "aggregate").iterdir()}
                )
            )
            analysis = json.loads(
                (root / "aggregate" / "analysis_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("analysis_code_revision", analysis)
            self.assertIn("analysis_source_tree_hash", analysis)
            self.assertEqual(5000, analysis["confidence_interval_resamples"])
            self.assertEqual(
                "public_task_cluster_bootstrap",
                analysis["confidence_interval_method"],
            )
            self.assertEqual("development", analysis["dataset_split"])
            self.assertEqual("silenttwin-tier1-v1", analysis["dataset_revision"])
            self.assertEqual("unfrozen", analysis["heldout_sample_size_status"])
            self.assertIsNone(analysis["frozen_test_sample_size"])
            self.assertEqual(
                {
                    "g2_minimum_auc": 0.65,
                    "g2_minimum_accuracy_gain": 0.1,
                    "g2_minimum_replicating_domains": 2,
                    "g2_control_accuracy_margin": 0.05,
                    "g3_minimum_gain": 0.05,
                    "g4_equivalence_margin": 0.05,
                    "g4_requires_ci_inside_margin": True,
                },
                analysis["analysis_plan"]["go_no_go_thresholds"],
            )
            self.assertTrue(analysis["model_environment_summary"])
            for group in aggregate["configuration_groups"]:
                interval = group["cluster_bootstrap_confidence_intervals"][
                    "accuracy_above_prior"
                ]
                self.assertEqual(
                    "accuracy_minus_best_empirical_prior_accuracy",
                    interval["estimand"],
                )
            self.assertEqual(
                "partial", aggregate["go_no_go_gates"]["G1"]["status"]
            )
            self.assertEqual(
                "pass",
                aggregate["go_no_go_gates"]["G1"]["criteria"][
                    "exact_silenttwin_tv_zero"
                ]["status"],
            )

    def test_e1_shuffled_control_pairs_public_task_means(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            grid = build_grid(
                experiment="e1",
                tiers=("tier1",),
                world_suites=("email",),
                runtimes=("generic_failure",),
                attackers=("mock_llm",),
                query_budgets=(0, 16),
                feedback_sources=("genuine", "shuffled"),
                feedback_source_query_budgets={
                    "genuine": (0,),
                    "shuffled": (16,),
                },
                seeds=(42,),
                num_samples=8,
                e1_paired_instances=4,
                e1_public_instances_per_shard=2,
                dataset_split="development",
                dataset_revision="silenttwin-tier1-v1",
            )
            manifest_path = Path(temporary) / "grid.jsonl"
            write_manifest(grid, manifest_path)
            self.assertEqual(6, grid.total_configurations)
            self.assertTrue(
                all(cell.configuration["num_samples"] == 4 for cell in grid.cells)
            )
            self.assertEqual(
                (2, 4),
                (
                    sum(
                        cell.configuration["feedback_source"] == "genuine"
                        for cell in grid.cells
                    ),
                    sum(
                        cell.configuration["feedback_source"] == "shuffled"
                        for cell in grid.cells
                    ),
                ),
            )
            for task in grid.tasks:
                for cell in task.cells:
                    run_experiment(
                        ExperimentConfig(
                            **dict(cell.configuration),
                            output_dir=root / cell.configuration_hash,
                            grid_hash=grid.grid_hash,
                            grid_task_id=task.task_id,
                            shard_id=cell.shard_id,
                        )
                    )
            aggregate = aggregate_experiment(
                experiment="e1",
                input_root=root,
                output_dir=root / "aggregate",
                expected_runs=6,
                expected_grid_manifest=manifest_path,
                expected_grid_hash=grid.grid_hash,
            )
            comparisons = [
                row
                for row in aggregate["paired_comparisons"]
                if row["comparison_kind"] == "e1_shuffled_q16_minus_q0"
            ]
            self.assertEqual(1, len(comparisons))
            comparison = comparisons[0]
            self.assertEqual("public_instance_hash_task_mean", comparison["pairing_unit"])
            self.assertEqual(4, comparison["matched_pair_count"])
            self.assertIsNone(comparison["matched_row_count"])
            self.assertEqual((16, 8), (
                comparison["target_row_count"],
                comparison["reference_row_count"],
            ))
            control = aggregate["go_no_go_gates"]["G2"]["criteria"][
                "shuffled_close_to_q0"
            ]
            self.assertEqual("pass", control["status"])
            self.assertEqual({"ci_inside": [-0.05, 0.05], "source": (
                "configs/silenttwin/analysis-v1.json#go_no_go_thresholds."
                "g2_control_accuracy_margin"
            )}, control["threshold"])

    def test_e2_secondary_oracle_and_random_contrasts_are_task_matched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e2"
            configurations = (
                ("genuine", 16),
                ("oracle", 0),
                ("random", 16),
            )
            for condition, budget in configurations:
                run_experiment(
                    ExperimentConfig(
                        experiment="e2",
                        world_suite="email",
                        runtime="generic_failure",
                        attacker="mock_llm",
                        query_budget=budget,
                        condition=condition,
                        seed=42,
                        num_samples=4,
                        output_dir=root / condition,
                    )
                )
            aggregate = aggregate_experiment(
                experiment="e2",
                input_root=root,
                output_dir=root / "aggregate",
                expected_runs=3,
            )
            secondary = [
                row
                for row in aggregate["paired_comparisons"]
                if row.get("analysis_role") == "secondary"
            ]
            self.assertEqual(
                {
                    "e2_secondary_oracle_minus_genuine",
                    "e2_secondary_genuine_minus_random_selection",
                },
                {row["comparison_kind"] for row in secondary},
            )
            self.assertEqual(4, len(secondary))
            self.assertTrue(all(not row["preregistered"] for row in secondary))
            self.assertTrue(all(row["matched_pair_count"] == 1 for row in secondary))
            self.assertTrue(
                all(row["ci_lower"] is not None and row["ci_upper"] is not None for row in secondary)
            )
            for group in aggregate["configuration_groups"]:
                self.assertNotIn("oracle_gap", group["metrics"])
                self.assertNotIn("random_selection_baseline", group["metrics"])

    def test_exact_pilot_d_aggregate_can_be_frozen_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "e1"
            grid = build_grid(
                experiment="e1",
                tiers=("tier1",),
                world_suites=("email",),
                runtimes=("generic_failure",),
                attackers=("mock_llm",),
                query_budgets=(0, 16),
                feedback_sources=("genuine",),
                seeds=(42,),
                num_samples=4,
                dataset_split="development",
                dataset_revision="silenttwin-tier1-v1",
                pilot_id="pilot_d",
            )
            grid_path = base / "pilot-d-grid.jsonl"
            write_manifest(grid, grid_path)
            for task in grid.tasks:
                for cell in task.cells:
                    run_experiment(
                        ExperimentConfig(
                            **dict(cell.configuration),
                            output_dir=root / cell.configuration_hash,
                            grid_hash=grid.grid_hash,
                            grid_task_id=task.task_id,
                            shard_id=cell.shard_id,
                            pilot_id="pilot_d",
                        )
                    )
            estimate = PowerEstimate(
                sample_size=50,
                effect=0.05,
                discordance=1.0,
                alpha=0.05,
                simulations=10000,
                seed=20260821,
                power=0.81,
            )
            with patch(
                "silenttwin.experiments.aggregate.find_required_sample_size",
                return_value=RequiredSampleSize(0.8, 50, (estimate,)),
            ):
                aggregate_experiment(
                    experiment="e1",
                    input_root=root,
                    output_dir=root / "aggregate",
                    expected_runs=2,
                    expected_grid_manifest=grid_path,
                    expected_grid_hash=grid.grid_hash,
                )
            analysis_path = root / "aggregate" / "analysis_manifest.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            power = analysis["development_power_analysis"]
            self.assertEqual("estimated_not_frozen", power["status"])
            self.assertEqual(
                "conservative_complete_block_binary_success",
                power["discordance_artifact"]["binary_task_estimand"]["name"],
            )
            self.assertEqual(2, power["strata"][0]["public_task_count"])
            freeze_path = base / "freeze.json"
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
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            self.assertEqual(50, freeze["sample_size"])
            self.assertEqual("e1_ordinary_q16_minus_q0", freeze["contrast_id"])

    def test_aggregation_rejects_mixed_matched_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            run_experiment(run_config(root / "generic", runtime="generic_failure"))
            exact_dir = root / "exact"
            run_experiment(run_config(exact_dir, runtime="silenttwin"))

            result_path = exact_dir / "result.jsonl"
            records = read_jsonl(result_path)
            records[0]["public_instance_hash"] = "f" * 64
            atomic_write_jsonl(result_path, records)
            manifest_path = exact_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["result_sha256"] = sha256_file(result_path)
            atomic_write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ResultValidationError, "cohort mismatch"):
                aggregate_experiment(
                    experiment="e1",
                    input_root=root,
                    output_dir=root / "aggregate",
                    expected_runs=2,
                )

    def test_aggregation_rejects_missing_expected_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            run_experiment(run_config(root / "only"))
            with self.assertRaisesRegex(ResultValidationError, "expected 2"):
                aggregate_experiment(
                    experiment="e1",
                    input_root=root,
                    output_dir=root / "aggregate",
                    expected_runs=2,
                )

    def test_observed_only_aggregation_rejects_heldout_test_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            run_experiment(
                ExperimentConfig(
                    experiment="e1",
                    world_suite="email",
                    runtime="generic_failure",
                    attacker="mock_llm",
                    query_budget=4,
                    seed=42,
                    num_samples=2,
                    output_dir=root / "test-cell",
                    dataset_split="test",
                    dataset_revision="silenttwin-tier1-v1",
                    sample_size_freeze_hash="a" * 64,
                    development_manifest_hash="b" * 64,
                    frozen_public_instances=1,
                    primary_contrast_id="e1_ordinary_q16_minus_q0",
                )
            )
            with self.assertRaisesRegex(
                ResultValidationError,
                "held-out test aggregation requires an exact expected-grid manifest",
            ):
                aggregate_experiment(
                    experiment="e1",
                    input_root=root,
                    output_dir=root / "aggregate",
                    expected_runs=1,
                )

    def test_exact_grid_accepts_expected_members_and_rejects_wrong_same_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "e1"
            expected_grid = build_grid(
                experiment="e1",
                tiers=("tier1",),
                world_suites=("email",),
                runtimes=("generic_failure", "silenttwin"),
                attackers=("mock_llm",),
                query_budgets=(4,),
                seeds=(42,),
                num_samples=2,
                dataset_split="development",
                dataset_revision="silenttwin-tier1-v1",
            )
            expected_path = base / "expected-grid.jsonl"
            write_manifest(expected_grid, expected_path)
            for task in expected_grid.tasks:
                for cell in task.cells:
                    selected = ExperimentConfig(
                        **dict(cell.configuration),
                        output_dir=root / cell.configuration_hash,
                        grid_hash=expected_grid.grid_hash,
                        grid_task_id=task.task_id,
                        shard_id=cell.shard_id,
                    )
                    run_experiment(selected)

            aggregate = aggregate_experiment(
                experiment="e1",
                input_root=root,
                output_dir=root / "aggregate-correct",
                expected_runs=2,
                expected_grid_manifest=expected_path,
                expected_grid_hash=expected_grid.grid_hash,
            )
            self.assertEqual("exact_expected_grid", aggregate["grid_validation_mode"])
            validated = json.loads(
                (root / "aggregate-correct" / "validated_run_index.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(2, len(validated["runs"]))

            wrong_grid = build_grid(
                experiment="e1",
                tiers=("tier1",),
                world_suites=("email",),
                runtimes=("generic_failure", "binary_denial"),
                attackers=("mock_llm",),
                query_budgets=(4,),
                seeds=(42,),
                num_samples=2,
                dataset_split="development",
                dataset_revision="silenttwin-tier1-v1",
            )
            self.assertEqual(
                expected_grid.total_configurations,
                wrong_grid.total_configurations,
            )
            wrong_path = base / "wrong-grid.jsonl"
            write_manifest(wrong_grid, wrong_path)
            with self.assertRaisesRegex(ResultValidationError, "membership mismatch"):
                aggregate_experiment(
                    experiment="e1",
                    input_root=root,
                    output_dir=root / "aggregate-wrong",
                    expected_runs=2,
                    expected_grid_manifest=wrong_path,
                    expected_grid_hash=wrong_grid.grid_hash,
                )

    def test_analysis_combines_contiguous_physical_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "e1"
            grid = build_grid(
                experiment="e1",
                tiers=("tier1",),
                world_suites=("email",),
                runtimes=("generic_failure", "silenttwin"),
                attackers=("mock_llm",),
                query_budgets=(4,),
                seeds=(42,),
                num_samples=4,
                episodes_per_shard=2,
                dataset_split="development",
                dataset_revision="silenttwin-tier1-v1",
            )
            manifest_path = base / "grid.jsonl"
            write_manifest(grid, manifest_path)
            for task in grid.tasks:
                for cell in task.cells:
                    run_experiment(
                        ExperimentConfig(
                            **dict(cell.configuration),
                            output_dir=root / cell.configuration_hash,
                            grid_hash=grid.grid_hash,
                            grid_task_id=task.task_id,
                            shard_id=cell.shard_id,
                        )
                    )
            aggregate = aggregate_experiment(
                experiment="e1",
                input_root=root,
                output_dir=root / "aggregate",
                expected_runs=4,
                expected_grid_manifest=manifest_path,
                expected_grid_hash=grid.grid_hash,
            )
            self.assertEqual(4, aggregate["leaf_run_count"])
            self.assertEqual(2, aggregate["analysis_cohort_count"])
            self.assertEqual(
                [4, 4],
                sorted(
                    group["sample_count"]
                    for group in aggregate["configuration_groups"]
                ),
            )
            self.assertTrue(
                all(
                    group["source_leaf_count"] == 2
                    for group in aggregate["configuration_groups"]
                )
            )


if __name__ == "__main__":
    unittest.main()
