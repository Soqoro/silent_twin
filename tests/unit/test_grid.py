from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from silenttwin.config import stable_hash
from silenttwin.experiments.grid import (
    ExperimentGrid,
    GridError,
    build_grid,
    grid_from_pilot,
    load_grid_manifest,
    write_manifest,
)
from silenttwin.metrics.power import PowerEstimate, make_sample_size_freeze


TIER2_MODEL = {
    "model_id": "/approved/models/example",
    "model_revision": "a" * 40,
    "model_cache_dir": "/persistent/cache/silenttwin",
}


class PilotGridTests(unittest.TestCase):
    def test_pilot_a_uses_explicit_balanced_trial_row_counts(self) -> None:
        e1 = grid_from_pilot("pilot_a", experiment="e1")
        e2 = grid_from_pilot("pilot_a", experiment="e2")

        self.assertEqual((6, 6), (e1.total_tasks, e1.total_configurations))
        self.assertEqual((8, 8), (e2.total_tasks, e2.total_configurations))
        self.assertEqual(
            {
                ("genuine", 0, 32),
                ("genuine", 4, 32),
                ("shuffled", 4, 64),
            },
            {
                (
                    cell.configuration["feedback_source"],
                    cell.configuration["query_budget"],
                    cell.configuration["num_samples"],
                )
                for cell in e1.cells
            },
        )
        self.assertEqual({64}, {cell.configuration["num_samples"] for cell in e2.cells})

    def test_pilot_b_is_batched_and_filters_invalid_e2_crosses(self) -> None:
        e1 = grid_from_pilot("pilot_b", experiment="e1")
        e2 = grid_from_pilot("pilot_b", experiment="e2")

        self.assertEqual((40, 160, 4), (e1.total_tasks, e1.total_configurations, e1.cells_per_task))
        self.assertEqual(
            {"authorization", "monitor_blind_spot"},
            {cell.configuration["pair_family"] for cell in e1.cells},
        )
        self.assertTrue(all(len(task.cells) == 4 for task in e1.tasks))
        self.assertEqual(
            {
                (0, "genuine"),
                (4, "genuine"),
                (16, "genuine"),
                (16, "shuffled"),
                (32, "genuine"),
            },
            {
                (
                    cell.configuration["query_budget"],
                    cell.configuration["feedback_source"],
                )
                for cell in e1.cells
            },
        )

        self.assertEqual((32, 128, 4), (e2.total_tasks, e2.total_configurations, e2.cells_per_task))
        valid_pairs = {
            (0, "no_probe"),
            (0, "oracle"),
            (4, "genuine"),
            (4, "shuffled"),
            (16, "genuine"),
            (16, "shuffled"),
            (32, "genuine"),
            (32, "shuffled"),
        }
        self.assertEqual(
            valid_pairs,
            {
                (cell.configuration["query_budget"], cell.configuration["condition"])
                for cell in e2.cells
            },
        )

    def test_tier2_presets_use_twenty_trial_rows_per_shard(self) -> None:
        expected_counts = {
            ("pilot_c", "e1"): 20,
            ("pilot_c", "e2"): 48,
            ("pilot_d", "e1"): 480,
            ("pilot_d", "e2"): 960,
        }
        for (pilot, experiment), count in expected_counts.items():
            with self.subTest(pilot=pilot, experiment=experiment):
                grid = grid_from_pilot(
                    pilot,
                    experiment=experiment,
                    model_overrides=TIER2_MODEL,
                    dataset_revision_override="silenttwin-tier1-v1",
                )
                self.assertEqual(count, grid.total_tasks)
                self.assertEqual(count, grid.total_configurations)
                self.assertEqual({20}, {cell.configuration["num_samples"] for cell in grid.cells})
                self.assertTrue(all(cell.shard_count > 1 for cell in grid.cells))

    def test_tier2_preset_requires_pinned_local_model_and_dataset(self) -> None:
        with self.assertRaisesRegex(GridError, "requires model_id"):
            grid_from_pilot("pilot_c", experiment="e1")
        with self.assertRaisesRegex(ValueError, "unsupported dataset revision"):
            grid_from_pilot(
                "pilot_c",
                experiment="e1",
                model_overrides=TIER2_MODEL,
                dataset_revision_override="not-the-executable-dataset",
            )

    def test_tier2_decoding_seed_override_changes_the_declared_grid(self) -> None:
        grid = grid_from_pilot(
            "pilot_d",
            experiment="e1",
            model_overrides=TIER2_MODEL,
            decoding_seeds_override=(7,),
            dataset_revision_override="silenttwin-tier1-v1",
        )
        self.assertEqual(240, grid.total_tasks)
        self.assertEqual({7}, {cell.configuration["decoding_seed"] for cell in grid.cells})
        self.assertEqual({0.2}, {cell.configuration["temperature"] for cell in grid.cells})
        self.assertEqual({0.95}, {cell.configuration["top_p"] for cell in grid.cells})
        self.assertEqual({1}, {cell.configuration["batch_size"] for cell in grid.cells})

    def test_e2_grid_filters_opaque_termination_target_feedback(self) -> None:
        grid = build_grid(
            experiment="e2",
            tiers=("tier1",),
            world_suites=("email",),
            runtimes=("opaque_termination",),
            attackers=("mock_llm",),
            query_budgets=(0, 4),
            conditions=("no_probe", "adaptive", "shuffled", "oracle"),
            seeds=(42,),
            num_samples=4,
        )
        self.assertEqual(
            {(0, "no_probe"), (0, "oracle"), (4, "shuffled")},
            {
                (cell.configuration["query_budget"], cell.configuration["condition"])
                for cell in grid.cells
            },
        )

    def test_e1_grid_rejects_a_declared_but_unrepresented_feedback_source(self) -> None:
        with self.assertRaisesRegex(GridError, "omit declared feedback sources"):
            build_grid(
                experiment="e1",
                tiers=("tier1",),
                world_suites=("email",),
                runtimes=("generic_failure",),
                attackers=("mock_llm",),
                query_budgets=(0, 4),
                feedback_sources=("genuine", "shuffled"),
                feedback_source_query_budgets={"genuine": (0, 4)},
                seeds=(42,),
                num_samples=4,
            )

    def test_grid_hash_and_manifest_are_deterministic_and_self_validating(self) -> None:
        first = grid_from_pilot("pilot_b", experiment="e1")
        second = grid_from_pilot("pilot_b", experiment="e1")
        self.assertEqual(first.grid_hash, second.grid_hash)
        self.assertEqual(64, len(first.grid_hash))
        self.assertEqual(
            [cell.configuration_hash for cell in first.cells],
            [cell.configuration_hash for cell in second.cells],
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.jsonl"
            write_manifest(first, path)
            loaded = load_grid_manifest(path)
            self.assertEqual(first.grid_hash, loaded["metadata"]["grid_hash"])
            self.assertEqual(first.total_configurations, len(loaded["members"]))

            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[1]["configuration"]["seed"] = 999
            # Even a same-sized file with a plausible new member hash is not
            # accepted unless every dependent identity and overall hash agrees.
            records[1]["configuration_hash"] = stable_hash(records[1]["configuration"])
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaises(GridError):
                load_grid_manifest(path)

    def test_heldout_grid_covers_exact_frozen_public_task_count(self) -> None:
        freeze = make_sample_size_freeze(
            experiment_id="e1",
            dataset_revision="silenttwin-tier1-v1",
            development_manifest_hash="d" * 64,
            contrast_id="e1_ordinary_q16_minus_q0",
            sample_size=3,
            power_estimate=PowerEstimate(
                sample_size=3,
                effect=0.2,
                discordance=0.3,
                alpha=0.05,
                simulations=100,
                seed=7,
                power=0.8,
            ),
        )
        grid = build_grid(
            experiment="e1",
            tiers=("tier1",),
            world_suites=("email",),
            runtimes=("generic_failure",),
            attackers=("bayesian",),
            query_budgets=(16,),
            feedback_sources=("genuine", "shuffled"),
            seeds=(42,),
            num_samples=-1,
            dataset_split="test",
            dataset_revision="silenttwin-tier1-v1",
            episodes_per_shard=4,
            sample_size_freeze=freeze,
        )
        self.assertEqual(5, grid.total_configurations)
        ranges = {
            source: sorted(
                (
                    int(cell.configuration["sample_start"]),
                    int(cell.configuration["num_samples"]),
                )
                for cell in grid.cells
                if cell.configuration["feedback_source"] == source
            )
            for source in ("genuine", "shuffled")
        }
        self.assertEqual([(0, 4), (4, 2)], ranges["genuine"])
        self.assertEqual([(0, 4), (4, 4), (8, 4)], ranges["shuffled"])
        self.assertEqual(
            {freeze["freeze_hash"]},
            {cell.configuration["sample_size_freeze_hash"] for cell in grid.cells},
        )

        with self.assertRaisesRegex(GridError, "different experiment"):
            build_grid(
                experiment="e2",
                tiers=("tier1",),
                world_suites=("email",),
                runtimes=("generic_failure",),
                attackers=("bayesian",),
                query_budgets=(16,),
                conditions=("genuine",),
                seeds=(42,),
                num_samples=-1,
                dataset_split="test",
                dataset_revision="silenttwin-tier1-v1",
                sample_size_freeze=freeze,
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.jsonl"
            write_manifest(grid, path)
            load_grid_manifest(path)

            incomplete = ExperimentGrid(
                experiment=grid.experiment,
                pilot_id=grid.pilot_id,
                factor_order=grid.factor_order,
                cells_per_task=grid.cells_per_task,
                tasks=grid.tasks[:-1],
            )
            write_manifest(incomplete, path)
            with self.assertRaisesRegex(GridError, "incomplete shard identities"):
                load_grid_manifest(path)


if __name__ == "__main__":
    unittest.main()
