from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from silenttwin.io.jsonl import atomic_write_json
from silenttwin.metrics.power import PowerEstimate, make_sample_size_freeze


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPOSITORY / "experiments" / "silenttwin").glob("*.sh"))
E1 = REPOSITORY / "experiments" / "silenttwin" / "run_experiment_1_feedback_leakage.sh"
E2 = REPOSITORY / "experiments" / "silenttwin" / "run_experiment_2_feedback_assisted_bypass.sh"
E1_TIER2 = (
    REPOSITORY
    / "experiments"
    / "silenttwin"
    / "run_experiment_1_feedback_leakage_tier2.sh"
)
E2_TIER2 = (
    REPOSITORY
    / "experiments"
    / "silenttwin"
    / "run_experiment_2_feedback_assisted_bypass_tier2.sh"
)


class ShellGridTests(unittest.TestCase):
    def test_all_entrypoints_pass_bash_syntax_check(self) -> None:
        process = subprocess.run(
            ["bash", "-n", *(str(path) for path in SCRIPTS)],
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, process.returncode, process.stderr)

    def test_grid_enumeration_matches_documented_order(self) -> None:
        environment = {
            **os.environ,
            "E1_STAGE": "grid",
            "TIERS": "tier1",
            "WORLD_SUITES": "email",
            "RUNTIMES": "generic_failure",
            "ATTACKERS": "bayesian",
            "QUERY_BUDGETS": "0 4",
            "SEEDS": "42 43",
        }
        process = subprocess.run(
            ["bash", str(E1)],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, process.returncode, process.stderr)
        lines = [line for line in process.stdout.splitlines() if line.startswith("task_id=")]
        self.assertEqual(4, len(lines))
        self.assertIn("task_id=0", lines[0])
        self.assertIn("query_budget=0 seed=42", lines[0])
        self.assertIn("task_id=3", lines[-1])
        self.assertIn("query_budget=4 seed=43", lines[-1])

    def test_every_printed_row_has_exact_hashes_and_repeat_is_identical(self) -> None:
        environment = {
            **os.environ,
            "E1_STAGE": "grid",
            "PILOT_PRESET": "pilot_b",
        }
        first = subprocess.run(
            ["bash", str(E1)],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
        )
        second = subprocess.run(
            ["bash", str(E1)],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertIn("total_tasks=40", first.stdout)
        self.assertIn("total_configurations=160", first.stdout)
        self.assertIn("cells_per_task=4", first.stdout)
        rows = [line for line in first.stdout.splitlines() if line.startswith("task_id=")]
        self.assertEqual(160, len(rows))
        self.assertTrue(all("configuration_hash=" in row for row in rows))
        self.assertTrue(all("grid_hash=" in row for row in rows))
        self.assertTrue(all("shard_id=" in row for row in rows))
        self.assertEqual({"0", "1", "2", "3"}, {
            next(part.split("=", 1)[1] for part in row.split() if part.startswith("batch_offset="))
            for row in rows
        })
        self.assertIn("pair_family=authorization", first.stdout)
        self.assertIn("pair_family=monitor_blind_spot", first.stdout)
        self.assertIn("query_budget=16", first.stdout)
        self.assertIn("feedback_source=shuffled", first.stdout)

    def test_e2_pilot_b_contains_only_declared_valid_condition_budget_pairs(self) -> None:
        process = subprocess.run(
            ["bash", str(E2)],
            cwd=REPOSITORY,
            env={**os.environ, "E2_STAGE": "grid", "PILOT_PRESET": "pilot_b"},
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, process.returncode, process.stderr)
        self.assertIn("total_tasks=32", process.stdout)
        self.assertIn("total_configurations=128", process.stdout)
        rows = [line for line in process.stdout.splitlines() if line.startswith("task_id=")]
        pairs = set()
        for row in rows:
            fields = dict(part.split("=", 1) for part in row.split() if "=" in part)
            pairs.add((int(fields["query_budget"]), fields["condition"]))
        self.assertEqual(
            {
                (0, "no_probe"),
                (0, "oracle"),
                (4, "genuine"),
                (4, "shuffled"),
                (16, "genuine"),
                (16, "shuffled"),
                (32, "genuine"),
                (32, "shuffled"),
            },
            pairs,
        )

    def test_out_of_range_array_index_fails_before_python(self) -> None:
        environment = {
            **os.environ,
            "TIERS": "tier1",
            "WORLD_SUITES": "email",
            "RUNTIMES": "silenttwin",
            "ATTACKERS": "bayesian",
            "QUERY_BUDGETS": "0",
            "SEEDS": "42",
            "SLURM_ARRAY_TASK_ID": "1",
        }
        process = subprocess.run(
            ["bash", str(E1)],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(0, process.returncode)
        self.assertIn("out of range", process.stderr)

    def test_unknown_grid_choice_is_rejected(self) -> None:
        environment = {
            **os.environ,
            "E1_STAGE": "grid",
            "RUNTIMES": "not_a_runtime",
        }
        process = subprocess.run(
            ["bash", str(E1)],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(0, process.returncode)
        self.assertIn("unsupported value", process.stderr)

    def test_nonzero_array_selection_matches_grid_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                **os.environ,
                "PYTHON_BIN": sys.executable,
                "OUT_ROOT": temporary,
                "NUM_SAMPLES": "2",
                "TIERS": "tier1",
                "WORLD_SUITES": "email",
                "RUNTIMES": "generic_failure",
                "ATTACKERS": "bayesian",
                "QUERY_BUDGETS": "0 4",
                "SEEDS": "42 43",
                "SLURM_ARRAY_TASK_ID": "3",
            }
            process = subprocess.run(
                ["bash", str(E1)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, process.returncode, process.stderr)
            self.assertIn("selected_task_id=3", process.stdout)
            selected = list((Path(temporary) / "e1/pilot=custom").rglob("manifest.json"))
            self.assertEqual(1, len(selected))

    def test_e2_minimal_smoke_uses_one_complete_four_row_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            process = subprocess.run(
                ["bash", str(E2)],
                cwd=REPOSITORY,
                env={
                    **os.environ,
                    "PYTHON_BIN": sys.executable,
                    "OUT_ROOT": temporary,
                    "NUM_SAMPLES": "4",
                    "TIERS": "tier1",
                    "WORLD_SUITES": "email",
                    "RUNTIMES": "generic_failure",
                    "ATTACKERS": "mock_llm",
                    "QUERY_BUDGETS": "4",
                    "E2_CONDITIONS": "adaptive",
                    "SEEDS": "42",
                    "SLURM_ARRAY_TASK_ID": "0",
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, process.returncode, process.stderr)
            manifests = list((Path(temporary) / "e2/pilot=custom").rglob("manifest.json"))
            self.assertEqual(1, len(manifests))

    def test_run_stage_executes_every_cell_in_a_small_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            process = subprocess.run(
                ["bash", str(E1)],
                cwd=REPOSITORY,
                env={
                    **os.environ,
                    "PYTHON_BIN": sys.executable,
                    "OUT_ROOT": temporary,
                    "NUM_SAMPLES": "2",
                    "TIERS": "tier1",
                    "WORLD_SUITES": "email",
                    "RUNTIMES": "generic_failure",
                    "ATTACKERS": "mock_llm",
                    "QUERY_BUDGETS": "0 4",
                    "SEEDS": "42",
                    "CELLS_PER_TASK": "2",
                    "SLURM_ARRAY_TASK_ID": "0",
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, process.returncode, process.stderr)
            self.assertIn("total_tasks=1", process.stdout)
            self.assertIn("total_configurations=2", process.stdout)
            self.assertEqual(2, len(list((Path(temporary) / "e1").rglob("manifest.json"))))

    def test_tier2_grid_inspection_is_model_free_and_rows_are_sharded(self) -> None:
        common = {
            **os.environ,
            "PILOT_PRESET": "pilot_c",
            "MODEL_ID": "/approved/models/example",
            "MODEL_REVISION": "a" * 40,
            "MODEL_CACHE_DIR": "/persistent/cache/silenttwin",
            "DATASET_REVISION": "silenttwin-tier1-v1",
        }
        for script, stage_name, expected_tasks in (
            (E1_TIER2, "E1_STAGE", 20),
            (E2_TIER2, "E2_STAGE", 48),
        ):
            with self.subTest(script=script.name):
                process = subprocess.run(
                    ["bash", str(script)],
                    cwd=REPOSITORY,
                    env={**common, stage_name: "grid"},
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(0, process.returncode, process.stderr)
                self.assertIn(f"total_tasks={expected_tasks}", process.stdout)
                rows = [
                    line for line in process.stdout.splitlines() if line.startswith("task_id=")
                ]
                self.assertEqual(expected_tasks, len(rows))
                self.assertTrue(all("num_samples=20" in row for row in rows))
                self.assertTrue(all("configuration_hash=" in row for row in rows))

    def test_frozen_heldout_grid_is_reachable_through_tier1_and_tier2_bash(self) -> None:
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
                seed=3,
                power=0.8,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            freeze_path = Path(temporary) / "freeze.json"
            atomic_write_json(freeze_path, freeze)
            shared = {
                **os.environ,
                "E1_STAGE": "grid",
                "WORLD_SUITES": "email",
                "RUNTIMES": "generic_failure",
                "QUERY_BUDGETS": "16",
                "FEEDBACK_SOURCES": "genuine shuffled",
                "SEEDS": "42",
                "PAIR_FAMILIES": "monitor_blind_spot",
                "DATASET_SPLIT": "test",
                "DATASET_REVISION": "silenttwin-tier1-v1",
                "SAMPLE_SIZE_FREEZE": str(freeze_path),
                "EPISODES_PER_SHARD": "4",
            }
            tier1 = subprocess.run(
                ["bash", str(E1)],
                cwd=REPOSITORY,
                env={**shared, "TIERS": "tier1", "ATTACKERS": "bayesian"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, tier1.returncode, tier1.stderr)
            self.assertIn("total_configurations=5", tier1.stdout)
            self.assertIn(f"sample_size_freeze_hash={freeze['freeze_hash']}", tier1.stdout)

            tier2 = subprocess.run(
                ["bash", str(E1_TIER2)],
                cwd=REPOSITORY,
                env={
                    **shared,
                    "PILOT_PRESET": "",
                    "FEEDBACK_SOURCES": "genuine",
                    "EPISODES_PER_SHARD": "20",
                    "MODEL_ID": "/approved/models/example",
                    "MODEL_REVISION": "a" * 40,
                    "MODEL_CACHE_DIR": "/persistent/cache/silenttwin",
                    "DECODING_SEEDS": "0",
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, tier2.returncode, tier2.stderr)
            self.assertIn("total_configurations=1", tier2.stdout)
            self.assertIn("tier=tier2", tier2.stdout)

    def test_tier2_run_is_rejected_outside_slurm_before_model_loading(self) -> None:
        process = subprocess.run(
            ["bash", str(E1_TIER2)],
            cwd=REPOSITORY,
            env={
                **os.environ,
                "E1_STAGE": "run",
                "PILOT_PRESET": "pilot_c",
                "MODEL_ID": "/path/that/does/not/exist",
                "MODEL_REVISION": "a" * 40,
                "MODEL_CACHE_DIR": "/persistent/cache/silenttwin",
                "DATASET_REVISION": "silenttwin-tier1-v1",
            },
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(0, process.returncode)
        self.assertIn("forbidden outside an authorized SLURM job", process.stderr)

    def test_tier2_rejects_ephemeral_model_cache_before_gpu_probe(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            process = subprocess.run(
                ["bash", str(E1_TIER2)],
                cwd=REPOSITORY,
                env={
                    **os.environ,
                    "E1_STAGE": "run",
                    "PILOT_PRESET": "pilot_c",
                    "MODEL_ID": "/approved/models/example",
                    "MODEL_REVISION": "a" * 40,
                    "MODEL_CACHE_DIR": str(Path(scratch) / "model-cache"),
                    "DATASET_REVISION": "silenttwin-tier1-v1",
                    "SLURM_JOB_ID": "test-job",
                    "SLURM_TMPDIR": scratch,
                },
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(0, process.returncode)
        self.assertIn("MODEL_CACHE_DIR must be persistent", process.stderr)

    def test_tier2_scripts_do_not_guess_cluster_gpu_flags(self) -> None:
        for script in (E1_TIER2, E2_TIER2):
            with self.subTest(script=script.name):
                directives = [
                    line.strip()
                    for line in script.read_text().splitlines()
                    if line.startswith("#SBATCH")
                ]
                joined = "\n".join(directives)
                self.assertNotIn("--account", joined)
                self.assertNotIn("--partition", joined)
                self.assertNotIn("--gres", joined)
                self.assertNotIn("--gpus", joined)
                self.assertFalse(any(line.startswith("#SBATCH -p") for line in directives))


if __name__ == "__main__":
    unittest.main()
