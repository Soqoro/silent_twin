from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPOSITORY / "experiments" / "silenttwin").glob("*.sh"))
E1 = REPOSITORY / "experiments" / "silenttwin" / "run_experiment_1_feedback_leakage.sh"


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
            selected = (
                Path(temporary)
                / "e1/tier=tier1/world=email/runtime=generic_failure"
                / "attacker=bayesian/q=4/seed=43/manifest.json"
            )
            self.assertTrue(selected.is_file())


if __name__ == "__main__":
    unittest.main()
