from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY / "experiments" / "silenttwin"

CASES = (
    (
        "e1",
        "run_experiment_1_feedback_leakage.sh",
        "E1_STAGE",
        {"RUNTIMES": "generic_failure", "QUERY_BUDGETS": "4"},
    ),
    (
        "e2",
        "run_experiment_2_feedback_assisted_bypass.sh",
        "E2_STAGE",
        {
            "RUNTIMES": "generic_failure",
            "QUERY_BUDGETS": "4",
            "E2_CONDITIONS": "adaptive",
        },
    ),
    (
        "e3",
        "run_experiment_3_channel_closure.sh",
        "E3_STAGE",
        {"RUNTIMES": "silenttwin", "QUERY_BUDGETS": "4"},
    ),
    (
        "e4",
        "run_experiment_4_useful_work.sh",
        "E4_STAGE",
        {
            "RUNTIMES": "silenttwin",
            "QUERY_BUDGETS": "0",
            "E4_WORKFLOWS": "independent",
        },
    ),
    (
        "e5",
        "run_experiment_5_assumption_ablations.sh",
        "E5_STAGE",
        {
            "RUNTIMES": "silenttwin",
            "QUERY_BUDGETS": "4",
            "E5_ABLATIONS": "timing_leak",
        },
    ),
)


class EntrypointSmokeTests(unittest.TestCase):
    def test_five_four_sample_runs_and_aggregators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "smoke"
            common = {
                **os.environ,
                "PYTHON_BIN": sys.executable,
                "OUT_ROOT": str(root),
                # E2 uses a fully crossed target/donor assignment, so every
                # smoke shard must contain a complete four-row block.
                "NUM_SAMPLES": "4",
                "TIERS": "tier1",
                "WORLD_SUITES": "email",
                "ATTACKERS": "mock_llm",
                "SEEDS": "42",
            }
            for experiment, script, stage_variable, overrides in CASES:
                with self.subTest(experiment=experiment, stage="run"):
                    environment = {
                        **common,
                        **overrides,
                        stage_variable: "run",
                        "SLURM_ARRAY_TASK_ID": "0",
                    }
                    process = subprocess.run(
                        ["bash", str(SCRIPT_ROOT / script)],
                        cwd=REPOSITORY,
                        env=environment,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(0, process.returncode, process.stderr)

                with self.subTest(experiment=experiment, stage="aggregate"):
                    environment = {
                        **common,
                        **overrides,
                        stage_variable: "aggregate",
                    }
                    process = subprocess.run(
                        ["bash", str(SCRIPT_ROOT / script)],
                        cwd=REPOSITORY,
                        env=environment,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(0, process.returncode, process.stderr)
                    summary_path = root / experiment / "aggregate" / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    self.assertEqual(1, summary["run_count"])
                    self.assertEqual(4, summary["total_sample_count"])


if __name__ == "__main__":
    unittest.main()
