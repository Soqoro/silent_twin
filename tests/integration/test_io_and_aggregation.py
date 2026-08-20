from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from silenttwin.config import ExperimentConfig
from silenttwin.experiments.aggregate import aggregate_experiment
from silenttwin.experiments.common import run_experiment
from silenttwin.io.jsonl import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)


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
                {"result.jsonl", "manifest.json", "run.log"},
                {path.name for path in output.iterdir()},
            )
            records = read_jsonl(output / "result.jsonl")
            self.assertEqual(3, len(records))
            self.assertEqual("summary", records[-1]["record_type"])
            self.assertEqual(
                1,
                sum(record["record_type"] == "summary" for record in records),
            )

            second = run_experiment(run_config(output))
            self.assertTrue(second.reused)

    def test_incompatible_existing_output_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_experiment(run_config(output, samples=2))
            with self.assertRaisesRegex(
                ResultValidationError, "incomplete or incompatible"
            ):
                run_experiment(run_config(output, samples=3))

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
            self.assertTrue((root / "aggregate" / "summary.json").is_file())
            self.assertTrue((root / "aggregate" / "summary.csv").is_file())

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


if __name__ == "__main__":
    unittest.main()
