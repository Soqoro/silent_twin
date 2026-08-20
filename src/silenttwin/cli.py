"""Internal command-line interface used by all human-facing shell scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from silenttwin.config import (
    ATTACKERS,
    EXPERIMENTS,
    RUNTIMES,
    TIERS,
    WORLD_SUITES,
    ExperimentConfig,
    canonical_json,
)
from silenttwin.experiments.aggregate import aggregate_experiment
from silenttwin.experiments.common import run_experiment
from silenttwin.io.jsonl import ResultValidationError
from silenttwin.io.manifests import validate_result_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silenttwin.cli",
        description="Run, validate, and aggregate SilentTwin experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run or reuse one finite-state configuration")
    run.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    run.add_argument("--tier", default="tier1", choices=TIERS)
    run.add_argument("--world-suite", default="email", choices=WORLD_SUITES)
    run.add_argument("--runtime", default="silenttwin", choices=RUNTIMES)
    run.add_argument("--attacker", default="bayesian", choices=ATTACKERS)
    run.add_argument("--query-budget", type=int, default=0)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="number of samples; -1 selects the deterministic Tier-1 default",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--condition", default=None, help="E2: adaptive/random/oracle/shuffled")
    run.add_argument(
        "--workflow", default=None, help="E4: independent/rejection_dependent/atomic"
    )
    run.add_argument("--ablation", default=None, help="E5 assumption removed from SilentTwin")
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing result, including a completed compatible result",
    )

    aggregate = subparsers.add_parser(
        "aggregate", help="strictly aggregate complete matched-task runs"
    )
    aggregate.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    aggregate.add_argument("--input-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="fail unless exactly this many complete, unique configurations are present",
    )

    validate = subparsers.add_parser(
        "validate-result", help="validate a result, manifest, digest, and provenance"
    )
    validate.add_argument("--result-dir", type=Path, required=True)
    validate.add_argument("--experiment", choices=EXPERIMENTS)
    validate.add_argument(
        "--allow-historical-provenance",
        action="store_true",
        help="validate internal provenance without requiring it to match current source",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    config = ExperimentConfig(
        experiment=args.experiment,
        tier=args.tier,
        world_suite=args.world_suite,
        runtime=args.runtime,
        attacker=args.attacker,
        query_budget=args.query_budget,
        seed=args.seed,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        condition=args.condition,
        workflow=args.workflow,
        ablation=args.ablation,
        overwrite=args.overwrite,
    )
    outcome = run_experiment(config)
    print(
        canonical_json(
            {
                "status": "reused" if outcome.reused else "completed",
                "experiment_id": config.experiment,
                "configuration_hash": outcome.configuration_hash,
                "sample_count": outcome.sample_count,
                "output_dir": str(outcome.output_dir),
            }
        )
    )
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    aggregate = aggregate_experiment(
        experiment=args.experiment,
        input_root=args.input_root,
        output_dir=args.output_dir,
        expected_runs=args.expected_runs,
    )
    print(
        canonical_json(
            {
                "status": "completed",
                "experiment_id": args.experiment,
                "run_count": aggregate["run_count"],
                "total_sample_count": aggregate["total_sample_count"],
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    manifest = validate_result_directory(
        args.result_dir,
        expected_experiment=args.experiment,
        require_current_provenance=not args.allow_historical_provenance,
    )
    print(
        canonical_json(
            {
                "status": "valid",
                "experiment_id": manifest["experiment_id"],
                "configuration_hash": manifest["configuration_hash"],
                "sample_count": manifest["actual_sample_count"],
                "result_dir": str(args.result_dir),
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "aggregate":
            return _aggregate(args)
        if args.command == "validate-result":
            return _validate(args)
    except (ResultValidationError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
