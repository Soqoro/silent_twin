"""Internal command-line interface used by all human-facing shell scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from silenttwin.config import (
    ATTACKERS,
    DATASET_SPLITS,
    EXPERIMENTS,
    FEEDBACK_SOURCES,
    PAIR_FAMILIES,
    RUNTIMES,
    TIERS,
    WORLD_SUITES,
    ExperimentConfig,
    canonical_json,
    stable_hash,
)
from silenttwin.experiments.aggregate import aggregate_experiment
from silenttwin.experiments.common import run_experiment
from silenttwin.io.jsonl import ResultValidationError
from silenttwin.io.jsonl import atomic_write_json
from silenttwin.io.manifests import validate_result_directory
from silenttwin.metrics.power import (
    PowerEstimate,
    make_sample_size_freeze,
    validate_sample_size_freeze,
)


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
    run.add_argument("--sample-start", type=int, default=0)
    run.add_argument(
        "--pair-family",
        choices=(*PAIR_FAMILIES, "blind_spot"),
        default="monitor_blind_spot",
    )
    run.add_argument("--template-id", default=None)
    run.add_argument("--dataset-split", choices=DATASET_SPLITS, default="development")
    run.add_argument("--dataset-revision", default="silenttwin-tier1-v1")
    run.add_argument("--analysis-revision", default="silenttwin-analysis-v1")
    run.add_argument("--feedback-source", choices=FEEDBACK_SOURCES, default="genuine")
    run.add_argument(
        "--condition",
        default=None,
        help="E2: no_probe/genuine (adaptive alias)/shuffled/random/oracle",
    )
    run.add_argument(
        "--workflow", default=None, help="E4: independent/rejection_dependent/atomic"
    )
    run.add_argument("--ablation", default=None, help="E5 assumption removed from SilentTwin")
    run.add_argument("--confidence-threshold", type=float, default=0.9)
    run.add_argument("--model-id", default=None)
    run.add_argument("--model-revision", default=None)
    run.add_argument("--model-cache-dir", default=None)
    run.add_argument("--dtype", default="auto")
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--decoding-seed", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--grid-hash", default=None)
    run.add_argument("--grid-task-id", type=int, default=None)
    run.add_argument("--shard-id", default=None)
    run.add_argument("--pilot-id", default=None)
    run.add_argument(
        "--sample-size-freeze",
        type=Path,
        default=None,
        help="required hash-bound freeze JSON before any held-out test run",
    )
    run.add_argument(
        "--expected-configuration-hash",
        default=None,
        help="fail before execution unless the selected scientific configuration has this hash",
    )
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
    aggregate.add_argument(
        "--expected-grid-manifest",
        type=Path,
        default=None,
        help="authoritative silenttwin.grid.v1 JSONL manifest",
    )
    aggregate.add_argument(
        "--expected-grid-hash",
        default=None,
        help="optional explicit hash that must match the expected grid manifest",
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

    fingerprint = subparsers.add_parser(
        "fingerprint-model",
        help="compute the immutable sha256 revision for a local checkpoint directory",
    )
    fingerprint.add_argument("--model-dir", type=Path, required=True)
    fingerprint.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="write a reusable fingerprint manifest below this persistent cache",
    )

    freeze = subparsers.add_parser(
        "freeze-sample-size",
        help="freeze a Pilot-D development power recommendation before test inspection",
    )
    freeze.add_argument("--analysis-manifest", type=Path, required=True)
    freeze.add_argument("--output-file", type=Path, required=True)
    return parser


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _validated_freeze_fields(args: argparse.Namespace) -> dict[str, object | None]:
    if args.dataset_split != "test":
        if args.sample_size_freeze is not None:
            raise ValueError("--sample-size-freeze is valid only with --dataset-split test")
        return {
            "sample_size_freeze_hash": None,
            "development_manifest_hash": None,
            "frozen_public_instances": None,
            "primary_contrast_id": None,
        }
    if args.sample_size_freeze is None:
        raise ValueError("--sample-size-freeze is required with --dataset-split test")
    freeze = _read_json_object(args.sample_size_freeze, "sample-size freeze")
    contrast_id = freeze.get("contrast_id")
    development_hash = freeze.get("development_manifest_hash")
    if not isinstance(contrast_id, str) or not isinstance(development_hash, str):
        raise ValueError("sample-size freeze lacks contrast/development identifiers")
    sample_size = validate_sample_size_freeze(
        freeze,
        experiment_id=args.experiment,
        dataset_revision=args.dataset_revision,
        contrast_id=contrast_id,
        development_manifest_hash=development_hash,
    )
    freeze_hash = freeze.get("freeze_hash")
    if not isinstance(freeze_hash, str):
        raise ValueError("sample-size freeze lacks its validated hash")
    return {
        "sample_size_freeze_hash": freeze_hash,
        "development_manifest_hash": development_hash,
        "frozen_public_instances": sample_size,
        "primary_contrast_id": contrast_id,
    }


def _run(args: argparse.Namespace) -> int:
    freeze_fields = _validated_freeze_fields(args)
    config = ExperimentConfig(
        experiment=args.experiment,
        tier=args.tier,
        world_suite=args.world_suite,
        runtime=args.runtime,
        attacker=args.attacker,
        query_budget=args.query_budget,
        seed=args.seed,
        num_samples=args.num_samples,
        sample_start=args.sample_start,
        output_dir=args.output_dir,
        pair_family=args.pair_family,
        template_id=args.template_id,
        dataset_split=args.dataset_split,
        dataset_revision=args.dataset_revision,
        analysis_revision=args.analysis_revision,
        feedback_source=args.feedback_source,
        condition=args.condition,
        workflow=args.workflow,
        ablation=args.ablation,
        confidence_threshold=args.confidence_threshold,
        model_id=args.model_id,
        model_revision=args.model_revision,
        model_cache_dir=args.model_cache_dir,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        decoding_seed=args.decoding_seed,
        batch_size=args.batch_size,
        grid_hash=args.grid_hash,
        grid_task_id=args.grid_task_id,
        shard_id=args.shard_id,
        pilot_id=args.pilot_id,
        **freeze_fields,
        overwrite=args.overwrite,
    )
    if (
        args.expected_configuration_hash is not None
        and args.expected_configuration_hash != config.configuration_hash
    ):
        raise ValueError(
            "selected configuration hash does not match --expected-configuration-hash: "
            f"expected {args.expected_configuration_hash}, got {config.configuration_hash}"
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
        expected_grid_manifest=args.expected_grid_manifest,
        expected_grid_hash=args.expected_grid_hash,
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


def _fingerprint_model(args: argparse.Namespace) -> int:
    from silenttwin.model_clients import (
        fingerprint_local_checkpoint,
        prepare_local_checkpoint_fingerprint,
    )

    prepared = (
        prepare_local_checkpoint_fingerprint(args.model_dir, args.cache_dir)
        if args.cache_dir is not None
        else {
            "model_revision": fingerprint_local_checkpoint(args.model_dir),
            "manifest_path": None,
            "manifest_hash": None,
        }
    )
    print(
        canonical_json(
            {
                "model_dir": str(args.model_dir.resolve()),
                **prepared,
            }
        )
    )
    return 0


def _freeze_sample_size(args: argparse.Namespace) -> int:
    manifest = _read_json_object(args.analysis_manifest, "analysis manifest")
    if manifest.get("schema_version") != "silenttwin.analysis-manifest.v1":
        raise ValueError("unsupported analysis manifest schema")
    if manifest.get("dataset_split") != "development":
        raise ValueError("sample size can be frozen only from development evidence")
    experiment_id = manifest.get("experiment_id")
    if experiment_id not in {"e1", "e2"}:
        raise ValueError("analysis manifest lacks a supported E1/E2 experiment identity")
    dataset_revision = manifest.get("dataset_revision")
    if not isinstance(dataset_revision, str):
        raise ValueError("analysis manifest lacks one dataset revision")
    power = manifest.get("development_power_analysis")
    if not isinstance(power, dict) or power.get("status") != "estimated_not_frozen":
        raise ValueError("analysis manifest has no completed development power estimate")
    simulation = power.get("simulation_power")
    if not isinstance(simulation, dict):
        raise ValueError("development power estimate lacks simulation results")
    selected = simulation.get("selected_sample_size")
    estimates = simulation.get("estimates")
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise ValueError("development power did not reach the target within candidates")
    if not isinstance(estimates, list):
        raise ValueError("development power estimate has no candidate estimates")
    matches = [
        row
        for row in estimates
        if isinstance(row, dict) and row.get("sample_size") == selected
    ]
    if len(matches) != 1:
        raise ValueError("selected power estimate is absent or ambiguous")
    row = matches[0]
    estimate = PowerEstimate(
        sample_size=selected,
        effect=float(row["effect"]),
        discordance=float(row["discordance"]),
        alpha=float(row["alpha"]),
        simulations=int(row["simulations"]),
        seed=int(row["seed"]),
        power=float(row["power"]),
    )
    strata = power.get("strata")
    contrast_ids = {
        str(item.get("contrast"))
        for item in strata
        if isinstance(item, dict) and item.get("contrast")
    } if isinstance(strata, list) else set()
    if len(contrast_ids) != 1:
        raise ValueError("development power evidence does not identify one primary contrast")
    contrast_id = next(iter(contrast_ids))
    development_manifest_hash = stable_hash(manifest)
    freeze = make_sample_size_freeze(
        experiment_id=str(experiment_id),
        dataset_revision=dataset_revision,
        development_manifest_hash=development_manifest_hash,
        contrast_id=contrast_id,
        sample_size=selected,
        power_estimate=estimate,
    )
    if args.output_file.exists():
        existing = _read_json_object(args.output_file, "existing sample-size freeze")
        if existing != freeze:
            raise ValueError(
                "refusing to replace a different sample-size freeze; choose a new output path"
            )
    else:
        atomic_write_json(args.output_file, freeze)
    print(
        canonical_json(
            {
                "status": "frozen",
                "sample_size": selected,
                "contrast_id": contrast_id,
                "development_manifest_hash": development_manifest_hash,
                "freeze_hash": freeze["freeze_hash"],
                "output_file": str(args.output_file),
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
        if args.command == "fingerprint-model":
            return _fingerprint_model(args)
        if args.command == "freeze-sample-size":
            return _freeze_sample_size(args)
    except (ResultValidationError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
