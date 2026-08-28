"""CPU-only commands for freezing and checking AgentDojo release artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .config import AGENTDOJO_BENCHMARK_VERSION, AGENTDOJO_SOURCE_REVISION


DEFAULT_CATALOG_PATH = Path("configs/silenttwin/agentdojo/catalog-v1.json")
DEFAULT_SPLITS_PATH = Path("configs/silenttwin/agentdojo/splits-v1.json")
DEFAULT_ACTION_ELIGIBILITY_PATH = Path(
    "configs/silenttwin/agentdojo/action-eligibility-v1.json"
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    """Install an immutable JSON object; refuse a conflicting collision.

    Returns ``True`` only when an identical pre-existing freeze was reused.
    A hard-link install makes the fully fsynced temporary inode visible in one
    operation without ever replacing an earlier scientific freeze.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_object(path) == dict(value):
            return True
        raise ValueError(f"refusing to overwrite conflicting frozen artifact {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if _read_object(path) == dict(value):
                return True
            raise ValueError(
                f"refusing to overwrite conflicting frozen artifact {path}"
            )
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return False
    finally:
        temporary_path.unlink(missing_ok=True)


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    # These imports are deliberately below argument parsing so `--help` works
    # in the ordinary AgentDojo-free Python environment.
    from .catalog import build_catalog, catalog_summary
    from .splits import build_split_manifest

    catalog = build_catalog(
        deployment_source_revision=args.source_revision,
        benchmark_version=args.benchmark_version,
    )
    splits = build_split_manifest(catalog)
    _atomic_write(args.catalog_output, catalog)
    _atomic_write(args.splits_output, splits)
    return {
        **catalog_summary(catalog),
        "split_manifest_hash": splits["split_manifest_hash"],
        "catalog_path": str(args.catalog_output),
        "splits_path": str(args.splits_output),
    }


def _freeze_action_eligibility(args: argparse.Namespace) -> dict[str, Any]:
    from .action_eligibility import make_action_eligibility_manifest

    if args.assert_no_learned_outcomes_inspected is not True:
        raise ValueError(
            "freezing action eligibility requires the no-learned-outcomes assertion"
        )
    catalog = _read_object(args.catalog)
    splits = _read_object(args.splits)
    manifest = make_action_eligibility_manifest(
        catalog=catalog,
        split_manifest=splits,
    )
    reused = _atomic_write_immutable(args.output, manifest)
    pilot = manifest["pilot_scenario_ids_by_split"]
    return {
        "action_eligibility_manifest_hash": manifest[
            "action_eligibility_manifest_hash"
        ],
        "protocol_disposition": manifest["protocol_disposition"],
        "train_scenario_count": len(pilot["train"]),
        "development_scenario_count": len(pilot["development"]),
        "test_scenario_count": len(pilot["test"]),
        "held_out_evaluation_permitted": False,
        "output": str(args.output),
        "reused_existing_freeze": reused,
    }


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    from .catalog import catalog_summary, validate_catalog
    from .splits import validate_split_manifest

    catalog = _read_object(args.catalog)
    splits = _read_object(args.splits)
    validate_catalog(catalog)
    validate_split_manifest(splits, catalog=catalog)
    return {
        **catalog_summary(catalog),
        "split_manifest_hash": splits["split_manifest_hash"],
        "catalog_path": str(args.catalog),
        "splits_path": str(args.splits),
        "verified": True,
    }


def _freeze_sample_size(args: argparse.Namespace) -> dict[str, Any]:
    from .action_eligibility import ESTIMATION_ONLY_DISPOSITION
    from .config import AGENTDOJO_SUITES, bundle_hash
    from .freeze import (
        deterministic_test_allocation,
        make_agentdojo_sample_size_freeze,
        validate_development_analysis_manifest,
    )
    from .grid import load_frozen_inputs, validate_structural_splits

    inputs = load_frozen_inputs(
        catalog_path=args.catalog,
        splits_path=args.splits,
        strategy_catalog_path=args.strategy_catalog,
        pair_registry_path=args.pair_registry,
        analysis_plan_path=args.analysis_plan,
        dependency_lock_path=args.dependency_lock,
    )
    for label, artifact in (
        ("candidate-strategy catalog", inputs.strategy_catalog),
        ("pair registry", inputs.pair_registry),
    ):
        if artifact.get("scientific_evidence_eligible") is False or artifact.get(
            "artifact_class"
        ) == "deterministic_fake_smoke_fixture":
            raise ValueError(
                f"{label} is an engineering-smoke fixture and cannot freeze held-out evidence"
            )
    if (
        inputs.pair_registry.get("protocol_disposition")
        == ESTIMATION_ONLY_DISPOSITION
    ):
        raise ValueError(
            "the action-representable pilot is estimation-only and cannot freeze "
            "held-out evidence"
        )
    manifest = _read_object(args.development_analysis_manifest)
    primary = inputs.analysis_plan.get("primary_contrasts", {}).get(args.experiment)
    if not isinstance(primary, str):
        raise ValueError("analysis plan lacks the selected primary contrast")
    development = validate_development_analysis_manifest(
        manifest,
        experiment_id=args.experiment,
        primary_contrast_id=primary,
        upstream=inputs.upstream,
    )
    recorded_manifest_hash = development["analysis_manifest_hash"]
    development_evidence_hash = development["development_evidence_hash"]
    power = development["development_power_analysis"]
    test_groups = set(validate_structural_splits(inputs)["test"])
    available_ids = {
        suite: sorted(
            {
                str(row["structural_group_id"])
                for row in inputs.scenarios
                if row["suite"] == suite
                and str(row["structural_group_id"]) in test_groups
            }
        )
        for suite in AGENTDOJO_SUITES
    }
    available_counts = {
        suite: len(available_ids[suite]) for suite in AGENTDOJO_SUITES
    }
    if any(count <= 0 for count in available_counts.values()):
        raise ValueError("held-out catalog omits a required suite stratum")
    spec = power["power_analysis_spec"]
    minimum_per_suite = int(spec["minimum_structural_groups_per_suite"])
    recommended = power["required_sample_size"]["selected_sample_size"]
    available_total = sum(available_counts.values())
    confirmatory_feasible = (
        isinstance(recommended, int)
        and not isinstance(recommended, bool)
        and recommended >= minimum_per_suite * len(AGENTDOJO_SUITES)
        and recommended <= available_total
        and all(count >= minimum_per_suite for count in available_counts.values())
    )
    selected_total = int(recommended) if confirmatory_feasible else available_total
    selected_counts = deterministic_test_allocation(
        available_counts, requested_total=selected_total
    )
    selected_ids = {
        suite: available_ids[suite][: selected_counts[suite]]
        for suite in AGENTDOJO_SUITES
    }
    bundle_hashes: dict[str, str] = {}
    for suite in AGENTDOJO_SUITES:
        selected_set = set(selected_ids[suite])
        scenario_ids = sorted(
            str(row["scenario_id"])
            for row in inputs.scenarios
            if row["suite"] == suite
            and row["dataset_split"] == "test"
            and str(row["structural_group_id"]) in selected_set
        )
        bundle_hashes[suite] = bundle_hash(
            suite=suite,
            dataset_split="test",
            scenario_ids=scenario_ids,
            structural_group_ids=selected_ids[suite],
        )
    frozen = make_agentdojo_sample_size_freeze(
        experiment_id=args.experiment,
        primary_contrast_id=primary,
        upstream=inputs.upstream,
        development_analysis_manifest_hash=str(recorded_manifest_hash),
        development_evidence_hash=development_evidence_hash,
        independent_unit_count_by_suite=selected_counts,
        available_test_independent_unit_count_by_suite=available_counts,
        selected_test_bundle_hash_by_suite=bundle_hashes,
        selected_structural_group_ids_by_suite=selected_ids,
        power_evidence=power,
    )
    reused = _atomic_write_immutable(args.output, frozen)
    return {
        "freeze_path": str(args.output),
        "freeze_hash": frozen["freeze_hash"],
        "experiment_id": args.experiment,
        "primary_contrast_id": primary,
        "claim_disposition": frozen["claim_disposition"],
        "selected_total_independent_unit_count": frozen[
            "selected_total_independent_unit_count"
        ],
        "structural_minimum_shortfalls": frozen[
            "structural_minimum_shortfalls"
        ],
        "reused_identical_freeze": reused,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silenttwin.agentdojo.cli",
        description="Freeze or verify pinned AgentDojo Tier-2 CPU artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze-catalog",
        aliases=["catalog"],
        help="introspect all four suites and atomically freeze catalog + splits",
    )
    freeze.add_argument(
        "--source-revision",
        required=True,
        help=f"deployment assertion; must equal {AGENTDOJO_SOURCE_REVISION}",
    )
    freeze.add_argument(
        "--benchmark-version",
        default=AGENTDOJO_BENCHMARK_VERSION,
        help="released benchmark-data version (default: %(default)s)",
    )
    freeze.add_argument(
        "--catalog-output",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="catalog JSON destination (default: %(default)s)",
    )
    freeze.add_argument(
        "--splits-output",
        type=Path,
        default=DEFAULT_SPLITS_PATH,
        help="split manifest JSON destination (default: %(default)s)",
    )
    freeze.set_defaults(handler=_freeze)

    verify = commands.add_parser(
        "verify-catalog", help="validate frozen hashes and structural isolation"
    )
    verify.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    verify.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    verify.set_defaults(handler=_verify)

    eligibility = commands.add_parser(
        "freeze-action-eligibility",
        help=(
            "atomically freeze the conservative train/development "
            "action-representable estimation subset"
        ),
    )
    eligibility.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    eligibility.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    eligibility.add_argument(
        "--output", type=Path, default=DEFAULT_ACTION_ELIGIBILITY_PATH
    )
    eligibility.add_argument(
        "--assert-no-learned-outcomes-inspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that no learned attacker, monitor, or held-out "
            "model outcome informed the eligibility freeze"
        ),
    )
    eligibility.set_defaults(handler=_freeze_action_eligibility)

    sample = commands.add_parser(
        "freeze-sample-size",
        help="atomically freeze a deterministic held-out E1-E4 cohort from development power",
    )
    sample.add_argument("--experiment", choices=("e1", "e2", "e3", "e4"), required=True)
    sample.add_argument("--catalog", type=Path, required=True)
    sample.add_argument("--splits", type=Path, required=True)
    sample.add_argument("--strategy-catalog", type=Path, required=True)
    sample.add_argument("--pair-registry", type=Path, required=True)
    sample.add_argument("--analysis-plan", type=Path, required=True)
    sample.add_argument("--dependency-lock", type=Path, required=True)
    sample.add_argument("--development-analysis-manifest", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument(
        "--assert-test-results-uninspected",
        action="store_true",
        required=True,
        help="required operator assertion that no held-out outcome has been inspected",
    )
    sample.set_defaults(handler=_freeze_sample_size)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = args.handler(args)
    except Exception as error:
        parser.exit(2, f"AgentDojo artifact error: {type(error).__name__}: {error}\n")
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module command
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ACTION_ELIGIBILITY_PATH",
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_SPLITS_PATH",
    "main",
]
