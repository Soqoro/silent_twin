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
DEFAULT_RECIPIENT_SEPARATION_PROTOCOL_PATH = Path(
    "configs/silenttwin/agentdojo/scientific-v6-recipient-separation-protocol-v1.json"
)
DEFAULT_RECIPIENT_SEPARATION_ANALYSIS_PATH = Path(
    "configs/silenttwin/agentdojo/analysis/recipient-separation-v1.json"
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


def _clean_source_tree_hash() -> str:
    """Return the executable source identity only for a clean Git checkpoint."""

    from silenttwin.io.provenance import collect_provenance, source_tree_hash

    observed = source_tree_hash()
    provenance = collect_provenance()
    revision = provenance.get("code_revision")
    if (
        provenance.get("code_dirty") is not False
        or provenance.get("source_tree_hash") != observed
        or not isinstance(revision, str)
        or len(revision) != 40
    ):
        raise ValueError(
            "runtime-bound scientific artifacts require a clean Git checkpoint"
        )
    return observed


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


def _assess_train_pair_feasibility(args: argparse.Namespace) -> dict[str, Any]:
    from silenttwin.io.jsonl import read_jsonl
    from silenttwin.io.provenance import source_tree_hash

    from .pair_mining import make_train_pair_feasibility_report

    if args.assert_development_and_test_results_uninspected is not True:
        raise ValueError(
            "train feasibility requires the development/test-uninspected assertion"
        )
    report = make_train_pair_feasibility_report(
        catalog=_read_object(args.catalog),
        split_manifest=_read_object(args.splits),
        strategy_catalog=_read_object(args.strategy_catalog),
        train_observations=read_jsonl(args.train_observations),
        train_observation_manifest=_read_object(
            args.train_observation_manifest
        ),
        action_eligibility_manifest=_read_object(args.action_eligibility),
        analysis_source_tree_hash=source_tree_hash(),
    )
    reused = _atomic_write_immutable(args.output, report)
    return {
        "train_pair_feasibility_hash": report["train_pair_feasibility_hash"],
        "overall_disposition": report["overall_disposition"],
        "development_submission_permitted": report[
            "development_submission_permitted"
        ],
        "maximum_complementary_scenario_count_by_suite": {
            suite: value["maximum_complementary_scenario_count"]
            for suite, value in report["suite_reports"].items()
        },
        "output": str(args.output),
        "reused_existing_freeze": reused,
        "development_observations_inspected": False,
        "test_outcomes_inspected": False,
    }


def _audit_train_pair_design(args: argparse.Namespace) -> dict[str, Any]:
    from silenttwin.io.jsonl import read_jsonl
    from silenttwin.io.provenance import source_tree_hash

    from .pair_mining import make_train_pair_design_audit

    if args.assert_development_and_test_results_uninspected is not True:
        raise ValueError(
            "train design audit requires the development/test-uninspected assertion"
        )
    report = make_train_pair_design_audit(
        catalog=_read_object(args.catalog),
        split_manifest=_read_object(args.splits),
        strategy_catalog=_read_object(args.strategy_catalog),
        train_observations=read_jsonl(args.train_observations),
        train_observation_manifest=_read_object(
            args.train_observation_manifest
        ),
        train_pair_feasibility_report=_read_object(
            args.train_pair_feasibility_report
        ),
        action_eligibility_manifest=_read_object(args.action_eligibility),
        analysis_source_tree_hash=source_tree_hash(),
    )
    reused = _atomic_write_immutable(args.output, report)
    return {
        "train_pair_design_audit_hash": report[
            "train_pair_design_audit_hash"
        ],
        "overall_disposition": report["overall_disposition"],
        "development_submission_permitted": report[
            "development_submission_permitted"
        ],
        "suite_geometry": {
            suite: {
                "maximum_within_scenario_complementarity_across_profile_pairs": (
                    value[
                        "maximum_within_scenario_complementarity_across_profile_pairs"
                    ]
                ),
                "observed_attainability_dispositions": value[
                    "observed_attainability_dispositions"
                ],
            }
            for suite, value in report["suite_geometry"].items()
        },
        "output": str(args.output),
        "reused_existing_freeze": reused,
        "development_observations_inspected": False,
        "test_outcomes_inspected": False,
    }


def _census_scientific_v5_representability(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from silenttwin.io.provenance import source_tree_hash

    from . import compat
    from .successor_design import make_scientific_v5_representability_census

    if args.assert_development_and_test_results_uninspected is not True:
        raise ValueError(
            "scientific-v5 census requires the development/test-uninspected "
            "assertion"
        )
    if args.acknowledge_adaptive_use_of_v4_train_results is not True:
        raise ValueError(
            "scientific-v5 census requires acknowledgement that aggregate v4 "
            "train geometry informed the successor"
        )
    report = make_scientific_v5_representability_census(
        catalog=_read_object(args.catalog),
        split_manifest=_read_object(args.splits),
        action_eligibility_manifest=_read_object(args.action_eligibility),
        predecessor_strategy_catalog=_read_object(
            args.predecessor_strategy_catalog
        ),
        predecessor_train_design_audit=_read_object(
            args.predecessor_train_design_audit
        ),
        analysis_source_tree_hash=source_tree_hash(),
        compat=compat,
    )
    reused = _atomic_write_immutable(args.output, report)
    return {
        "scientific_v5_representability_hash": report[
            "scientific_v5_representability_hash"
        ],
        "protocol_amendment_hash": report["protocol_amendment"][
            "protocol_amendment_hash"
        ],
        "overall_disposition": report["overall_disposition"],
        "successor_catalog_authoring_permitted": report[
            "successor_catalog_authoring_permitted"
        ],
        "selected_scenario_count": report["selected_scenario_count"],
        "action_validation_count": report["action_validation_count"],
        "coverage_by_suite_split": report["coverage_by_suite_split"],
        "h200_submission_permitted": False,
        "development_submission_permitted": False,
        "development_monitor_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "output": str(args.output),
        "reused_existing_freeze": reused,
    }


def _freeze_scientific_v5_candidate_catalog(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from silenttwin.io.provenance import source_tree_hash

    from .successor_design import make_scientific_v5_candidate_strategy_catalog

    if args.assert_development_and_test_results_uninspected is not True:
        raise ValueError(
            "scientific-v5 catalog freeze requires the development/test-"
            "uninspected assertion"
        )
    report = make_scientific_v5_candidate_strategy_catalog(
        census=_read_object(args.representability_census),
        catalog=_read_object(args.catalog),
        split_manifest=_read_object(args.splits),
        action_eligibility_manifest=_read_object(args.action_eligibility),
        predecessor_strategy_catalog=_read_object(
            args.predecessor_strategy_catalog
        ),
        predecessor_train_design_audit=_read_object(
            args.predecessor_train_design_audit
        ),
        authoring_source_tree_hash=source_tree_hash(),
    )
    reused = _atomic_write_immutable(args.output, report)
    cohort = report["scenario_cohort"]["selected_scenario_ids_by_split"]
    return {
        "candidate_strategy_catalog_hash": report[
            "candidate_strategy_catalog_hash"
        ],
        "representability_census_hash": report[
            "representability_census_hash"
        ],
        "scenario_cohort_hash": report["scenario_cohort"]["cohort_hash"],
        "train_scenario_count": len(cohort["train"]),
        "development_scenario_count": len(cohort["development"]),
        "test_scenario_count": len(cohort["test"]),
        "overall_disposition": report["overall_disposition"],
        "learned_wheel_build_permitted": report[
            "learned_wheel_build_permitted"
        ],
        "h200_submission_permitted": False,
        "development_submission_permitted": False,
        "development_monitor_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "output": str(args.output),
        "reused_existing_freeze": reused,
    }


def _bind_scientific_v5_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from .runtime_integrity import (
        derive_learned_runtime_fingerprint,
        make_learned_runtime_provenance,
        verify_installed_distribution_against_wheel,
    )
    from .successor_design import (
        make_runtime_bound_scientific_v5_candidate_strategy_catalog,
    )

    if args.assert_development_and_test_results_uninspected is not True:
        raise ValueError(
            "scientific-v5 runtime binding requires the development/test-"
            "uninspected assertion"
        )
    if args.assert_wheel_built_from_current_clean_commit is not True:
        raise ValueError(
            "scientific-v5 runtime binding requires the clean-commit wheel "
            "assertion"
        )
    runtime_source = _clean_source_tree_hash()
    runtime_report = derive_learned_runtime_fingerprint(args.dependency_lock)
    runtime_provenance = make_learned_runtime_provenance(runtime_report)
    wheel_verification = verify_installed_distribution_against_wheel(
        wheel_artifact=args.wheel_artifact,
        distribution_name="silenttwin",
        expected_version="0.1.0",
    )
    report = make_runtime_bound_scientific_v5_candidate_strategy_catalog(
        design_catalog=_read_object(args.design_strategy_catalog),
        census=_read_object(args.representability_census),
        catalog=_read_object(args.catalog),
        split_manifest=_read_object(args.splits),
        action_eligibility_manifest=_read_object(args.action_eligibility),
        predecessor_strategy_catalog=_read_object(
            args.predecessor_strategy_catalog
        ),
        predecessor_train_design_audit=_read_object(
            args.predecessor_train_design_audit
        ),
        runtime_source_tree_hash=runtime_source,
        learned_runtime_provenance=runtime_provenance,
        installed_wheel_verification=wheel_verification,
    )
    reused = _atomic_write_immutable(args.output, report)
    binding = report["runtime_binding"]
    return {
        "candidate_strategy_catalog_hash": report[
            "candidate_strategy_catalog_hash"
        ],
        "design_candidate_strategy_catalog_hash": binding[
            "design_candidate_strategy_catalog_hash"
        ],
        "runtime_binding_hash": binding["runtime_binding_hash"],
        "runtime_source_tree_hash": runtime_source,
        "learned_runtime_fingerprint": binding[
            "learned_runtime_fingerprint"
        ],
        "wheel_sha256": wheel_verification["wheel_sha256"],
        "overall_disposition": report["overall_disposition"],
        "engineering_conformance_spec_authoring_permitted": True,
        "h200_submission_permitted": False,
        "development_submission_permitted": False,
        "development_monitor_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "output": str(args.output),
        "reused_existing_freeze": reused,
    }


def _freeze_scientific_v5_conformance_spec(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from .conformance import make_scientific_v5_conformance_spec
    from .successor_design import (
        validate_runtime_bound_scientific_v5_candidate_strategy_catalog,
    )

    if args.assert_development_and_test_results_uninspected is not True:
        raise ValueError(
            "scientific-v5 conformance freeze requires the development/test-"
            "uninspected assertion"
        )
    runtime_source = _clean_source_tree_hash()
    catalog = _read_object(args.catalog)
    splits = _read_object(args.splits)
    runtime_catalog = _read_object(args.runtime_strategy_catalog)
    validate_runtime_bound_scientific_v5_candidate_strategy_catalog(
        runtime_catalog,
        design_catalog=_read_object(args.design_strategy_catalog),
        census=_read_object(args.representability_census),
        catalog=catalog,
        split_manifest=splits,
        action_eligibility_manifest=_read_object(args.action_eligibility),
        predecessor_strategy_catalog=_read_object(
            args.predecessor_strategy_catalog
        ),
        predecessor_train_design_audit=_read_object(
            args.predecessor_train_design_audit
        ),
        runtime_source_tree_hash=runtime_source,
    )
    report = make_scientific_v5_conformance_spec(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=runtime_catalog,
        source_tree_hash=runtime_source,
    )
    reused = _atomic_write_immutable(args.output, report)
    return {
        "conformance_spec_hash": report["conformance_spec_hash"],
        "candidate_strategy_catalog_hash": report[
            "candidate_strategy_catalog_hash"
        ],
        "runtime_fingerprint": report["runtime_fingerprint"],
        "source_tree_hash": report["source_tree_hash"],
        "scenario_id": report["scenario_id"],
        "strategy_ids": report["strategy_ids"],
        "monitor_profile_ids": report["monitor_profile_ids"],
        "scientific_evidence_eligible": False,
        "h200_submission_permitted": False,
        "development_monitor_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "output": str(args.output),
        "reused_existing_freeze": reused,
    }


def _freeze_scientific_v6_recipient_separation(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from .recipient_separation import (
        make_scientific_v6_recipient_separation_artifacts,
    )

    if args.assert_development_and_test_results_uninspected is not True:
        raise ValueError(
            "scientific-v6 recipient separation requires the development/test-"
            "uninspected assertion"
        )
    if args.acknowledge_adaptive_use_of_v5_train_results is not True:
        raise ValueError(
            "scientific-v6 recipient separation requires acknowledgement that "
            "scientific-v5 train geometry informed the redesign"
        )
    source_hash = _clean_source_tree_hash()
    strategy_catalog, pair_registry = (
        make_scientific_v6_recipient_separation_artifacts(
            protocol=_read_object(args.protocol),
            catalog=_read_object(args.catalog),
            split_manifest=_read_object(args.splits),
            action_eligibility_manifest=_read_object(
                args.action_eligibility
            ),
            predecessor_strategy_catalog=_read_object(
                args.predecessor_strategy_catalog
            ),
            predecessor_train_design_audit=_read_object(
                args.predecessor_train_design_audit
            ),
            analysis_plan=_read_object(args.analysis_plan),
            authoring_source_tree_hash=source_hash,
        )
    )
    # Detect every collision before installing either half of the frozen pair.
    for path, value in (
        (args.strategy_catalog_output, strategy_catalog),
        (args.pair_registry_output, pair_registry),
    ):
        if path.exists() and _read_object(path) != value:
            raise ValueError(
                f"refusing to overwrite conflicting frozen artifact {path}"
            )
    reused_strategy = _atomic_write_immutable(
        args.strategy_catalog_output, strategy_catalog
    )
    reused_pair = _atomic_write_immutable(
        args.pair_registry_output, pair_registry
    )
    selected = pair_registry["pilot_scenario_ids_by_split"]
    return {
        "recipient_separation_protocol_hash": pair_registry[
            "recipient_separation_protocol_hash"
        ],
        "candidate_strategy_catalog_hash": strategy_catalog[
            "candidate_strategy_catalog_hash"
        ],
        "pair_registry_hash": pair_registry["pair_registry_hash"],
        "protocol_disposition": pair_registry["protocol_disposition"],
        "authoring_source_tree_hash": source_hash,
        "train_scenario_count": len(selected["train"]),
        "development_scenario_count_reserved": len(
            selected["development"]
        ),
        "test_scenario_count": len(selected["test"]),
        "execution_permitted_splits": pair_registry[
            "execution_permitted_splits"
        ],
        "security_experiments_ready": True,
        "clean_repair_experiment_ready": False,
        "development_submission_permitted": False,
        "held_out_evaluation_permitted": False,
        "strategy_catalog_output": str(args.strategy_catalog_output),
        "pair_registry_output": str(args.pair_registry_output),
        "reused_existing_strategy_catalog": reused_strategy,
        "reused_existing_pair_registry": reused_pair,
    }


def _freeze_sample_size(args: argparse.Namespace) -> dict[str, Any]:
    from .config import AGENTDOJO_SUITES, bundle_hash
    from .freeze import (
        deterministic_test_allocation,
        make_agentdojo_sample_size_freeze,
        validate_development_analysis_manifest,
    )
    from .grid import (
        is_estimation_only_protocol_disposition,
        load_frozen_inputs,
        validate_structural_splits,
    )

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
    if is_estimation_only_protocol_disposition(
        inputs.pair_registry.get("protocol_disposition")
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

    feasibility = commands.add_parser(
        "assess-train-pair-feasibility",
        help=(
            "validate train observations and freeze the complementary-pair "
            "feasibility gate before development"
        ),
    )
    feasibility.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    feasibility.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    feasibility.add_argument(
        "--action-eligibility",
        type=Path,
        default=DEFAULT_ACTION_ELIGIBILITY_PATH,
    )
    feasibility.add_argument("--strategy-catalog", type=Path, required=True)
    feasibility.add_argument("--train-observations", type=Path, required=True)
    feasibility.add_argument(
        "--train-observation-manifest", type=Path, required=True
    )
    feasibility.add_argument("--output", type=Path, required=True)
    feasibility.add_argument(
        "--assert-development-and-test-results-uninspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that neither development nor held-out outcomes "
            "informed this train gate"
        ),
    )
    feasibility.set_defaults(handler=_assess_train_pair_feasibility)

    design_audit = commands.add_parser(
        "audit-train-pair-design",
        help=(
            "validate train observations and freeze exact profile/candidate "
            "decision geometry without weakening the feasibility gate"
        ),
    )
    design_audit.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG_PATH
    )
    design_audit.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS_PATH
    )
    design_audit.add_argument(
        "--action-eligibility",
        type=Path,
        default=DEFAULT_ACTION_ELIGIBILITY_PATH,
    )
    design_audit.add_argument("--strategy-catalog", type=Path, required=True)
    design_audit.add_argument("--train-observations", type=Path, required=True)
    design_audit.add_argument(
        "--train-observation-manifest", type=Path, required=True
    )
    design_audit.add_argument(
        "--train-pair-feasibility-report", type=Path, required=True
    )
    design_audit.add_argument("--output", type=Path, required=True)
    design_audit.add_argument(
        "--assert-development-and-test-results-uninspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that neither development nor held-out outcomes "
            "informed this train-only diagnostic"
        ),
    )
    design_audit.set_defaults(handler=_audit_train_pair_design)

    v5_census = commands.add_parser(
        "census-scientific-v5-representability",
        help=(
            "execute and freeze the model-free common-objective scientific-v5 "
            "candidate census"
        ),
    )
    v5_census.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG_PATH
    )
    v5_census.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS_PATH
    )
    v5_census.add_argument(
        "--action-eligibility",
        type=Path,
        default=DEFAULT_ACTION_ELIGIBILITY_PATH,
    )
    v5_census.add_argument(
        "--predecessor-strategy-catalog", type=Path, required=True
    )
    v5_census.add_argument(
        "--predecessor-train-design-audit", type=Path, required=True
    )
    v5_census.add_argument("--output", type=Path, required=True)
    v5_census.add_argument(
        "--assert-development-and-test-results-uninspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that neither development nor held-out monitor "
            "outcomes informed the successor census"
        ),
    )
    v5_census.add_argument(
        "--acknowledge-adaptive-use-of-v4-train-results",
        action="store_true",
        required=True,
        help=(
            "record that aggregate scientific-v4 train geometry informed the "
            "successor design"
        ),
    )
    v5_census.set_defaults(handler=_census_scientific_v5_representability)

    v5_catalog = commands.add_parser(
        "freeze-scientific-v5-candidate-catalog",
        help=(
            "derive and immutably freeze the subset-aware scientific-v5 "
            "candidate catalog"
        ),
    )
    v5_catalog.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG_PATH
    )
    v5_catalog.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS_PATH
    )
    v5_catalog.add_argument(
        "--action-eligibility",
        type=Path,
        default=DEFAULT_ACTION_ELIGIBILITY_PATH,
    )
    v5_catalog.add_argument(
        "--representability-census", type=Path, required=True
    )
    v5_catalog.add_argument(
        "--predecessor-strategy-catalog", type=Path, required=True
    )
    v5_catalog.add_argument(
        "--predecessor-train-design-audit", type=Path, required=True
    )
    v5_catalog.add_argument("--output", type=Path, required=True)
    v5_catalog.add_argument(
        "--assert-development-and-test-results-uninspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that no development or held-out monitor outcome "
            "informed the catalog freeze"
        ),
    )
    v5_catalog.set_defaults(handler=_freeze_scientific_v5_candidate_catalog)

    v5_runtime = commands.add_parser(
        "bind-scientific-v5-runtime",
        help=(
            "bind a reproducible installed SilentTwin wheel and learned "
            "runtime to the reviewed scientific-v5 design catalog"
        ),
    )
    v5_runtime.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG_PATH
    )
    v5_runtime.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS_PATH
    )
    v5_runtime.add_argument(
        "--action-eligibility",
        type=Path,
        default=DEFAULT_ACTION_ELIGIBILITY_PATH,
    )
    v5_runtime.add_argument(
        "--representability-census", type=Path, required=True
    )
    v5_runtime.add_argument(
        "--predecessor-strategy-catalog", type=Path, required=True
    )
    v5_runtime.add_argument(
        "--predecessor-train-design-audit", type=Path, required=True
    )
    v5_runtime.add_argument(
        "--design-strategy-catalog", type=Path, required=True
    )
    v5_runtime.add_argument("--dependency-lock", type=Path, required=True)
    v5_runtime.add_argument("--wheel-artifact", type=Path, required=True)
    v5_runtime.add_argument("--output", type=Path, required=True)
    v5_runtime.add_argument(
        "--assert-development-and-test-results-uninspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that no development or held-out model outcome "
            "informed the runtime freeze"
        ),
    )
    v5_runtime.add_argument(
        "--assert-wheel-built-from-current-clean-commit",
        action="store_true",
        required=True,
        help=(
            "required operator assertion that the supplied wheel was "
            "reproducibly built from the current clean source tree"
        ),
    )
    v5_runtime.set_defaults(handler=_bind_scientific_v5_runtime)

    v5_conformance = commands.add_parser(
        "freeze-scientific-v5-conformance-spec",
        help=(
            "derive one development-only engineering conformance spec from "
            "the exact runtime-bound scientific-v5 catalog"
        ),
    )
    v5_conformance.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG_PATH
    )
    v5_conformance.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS_PATH
    )
    v5_conformance.add_argument(
        "--action-eligibility",
        type=Path,
        default=DEFAULT_ACTION_ELIGIBILITY_PATH,
    )
    v5_conformance.add_argument(
        "--representability-census", type=Path, required=True
    )
    v5_conformance.add_argument(
        "--predecessor-strategy-catalog", type=Path, required=True
    )
    v5_conformance.add_argument(
        "--predecessor-train-design-audit", type=Path, required=True
    )
    v5_conformance.add_argument(
        "--design-strategy-catalog", type=Path, required=True
    )
    v5_conformance.add_argument(
        "--runtime-strategy-catalog", type=Path, required=True
    )
    v5_conformance.add_argument("--output", type=Path, required=True)
    v5_conformance.add_argument(
        "--assert-development-and-test-results-uninspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that no development or held-out outcome "
            "informed the engineering spec"
        ),
    )
    v5_conformance.set_defaults(
        handler=_freeze_scientific_v5_conformance_spec
    )

    v6_recipient = commands.add_parser(
        "freeze-scientific-v6-recipient-separation",
        help=(
            "derive the train-only scientific-v6 authored-authorization "
            "candidate catalog and pair registry"
        ),
    )
    v6_recipient.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_RECIPIENT_SEPARATION_PROTOCOL_PATH,
    )
    v6_recipient.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG_PATH
    )
    v6_recipient.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS_PATH
    )
    v6_recipient.add_argument(
        "--action-eligibility",
        type=Path,
        default=DEFAULT_ACTION_ELIGIBILITY_PATH,
    )
    v6_recipient.add_argument(
        "--analysis-plan",
        type=Path,
        default=DEFAULT_RECIPIENT_SEPARATION_ANALYSIS_PATH,
    )
    v6_recipient.add_argument(
        "--predecessor-strategy-catalog", type=Path, required=True
    )
    v6_recipient.add_argument(
        "--predecessor-train-design-audit", type=Path, required=True
    )
    v6_recipient.add_argument(
        "--strategy-catalog-output", type=Path, required=True
    )
    v6_recipient.add_argument(
        "--pair-registry-output", type=Path, required=True
    )
    v6_recipient.add_argument(
        "--assert-development-and-test-results-uninspected",
        action="store_true",
        required=True,
        help=(
            "required assertion that no development or held-out outcome "
            "informed the scientific-v6 freeze"
        ),
    )
    v6_recipient.add_argument(
        "--acknowledge-adaptive-use-of-v5-train-results",
        action="store_true",
        required=True,
        help=(
            "record that scientific-v5 train-only geometry informed the "
            "recipient-separation redesign"
        ),
    )
    v6_recipient.set_defaults(
        handler=_freeze_scientific_v6_recipient_separation
    )

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
