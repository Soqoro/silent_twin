"""Strict aggregation with matched cohorts and paired bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from silenttwin.config import EXPERIMENTS, stable_hash
from silenttwin.io.jsonl import ResultValidationError, atomic_write_json, read_jsonl
from silenttwin.io.manifests import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    utc_now,
    validate_result_directory,
)
from silenttwin.metrics.confidence_intervals import bootstrap_ci, paired_bootstrap_ci


AGGREGATE_SCHEMA_VERSION = "silenttwin.aggregate.v1"


@dataclass(frozen=True)
class RunData:
    directory: Path
    manifest: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.manifest["configuration"])

    @property
    def configuration_hash(self) -> str:
        return str(self.manifest["configuration_hash"])

    @property
    def cohort(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (
                    str(sample["paired_world_id"]),
                    str(sample["public_instance_hash"]),
                )
                for sample in self.samples
            )
        )

    @property
    def cohort_hash(self) -> str:
        return stable_hash(self.cohort)


def _is_within(path: Path, possible_parent: Path) -> bool:
    try:
        path.resolve().relative_to(possible_parent.resolve())
    except ValueError:
        return False
    return True


def discover_runs(
    input_root: Path | str,
    *,
    experiment: str,
    excluded_root: Path | None = None,
) -> list[RunData]:
    root = Path(input_root)
    if not root.is_dir():
        raise ResultValidationError(f"aggregation input root does not exist: {root}")
    manifests = sorted(root.rglob(MANIFEST_FILENAME))
    results = sorted(root.rglob(RESULT_FILENAME))
    if excluded_root is not None:
        manifests = [path for path in manifests if not _is_within(path, excluded_root)]
        results = [path for path in results if not _is_within(path, excluded_root)]
    manifest_dirs = {path.parent.resolve() for path in manifests}
    result_dirs = {path.parent.resolve() for path in results}
    orphan_results = result_dirs - manifest_dirs
    orphan_manifests = manifest_dirs - result_dirs
    if orphan_results:
        raise ResultValidationError(
            "result files without manifests: "
            + ", ".join(str(path) for path in sorted(orphan_results))
        )
    if orphan_manifests:
        raise ResultValidationError(
            "manifests without result files: "
            + ", ".join(str(path) for path in sorted(orphan_manifests))
        )
    if not manifests:
        raise ResultValidationError(f"no complete run manifests found below {root}")

    runs: list[RunData] = []
    seen_hashes: dict[str, Path] = {}
    source_hashes: set[str] = set()
    provenance_fingerprints: set[str] = set()
    for manifest_path in manifests:
        directory = manifest_path.parent
        manifest = validate_result_directory(directory, expected_experiment=experiment)
        configuration_hash = str(manifest["configuration_hash"])
        if configuration_hash in seen_hashes:
            raise ResultValidationError(
                f"duplicate configuration hash {configuration_hash}: "
                f"{seen_hashes[configuration_hash]} and {directory}"
            )
        seen_hashes[configuration_hash] = directory
        source_hash = manifest.get("provenance", {}).get("source_tree_hash")
        if not source_hash:
            raise ResultValidationError(f"manifest lacks source_tree_hash: {manifest_path}")
        source_hashes.add(str(source_hash))
        provenance = manifest.get("provenance", {})
        provenance_fingerprints.add(
            stable_hash(
                {
                    key: provenance.get(key)
                    for key in (
                        "code_revision",
                        "source_tree_hash",
                        "package_version",
                        "python_implementation",
                        "python_version",
                        "platform",
                    )
                }
            )
        )
        records = read_jsonl(directory / RESULT_FILENAME)
        runs.append(
            RunData(
                directory=directory,
                manifest=manifest,
                samples=tuple(records[:-1]),
                summary=dict(records[-1]),
            )
        )
    if len(source_hashes) != 1:
        raise ResultValidationError(
            "aggregation refuses incompatible code provenance; found source hashes "
            + ", ".join(sorted(source_hashes))
        )
    if len(provenance_fingerprints) != 1:
        raise ResultValidationError(
            "aggregation refuses runs generated under incompatible provenance"
        )
    return runs


def _matching_key(run: RunData, experiment: str) -> tuple[Any, ...]:
    config = run.config
    if experiment in {"e1", "e2", "e3"}:
        fields = ("tier", "world_suite", "attacker", "seed", "num_samples")
    elif experiment == "e4":
        fields = (
            "tier",
            "world_suite",
            "attacker",
            "query_budget",
            "seed",
            "num_samples",
            "workflow",
        )
    else:
        fields = (
            "tier",
            "world_suite",
            "attacker",
            "query_budget",
            "seed",
            "num_samples",
        )
    return tuple(config.get(field) for field in fields)


def validate_matched_cohorts(runs: Sequence[RunData], experiment: str) -> None:
    """Reject cross-condition groups that do not contain identical tasks."""

    grouped: dict[tuple[Any, ...], list[RunData]] = defaultdict(list)
    for run in runs:
        grouped[_matching_key(run, experiment)].append(run)
    for key, members in grouped.items():
        if len(members) < 2:
            continue
        expected = members[0].cohort
        mismatches = [member for member in members[1:] if member.cohort != expected]
        if mismatches:
            locations = [members[0].directory, *(member.directory for member in mismatches)]
            raise ResultValidationError(
                f"matched-task cohort mismatch for comparison stratum {key!r}: "
                + ", ".join(str(location) for location in locations)
            )


def _numeric_sample_metric(sample: Mapping[str, Any], name: str) -> float:
    if name == "hidden_state_inference_accuracy":
        return float(sample["secret_prediction_correct"])
    if name == "prohibited_effect_rate":
        return float(sample["prohibited_effect_indicator"])
    if name == "benign_task_success":
        return float(sample["task_success_indicator"])
    if name == "eligible_action_salvage_rate":
        return float(sample.get("eligible_action_salvage_rate", 1.0))
    if name == "rejected_descendant_commit_rate":
        return float(sample["rejected_descendant_commit_rate"])
    raise KeyError(name)


def _aligned_values(
    target: RunData, reference: RunData, metric: str
) -> tuple[list[float], list[float]]:
    target_by_hash = {
        str(sample["public_instance_hash"]): sample for sample in target.samples
    }
    reference_by_hash = {
        str(sample["public_instance_hash"]): sample for sample in reference.samples
    }
    if set(target_by_hash) != set(reference_by_hash):
        raise ResultValidationError(
            f"cannot compare unmatched cohorts {target.directory} and {reference.directory}"
        )
    keys = sorted(target_by_hash)
    return (
        [_numeric_sample_metric(target_by_hash[key], metric) for key in keys],
        [_numeric_sample_metric(reference_by_hash[key], metric) for key in keys],
    )


def _same_except(
    target: RunData, reference: RunData, changing_fields: set[str]
) -> bool:
    left = target.config
    right = reference.config
    keys = set(left) | set(right)
    return all(left.get(key) == right.get(key) for key in keys - changing_fields)


def _add_comparison(
    comparisons: list[dict[str, Any]],
    target: RunData,
    reference: RunData,
    metric: str,
    comparison_kind: str,
) -> None:
    left, right = _aligned_values(target, reference, metric)
    estimate = sum(a - b for a, b in zip(left, right)) / len(left)
    bootstrap_seed = int(stable_hash([target.configuration_hash, reference.configuration_hash, metric])[:8], 16)
    lower, upper = paired_bootstrap_ci(
        left, right, resamples=2000, seed=bootstrap_seed
    )
    comparisons.append(
        {
            "comparison_kind": comparison_kind,
            "metric": metric,
            "estimate_target_minus_reference": estimate,
            "ci_level": 0.95,
            "ci_lower": lower,
            "ci_upper": upper,
            "matched_pair_count": len(left),
            "target_configuration_hash": target.configuration_hash,
            "reference_configuration_hash": reference.configuration_hash,
            "target_factors": target.config,
            "reference_factors": reference.config,
            "cohort_hash": target.cohort_hash,
        }
    )


def paired_comparisons(runs: Sequence[RunData], experiment: str) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    if experiment in {"e1", "e3", "e4"}:
        for target in runs:
            if target.config.get("runtime") == "silenttwin":
                continue
            references = [
                run
                for run in runs
                if run.config.get("runtime") == "silenttwin"
                and _same_except(target, run, {"runtime"})
            ]
            if len(references) > 1:
                raise ResultValidationError("ambiguous SilentTwin reference configuration")
            if references:
                metrics = (
                    ("hidden_state_inference_accuracy",)
                    if experiment == "e1"
                    else ("benign_task_success", "eligible_action_salvage_rate")
                    if experiment == "e4"
                    else ("hidden_state_inference_accuracy", "prohibited_effect_rate")
                )
                for metric in metrics:
                    _add_comparison(
                        comparisons,
                        target,
                        references[0],
                        metric,
                        "runtime_vs_silenttwin",
                    )
    if experiment in {"e1", "e2", "e3"}:
        for target in runs:
            if int(target.config.get("query_budget", 0)) == 0:
                continue
            references = [
                run
                for run in runs
                if int(run.config.get("query_budget", 0)) == 0
                and _same_except(target, run, {"query_budget"})
            ]
            if len(references) > 1:
                raise ResultValidationError("ambiguous no-feedback reference configuration")
            if references:
                metric = (
                    "hidden_state_inference_accuracy"
                    if experiment == "e1"
                    else "prohibited_effect_rate"
                )
                _add_comparison(
                    comparisons,
                    target,
                    references[0],
                    metric,
                    "query_budget_vs_zero",
                )
    if experiment == "e5":
        for target in runs:
            if target.config.get("ablation") == "none":
                continue
            references = [
                run
                for run in runs
                if run.config.get("ablation") == "none"
                and _same_except(target, run, {"ablation"})
            ]
            if len(references) > 1:
                raise ResultValidationError("ambiguous exact-SilentTwin ablation reference")
            if references:
                for metric in (
                    "hidden_state_inference_accuracy",
                    "prohibited_effect_rate",
                    "rejected_descendant_commit_rate",
                    "benign_task_success",
                ):
                    _add_comparison(
                        comparisons,
                        target,
                        references[0],
                        metric,
                        "ablation_vs_exact_silenttwin",
                    )
    return comparisons


def _group_record(run: RunData) -> dict[str, Any]:
    metrics = dict(run.summary.get("metrics", {}))
    confidence_intervals: dict[str, dict[str, float]] = {}
    for metric, summary_value in metrics.items():
        if not isinstance(summary_value, (int, float)) or isinstance(summary_value, bool):
            continue
        sample_metric = None
        for candidate in (
            "hidden_state_inference_accuracy",
            "prohibited_effect_rate",
            "benign_task_success",
            "eligible_action_salvage_rate",
            "rejected_descendant_commit_rate",
        ):
            if metric == candidate:
                sample_metric = candidate
                break
        if sample_metric is None:
            continue
        values = [_numeric_sample_metric(sample, sample_metric) for sample in run.samples]
        seed = int(run.configuration_hash[:8], 16)
        lower, upper = bootstrap_ci(values, resamples=2000, seed=seed)
        confidence_intervals[metric] = {
            "level": 0.95,
            "lower": lower,
            "upper": upper,
        }
    return {
        "configuration_hash": run.configuration_hash,
        "configuration": run.config,
        "cohort_hash": run.cohort_hash,
        "sample_count": len(run.samples),
        "metrics": metrics,
        "bootstrap_confidence_intervals": confidence_intervals,
        "source_directory": str(run.directory),
    }


def _flatten_group(group: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_type": "configuration",
        "configuration_hash": group["configuration_hash"],
        "cohort_hash": group["cohort_hash"],
        "sample_count": group["sample_count"],
    }
    row.update(group["configuration"])
    row.update(group["metrics"])
    return row


def _flatten_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_type": "paired_comparison",
        "comparison_kind": comparison["comparison_kind"],
        "metric": comparison["metric"],
        "estimate_target_minus_reference": comparison[
            "estimate_target_minus_reference"
        ],
        "ci_level": comparison["ci_level"],
        "ci_lower": comparison["ci_lower"],
        "ci_upper": comparison["ci_upper"],
        "matched_pair_count": comparison["matched_pair_count"],
        "target_configuration_hash": comparison["target_configuration_hash"],
        "reference_configuration_hash": comparison["reference_configuration_hash"],
        "cohort_hash": comparison["cohort_hash"],
    }


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    # Put identifiers first while preserving a deterministic remainder.
    preferred = [
        "row_type",
        "experiment",
        "configuration_hash",
        "cohort_hash",
        "sample_count",
    ]
    fieldnames = [field for field in preferred if field in fields] + [
        field for field in fields if field not in preferred
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _factor_metric_rows(
    groups: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    factor_names = (
        "tier",
        "world_suite",
        "runtime",
        "attacker",
        "query_budget",
        "seed",
        "condition",
        "workflow",
        "ablation",
    )
    for group in groups:
        config = group["configuration"]
        metrics = group["metrics"]
        row = {
            "configuration_hash": group["configuration_hash"],
            "cohort_hash": group["cohort_hash"],
            "sample_count": group["sample_count"],
            **{factor: config.get(factor) for factor in factor_names},
            **{metric: metrics.get(metric) for metric in metric_names},
        }
        rows.append(row)
    return rows


def _write_experiment_artifacts(
    destination: Path,
    experiment: str,
    groups: Sequence[Mapping[str, Any]],
) -> list[str]:
    specifications: list[tuple[str, tuple[str, ...]]] = []
    if experiment in {"e1", "e3"}:
        specifications.extend(
            [
                ("inference_accuracy_vs_q.csv", ("hidden_state_inference_accuracy",)),
                ("transcript_auc_vs_q.csv", ("transcript_distinguisher_auc",)),
            ]
        )
    if experiment in {"e2", "e3"}:
        specifications.extend(
            [
                (
                    "prohibited_effect_rate_vs_q.csv",
                    ("prohibited_effect_rate", "no_feedback_prohibited_effect_rate"),
                ),
                ("feedback_assisted_gain.csv", ("feedback_assisted_gain",)),
            ]
        )
    if experiment in {"e3", "e4", "e5"}:
        specifications.append(
            (
                "privacy_safety_utility.csv",
                (
                    "hidden_state_inference_accuracy",
                    "transcript_distinguisher_auc",
                    "prohibited_effect_rate",
                    "benign_task_success",
                    "eligible_action_salvage_rate",
                    "utility",
                ),
            )
        )
    if experiment == "e5":
        specifications.append(
            (
                "ablation_table.csv",
                (
                    "hidden_state_inference_accuracy",
                    "transcript_distinguisher_auc",
                    "paired_transcript_equality_rate",
                    "prohibited_effect_rate",
                    "rejected_descendant_commit_rate",
                    "atomicity_violation_rate",
                    "utility",
                    "eligible_action_salvage_rate",
                    "leakage_attributable_to_ablation",
                ),
            )
        )
    filenames: list[str] = []
    for filename, metric_names in specifications:
        _atomic_write_csv(
            destination / filename,
            _factor_metric_rows(groups, metric_names),
        )
        filenames.append(filename)
    return filenames


def aggregate_experiment(
    *,
    experiment: str,
    input_root: Path | str,
    output_dir: Path | str,
    expected_runs: int | None = None,
) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}")
    destination = Path(output_dir)
    runs = discover_runs(
        input_root,
        experiment=experiment,
        excluded_root=destination,
    )
    if expected_runs is not None:
        if expected_runs <= 0:
            raise ValueError("expected run count must be positive")
        if len(runs) != expected_runs:
            raise ResultValidationError(
                f"expected {expected_runs} completed grid runs below {input_root}, found {len(runs)}"
            )
    validate_matched_cohorts(runs, experiment)
    groups = [_group_record(run) for run in runs]
    comparisons = paired_comparisons(runs, experiment)
    source_hash = str(runs[0].manifest["provenance"]["source_tree_hash"])
    destination.mkdir(parents=True, exist_ok=True)
    artifact_files = _write_experiment_artifacts(destination, experiment, groups)
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "experiment_id": experiment,
        "generated_at": utc_now(),
        "input_root": str(Path(input_root)),
        "run_count": len(runs),
        "expected_run_count": expected_runs,
        "total_sample_count": sum(len(run.samples) for run in runs),
        "code_source_tree_hash": source_hash,
        "configuration_groups": groups,
        "paired_comparisons": comparisons,
        "artifact_files": ["summary.json", "summary.csv", *artifact_files],
        "aggregation_guarantees": {
            "matched_task_cohorts": True,
            "configuration_hashes_kept_separate": True,
            "paired_bootstrap_resamples": 2000,
        },
    }
    rows = [_flatten_group(group) for group in groups]
    rows.extend(_flatten_comparison(comparison) for comparison in comparisons)
    atomic_write_json(destination / "summary.json", aggregate)
    _atomic_write_csv(destination / "summary.csv", rows)
    return aggregate


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "RunData",
    "aggregate_experiment",
    "discover_runs",
    "paired_comparisons",
    "validate_matched_cohorts",
]
