"""Run manifests and strict reusable-result checks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from silenttwin.config import (
    MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ExperimentConfig,
    stable_hash,
)
from silenttwin.io.jsonl import (
    ResultValidationError,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    validate_records,
)
from silenttwin.io.provenance import (
    collect_provenance,
    provenance_compatible,
    provenance_mismatches,
)
from silenttwin.io.checkpoints import (
    CHECKPOINT_MANIFEST,
    CHECKPOINT_SCHEMA_VERSION,
)


RESULT_FILENAME = "result.jsonl"
MANIFEST_FILENAME = "manifest.json"
LOG_FILENAME = "run.log"
FAILURES_FILENAME = "failures.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_manifest(
    config: ExperimentConfig,
    *,
    result_path: Path,
    failures_path: Path,
    provenance: Mapping[str, Any],
    started_at: str,
    completed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "result_schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": config.experiment,
        "configuration": config.as_manifest_config(),
        "configuration_hash": config.configuration_hash,
        "operational_configuration": config.operational_dict(),
        "expected_sample_count": config.num_samples,
        "actual_sample_count": config.num_samples,
        "result_file": RESULT_FILENAME,
        "result_sha256": sha256_file(result_path),
        "failures_file": FAILURES_FILENAME,
        "failures_sha256": sha256_file(failures_path),
        "failure_count": len(read_jsonl(failures_path)),
        "checkpoint_manifest": CHECKPOINT_MANIFEST,
        "started_at": started_at,
        "completed_at": completed_at or utc_now(),
        "provenance": dict(provenance),
        "generation_provenance": {
            "tier": config.tier,
            "agent": config.attacker,
            "model_client": config.model_id,
            "model_revision": config.model_revision,
            "external_api_calls": 0,
        },
        "evaluation_provenance": {
            "evaluator": "silenttwin-finite-state-v1",
            "uses_trusted_metadata": True,
        },
        "orchestration": {
            "grid_hash": config.grid_hash,
            "grid_task_id": config.grid_task_id,
            "shard_id": config.shard_id,
            "pilot_id": config.pilot_id,
            "scheduler": dict(provenance.get("scheduler", {})),
        },
    }


def write_manifest(path: Path | str, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(path, manifest)


def read_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as error:
        raise ResultValidationError(f"missing manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ResultValidationError(f"invalid manifest JSON: {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ResultValidationError(f"manifest is not a JSON object: {manifest_path}")
    return manifest


def validate_result_directory(
    result_dir: Path | str,
    *,
    expected_config: ExperimentConfig | None = None,
    expected_experiment: str | None = None,
    current_provenance: Mapping[str, Any] | None = None,
    require_current_provenance: bool = False,
) -> dict[str, Any]:
    directory = Path(result_dir)
    manifest = read_manifest(directory / MANIFEST_FILENAME)
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ResultValidationError("incompatible manifest schema version")
    if manifest.get("result_schema_version") != SCHEMA_VERSION:
        raise ResultValidationError("incompatible result schema version")
    if manifest.get("status") != "complete":
        raise ResultValidationError(f"manifest status is {manifest.get('status')!r}, not 'complete'")
    manifest_config = manifest.get("configuration")
    if not isinstance(manifest_config, dict):
        raise ResultValidationError("manifest configuration is missing or is not an object")
    if stable_hash(manifest_config) != manifest.get("configuration_hash"):
        raise ResultValidationError("manifest configuration does not match its hash")
    if expected_experiment and manifest.get("experiment_id") != expected_experiment:
        raise ResultValidationError(
            f"expected experiment {expected_experiment!r}, found {manifest.get('experiment_id')!r}"
        )
    expected_hash = None
    expected_count = None
    if expected_config is not None:
        expected_hash = expected_config.configuration_hash
        expected_count = expected_config.num_samples
        if manifest.get("configuration_hash") != expected_hash:
            raise ResultValidationError("existing result has an incompatible configuration hash")
        if manifest.get("expected_sample_count") != expected_count:
            raise ResultValidationError("existing result has an incompatible sample count")
    if manifest.get("result_file") != RESULT_FILENAME:
        raise ResultValidationError(
            f"manifest result_file must be {RESULT_FILENAME!r}"
        )
    result_path = directory / RESULT_FILENAME
    if not result_path.is_file():
        raise ResultValidationError(f"manifest references missing result file: {result_path}")
    actual_digest = sha256_file(result_path)
    if manifest.get("result_sha256") != actual_digest:
        raise ResultValidationError("result digest does not match manifest")
    records = read_jsonl(result_path)
    summary = validate_records(
        records,
        expected_experiment=expected_experiment or manifest.get("experiment_id"),
        expected_configuration_hash=expected_hash or manifest.get("configuration_hash"),
        expected_sample_count=expected_count or manifest.get("expected_sample_count"),
    )
    if summary.get("configuration") != manifest_config:
        raise ResultValidationError("result summary configuration does not match manifest")
    if manifest.get("actual_sample_count") != len(records) - 1:
        raise ResultValidationError("manifest actual_sample_count does not match result")
    if manifest.get("failures_file") != FAILURES_FILENAME:
        raise ResultValidationError(
            f"manifest failures_file must be {FAILURES_FILENAME!r}"
        )
    failures_path = directory / FAILURES_FILENAME
    if not failures_path.is_file():
        raise ResultValidationError(f"complete run is missing failures file: {failures_path}")
    failures = read_jsonl(failures_path)
    if manifest.get("failures_sha256") != sha256_file(failures_path):
        raise ResultValidationError("failure ledger digest does not match manifest")
    if manifest.get("failure_count") != len(failures):
        raise ResultValidationError("manifest failure_count does not match failures file")
    checkpoint_path = directory / CHECKPOINT_MANIFEST
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ResultValidationError(
            f"complete run is missing checkpoint manifest: {checkpoint_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ResultValidationError(f"invalid checkpoint manifest: {error}") from error
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ResultValidationError("incompatible checkpoint manifest schema")
    if checkpoint.get("configuration_hash") != manifest.get("configuration_hash"):
        raise ResultValidationError("checkpoint and run manifest configuration hashes differ")
    if checkpoint.get("source_tree_hash") != manifest.get("provenance", {}).get(
        "source_tree_hash"
    ):
        raise ResultValidationError("checkpoint and run manifest source hashes differ")
    if checkpoint.get("status") != "complete":
        raise ResultValidationError("complete run has an incomplete checkpoint manifest")
    log_path = directory / LOG_FILENAME
    if not log_path.is_file():
        raise ResultValidationError(f"complete run is missing log file: {log_path}")
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResultValidationError(f"cannot read run log {log_path}: {error}") from error
    if "status=complete" not in log_text:
        raise ResultValidationError(f"run log does not record successful completion: {log_path}")
    expected_log_hash = f"configuration_hash={manifest.get('configuration_hash')}"
    if expected_log_hash not in log_text.splitlines():
        raise ResultValidationError(
            f"run log is not bound to the manifest configuration: {log_path}"
        )
    if require_current_provenance:
        current = dict(current_provenance or collect_provenance())
        recorded = manifest.get("provenance")
        if not isinstance(recorded, dict) or not provenance_compatible(recorded, current):
            mismatch = provenance_mismatches(recorded or {}, current)
            raise ResultValidationError(f"existing result provenance is incompatible: {mismatch}")
    return manifest
