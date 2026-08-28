"""Strict JSONL validation and atomic publication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from silenttwin.config import SCHEMA_VERSION, canonical_json
from silenttwin.schemas import assert_agent_visible, is_agentdojo_public_envelope


class ResultValidationError(ValueError):
    """A result exists but is incomplete, malformed, or incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    result_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with result_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ResultValidationError(
                        f"{result_path}: blank line at JSONL line {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ResultValidationError(
                        f"{result_path}: invalid JSON at line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ResultValidationError(
                        f"{result_path}: line {line_number} is not a JSON object"
                    )
                records.append(record)
    except FileNotFoundError as error:
        raise ResultValidationError(f"missing result file: {result_path}") from error
    return records


def _check_visible_value(value: Any, path: str = "agent_visible_transcript") -> None:
    """Reject obvious trusted-only fields in the visible transcript namespace."""

    try:
        assert_agent_visible(value, path)
    except ValueError as error:
        raise ResultValidationError(str(error)) from error

    # Typed AgentDojo application payload is opaque to the legacy Tier-1 key
    # denylist below. The envelope constructor already rejected trusted DTOs.
    if is_agentdojo_public_envelope(value):
        return

    forbidden_keys = {
        "private_state",
        "actual_state",
        "donor_state",
        "feedback_source_state",
        "theta",
        "world",
        "world_pair",
        "session",
        "protected_session",
        "private_record",
        "private_records",
        "staged_effect",
        "staged_effects",
        "controller",
        "trusted_report",
        "trusted_evaluation",
        "true_policy",
        "monitor_decision",
        "monitor_verdict",
        "true_policy_label",
        "policy_epoch",
        "commit_token",
        "admissible",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden_keys:
                raise ResultValidationError(f"trusted-only field {key!r} appears at {path}")
            _check_visible_value(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _check_visible_value(child, f"{path}[{index}]")


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_experiment: str | None = None,
    expected_configuration_hash: str | None = None,
    expected_sample_count: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ResultValidationError("result JSONL is empty")
    summaries = [record for record in records if record.get("record_type") == "summary"]
    if len(summaries) != 1:
        raise ResultValidationError(
            f"result must contain exactly one summary record; found {len(summaries)}"
        )
    if records[-1].get("record_type") != "summary":
        raise ResultValidationError("the summary record must be the final JSONL record")
    samples = list(records[:-1])
    if any(record.get("record_type") != "sample" for record in samples):
        raise ResultValidationError("all records before the summary must be sample records")
    summary = dict(summaries[0])
    summary_count = summary.get("sample_count")
    if summary_count != len(samples):
        raise ResultValidationError(
            f"summary sample_count={summary_count!r}, but JSONL has {len(samples)} samples"
        )
    if expected_sample_count is not None and len(samples) != expected_sample_count:
        raise ResultValidationError(
            f"expected {expected_sample_count} samples, found {len(samples)}"
        )
    sample_ids: set[str] = set()
    for index, record in enumerate(records):
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ResultValidationError(
                f"record {index} schema {record.get('schema_version')!r} != {SCHEMA_VERSION!r}"
            )
        if expected_experiment and record.get("experiment_id") != expected_experiment:
            raise ResultValidationError(
                f"record {index} experiment {record.get('experiment_id')!r} "
                f"!= {expected_experiment!r}"
            )
        if expected_configuration_hash and record.get("configuration_hash") != expected_configuration_hash:
            raise ResultValidationError(
                f"record {index} has incompatible configuration hash"
            )
        if record.get("record_type") == "sample":
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ResultValidationError(f"sample record {index} has no sample_id")
            if sample_id in sample_ids:
                raise ResultValidationError(f"duplicate sample_id {sample_id!r}")
            sample_ids.add(sample_id)
            visible_fields = (
                "agent_visible_transcript",
                "delivered_visible_transcript",
                "decision_visible_transcript",
                "postselection_visible_transcript",
            )
            for field in visible_fields:
                if field in record:
                    _check_visible_value(record[field], field)
            callback_fields = (
                "agent_visible_transcript",
                "delivered_visible_transcript",
                "decision_visible_transcript",
            )
            callback_values = [record[field] for field in callback_fields if field in record]
            if callback_values and any(
                value != callback_values[0] for value in callback_values[1:]
            ):
                raise ResultValidationError(
                    f"sample {sample_id!r} has inconsistent attacker-visible transcript aliases"
                )
            trusted_evaluation = record.get("trusted_evaluation", {})
            if not isinstance(trusted_evaluation, Mapping) or not (
                {"private_state", "actual_hidden_state"} & set(trusted_evaluation)
            ):
                raise ResultValidationError(
                    f"sample {sample_id!r} lacks a trusted hidden-state field"
                )
    return summary


def atomic_write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> None:
    """Validate then atomically rename a complete JSONL into place."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(record) for record in records]
    validate_records(materialized)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in materialized:
                handle.write(canonical_json(record))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Parse the bytes that will actually be published, not only the source
        # Python values, before the atomic rename.
        validate_records(read_jsonl(temporary_path))
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_objects_jsonl(
    path: Path | str, records: Iterable[Mapping[str, Any]]
) -> None:
    """Atomically publish a generic JSONL stream, including an empty stream.

    Result bundles use :func:`atomic_write_jsonl`, whose summary-record contract
    is intentionally stricter.  Checkpoint and failure ledgers are ordinary
    append-like object streams and use this helper instead.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(record) for record in records]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in materialized:
                handle.write(canonical_json(record))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        parsed = read_jsonl(temporary_path) if materialized else []
        if parsed != materialized:
            raise ResultValidationError("generic JSONL round-trip mismatch")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Ensure the serialized file is complete JSON before publication.
        with temporary_path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, dict):
            raise ResultValidationError("atomic JSON output must be an object")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_create_json(path: Path | str, value: Mapping[str, Any]) -> None:
    """Atomically publish one JSON object without replacing an existing file.

    The temporary file is fully flushed and parsed before a same-filesystem
    hard link publishes it. ``os.link`` is the no-clobber commit point: if a
    concurrent writer already created the destination, publication fails and
    the existing bytes remain untouched.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with temporary_path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, dict):
            raise ResultValidationError("atomic JSON output must be an object")
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise ResultValidationError(
                f"refusing to replace existing JSON output: {destination}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
