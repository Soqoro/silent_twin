"""Development-only checkpoint protocol conformance for controlled runs.

This module is deliberately narrower than the experiment runner.  It checks one
development scenario, two frozen candidate strategies, one structured Qwen
attacker, and two Granite Guardian profiles backed by distinct retained local
clients, matching pair-observation generation's simultaneous-memory behavior.
The resulting artifact is engineering evidence only and cannot be used as a
scientific benchmark result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from silenttwin.backends.base import PublicPlan, PublicScenario
from silenttwin.io.jsonl import ResultValidationError, atomic_create_json
from silenttwin.io.provenance import collect_provenance

from .assembly import _monitor, _trusted_plan, model_client_from_identity
from .canonical import canonicalize_tool_schemas
from .catalog import validate_catalog
from .config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
    CONTROLLED_MODEL_PROMPT_HASH,
    ModelIdentity,
    load_json_object,
    stable_hash,
)
from .compat import (
    EXPECTED_ATTACKS,
    EXPECTED_DEFENSES,
    EXPECTED_INTERNAL_BENCHMARK_VERSIONS,
    EXPECTED_RELEASE_COUNTS,
    EXPECTED_WHEEL_SHA256,
)
from .monitors import MonitorInput
from .pair_mining import (
    _monitor_pair_compatibility,
    validate_candidate_strategy_catalog,
)
from .pipeline import StructuredControlledAttacker
from .runtime_validation import validate_environment_integrity
from .runtime_integrity import (
    RuntimeIntegrityError,
    validate_learned_runtime_provenance,
)
from .seeds import SeedSchedule
from .splits import validate_split_manifest
from .storage import _validate_model_call_record, _validate_monitor_call_record
from .visibility import public_value


CONFORMANCE_SPEC_SCHEMA_VERSION = (
    "silenttwin.agentdojo.controlled_conformance_spec.v1"
)
CONFORMANCE_REPORT_SCHEMA_VERSION = (
    "silenttwin.agentdojo.controlled_conformance_report.v1"
)
CONFORMANCE_ARTIFACT_CLASS = "development_checkpoint_conformance"
CONFORMANCE_EVIDENCE_CLASS = "engineering_conformance_only"
CONFORMANCE_CLAIM_BOUNDARY = (
    "development-only transport and protocol conformance; not scientific "
    "benchmark evidence"
)
QWEN_ATTACKER_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_ATTACKER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
QWEN_ATTACKER_CHECKPOINT_FINGERPRINT = (
    "sha256:bfb9ad97ebbceae4eb4b54fc85334d0a71f5e157176323712a7b3ed6e0d05e8e"
)
GRANITE_MONITOR_FAMILY = "granite_guardian_4_1_8b"
GRANITE_MONITOR_MODEL_ID = "ibm-granite/granite-guardian-4.1-8b"
GRANITE_MONITOR_REVISION = "e30b8a2343efe8030479777d467ebb305ca109e9"
GRANITE_MONITOR_CHECKPOINT_FINGERPRINT = (
    "sha256:31a587dc521951a7288ead06c9f8226bceb51d410094e8d47c04dee3602a746f"
)
_HEX_SHA256 = frozenset("0123456789abcdef")


class ConformanceError(ValueError):
    """A conformance specification or its frozen inputs are invalid."""


_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_class",
        "evidence_class",
        "scientific_evidence_eligible",
        "claim_boundary",
        "environment_backend",
        "tier2_track",
        "dataset_split",
        "test_outcomes_inspected",
        "pair_selection_eligible",
        "development_outcomes_generated",
        "catalog_hash",
        "split_manifest_hash",
        "candidate_strategy_catalog_hash",
        "runtime_fingerprint",
        "source_tree_hash",
        "scenario_id",
        "strategy_ids",
        "monitor_profile_ids",
        "attacker_identity",
        "conformance_spec_hash",
    }
)
_ATTACKER_IDENTITY_FIELDS = frozenset(
    {
        "role",
        "implementation",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "checkpoint_fingerprint",
        "prompt_hash",
        "runtime_fingerprint",
        "policy_hash",
        "threshold",
        "reasoning_mode",
        "dtype",
        "temperature",
        "top_p",
        "max_new_tokens",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_class",
        "evidence_class",
        "scientific_evidence_eligible",
        "claim_boundary",
        "environment_backend",
        "tier2_track",
        "dataset_split",
        "test_outcomes_inspected",
        "pair_selection_eligible",
        "development_outcomes_generated",
        "conformance_spec_hash",
        "source_tree_hash",
        "upstream_artifacts",
        "scenario",
        "selected_strategies",
        "selected_monitor_profiles",
        "attacker_identity",
        "learned_runtime",
        "compatibility",
        "source_provenance",
        "client_topology",
        "memory_evidence",
        "checks",
        "attacker_provenance",
        "errors",
        "summary",
        "status",
        "external_api_calls",
        "conformance_report_hash",
    }
)


@dataclass(frozen=True, slots=True)
class ConformanceDependencies:
    """Injectable process boundaries used by focused, model-free unit tests."""

    compat: Any | None = None
    model_client_factory: Callable[..., Any] = model_client_from_identity
    runtime_validator: Callable[..., Mapping[str, Any]] = (
        validate_environment_integrity
    )
    provenance_factory: Callable[[], Mapping[str, Any]] = collect_provenance
    memory_probe: Callable[..., Mapping[str, Any]] | None = None


def _cuda_memory_probe(
    *, stage: str, devices: Sequence[str]
) -> dict[str, Any]:
    """Capture process-local CUDA allocation and peak counters lazily."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - learned runtime only
        raise ConformanceError(
            "CUDA memory evidence requires the frozen learned Torch runtime"
        ) from exc
    if not torch.cuda.is_available():  # pragma: no cover - learned runtime only
        raise ConformanceError("CUDA memory evidence requested without a visible GPU")
    snapshots: list[dict[str, Any]] = []
    for device in dict.fromkeys(str(item) for item in devices):
        if not device.startswith("cuda"):
            raise ConformanceError(
                "checkpoint conformance requires CUDA device placement"
            )
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        snapshots.append(
            {
                "device": device,
                "device_name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return {"stage": stage, "devices": snapshots}


def _without_hash(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(document)
    payload.pop(field, None)
    return payload


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ConformanceError(
            f"{label} fields are not exact; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _two_distinct_strings(value: Any, *, label: str) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, str) or not item for item in value)
        or value[0] == value[1]
    ):
        raise ConformanceError(f"{label} must contain exactly two distinct IDs")
    return str(value[0]), str(value[1])


def _is_raw_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_SHA256 for character in value)
    )


def _validate_approved_attacker_identity(identity: ModelIdentity) -> None:
    if (
        identity.role != "attacker"
        or identity.implementation != "local_transformers"
        or identity.model_id != QWEN_ATTACKER_MODEL_ID
        or identity.model_revision != QWEN_ATTACKER_REVISION
        or identity.tokenizer_revision != QWEN_ATTACKER_REVISION
        or identity.checkpoint_fingerprint
        != QWEN_ATTACKER_CHECKPOINT_FINGERPRINT
        or identity.prompt_hash != CONTROLLED_MODEL_PROMPT_HASH
        or identity.dtype != "bfloat16"
        or float(identity.temperature) != 0.0
        or float(identity.top_p) != 1.0
        or identity.max_new_tokens != 256
        or identity.policy_hash is not None
        or identity.threshold is not None
        or identity.reasoning_mode is not None
    ):
        raise ConformanceError(
            "attacker_identity must select the approved immutable local "
            "Qwen2.5-7B controlled checkpoint"
        )


def validate_conformance_spec(document: Mapping[str, Any]) -> ModelIdentity:
    """Validate one exact, self-hashed, development-only conformance spec."""

    _exact_fields(document, _SPEC_FIELDS, label="conformance spec")
    if document.get("schema_version") != CONFORMANCE_SPEC_SCHEMA_VERSION:
        raise ConformanceError("unsupported conformance specification schema")
    fixed = {
        "artifact_class": CONFORMANCE_ARTIFACT_CLASS,
        "evidence_class": CONFORMANCE_EVIDENCE_CLASS,
        "scientific_evidence_eligible": False,
        "claim_boundary": CONFORMANCE_CLAIM_BOUNDARY,
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "dataset_split": "development",
        "test_outcomes_inspected": False,
        "pair_selection_eligible": False,
        "development_outcomes_generated": True,
    }
    for field, expected in fixed.items():
        if document.get(field) != expected:
            raise ConformanceError(
                f"conformance spec {field} must be {expected!r}"
            )
    recorded_hash = document.get("conformance_spec_hash")
    if not isinstance(recorded_hash, str) or recorded_hash != stable_hash(
        _without_hash(document, "conformance_spec_hash")
    ):
        raise ConformanceError("conformance_spec_hash is invalid")
    for field in (
        "catalog_hash",
        "split_manifest_hash",
        "candidate_strategy_catalog_hash",
        "source_tree_hash",
    ):
        value = document.get(field)
        if not _is_raw_sha256(value):
            raise ConformanceError(f"conformance spec {field} is not a SHA-256 hash")
    runtime_fingerprint = document.get("runtime_fingerprint")
    if (
        not isinstance(runtime_fingerprint, str)
        or not runtime_fingerprint.startswith("sha256:")
        or len(runtime_fingerprint) != 71
        or any(
            character not in "0123456789abcdef"
            for character in runtime_fingerprint.removeprefix("sha256:")
        )
    ):
        raise ConformanceError(
            "conformance spec runtime_fingerprint must be sha256:<64 lowercase hex>"
        )
    if not isinstance(document.get("scenario_id"), str) or not document.get(
        "scenario_id"
    ):
        raise ConformanceError("conformance spec requires one scenario_id")
    _two_distinct_strings(document.get("strategy_ids"), label="strategy_ids")
    _two_distinct_strings(
        document.get("monitor_profile_ids"), label="monitor_profile_ids"
    )
    identity_value = document.get("attacker_identity")
    if not isinstance(identity_value, Mapping):
        raise ConformanceError("attacker_identity must be an object")
    _exact_fields(
        identity_value,
        _ATTACKER_IDENTITY_FIELDS,
        label="attacker_identity",
    )
    try:
        identity = ModelIdentity.from_mapping(identity_value)
    except (TypeError, ValueError) as exc:
        raise ConformanceError(f"invalid attacker identity: {exc}") from exc
    _validate_approved_attacker_identity(identity)
    if identity.runtime_fingerprint != runtime_fingerprint:
        raise ConformanceError("attacker runtime differs from conformance spec")
    return identity


def _report_identity(value: Any, *, label: str) -> ModelIdentity:
    if not isinstance(value, Mapping):
        raise ConformanceError(f"{label} must be a model identity object")
    _exact_fields(value, _ATTACKER_IDENTITY_FIELDS, label=label)
    try:
        return ModelIdentity.from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise ConformanceError(f"invalid {label}: {exc}") from exc


def _validate_approved_monitor_identity(identity: ModelIdentity) -> None:
    if (
        identity.role != "monitor"
        or identity.implementation != "local_transformers"
        or identity.model_id != GRANITE_MONITOR_MODEL_ID
        or identity.model_revision != GRANITE_MONITOR_REVISION
        or identity.tokenizer_revision != GRANITE_MONITOR_REVISION
        or identity.checkpoint_fingerprint
        != GRANITE_MONITOR_CHECKPOINT_FINGERPRINT
        or identity.dtype != "bfloat16"
        or float(identity.temperature) != 0.0
        or float(identity.top_p) != 1.0
        or identity.max_new_tokens != 64
        or identity.reasoning_mode != "no_think"
        or identity.threshold != 0.5
    ):
        raise ConformanceError("report uses an unapproved Granite monitor identity")


def _validate_compatibility_report(value: Any) -> None:
    expected_fields = {
        "python_version",
        "package_version",
        "source_revision",
        "source_revision_verification",
        "benchmark_version",
        "wheel_sha256",
        "installed_payload_sha256",
        "distribution_verification",
        "wheel_artifact_verified",
        "suites",
        "defenses",
        "attacks",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ConformanceError("compatibility report fields are invalid")
    python_version = value.get("python_version")
    if (
        not isinstance(python_version, str)
        or not python_version.startswith("3.11.")
        or not python_version.removeprefix("3.11.").isdigit()
        or value.get("package_version") != AGENTDOJO_PACKAGE_VERSION
        or value.get("source_revision") != AGENTDOJO_SOURCE_REVISION
        or value.get("source_revision_verification")
        != "deployment_assertion_bound_to_published_wheel"
        or value.get("benchmark_version") != AGENTDOJO_BENCHMARK_VERSION
        or value.get("wheel_sha256") != EXPECTED_WHEEL_SHA256
        or not _is_raw_sha256(value.get("installed_payload_sha256"))
        or value.get("distribution_verification")
        not in {
            "installed_payload_against_frozen_wheel_payload_manifest",
            "verified_published_wheel_and_matching_installed_payload",
        }
        or not isinstance(value.get("wheel_artifact_verified"), bool)
        or value.get("defenses") != list(EXPECTED_DEFENSES)
        or value.get("attacks") != list(EXPECTED_ATTACKS)
    ):
        raise ConformanceError("compatibility report is not the pinned release")
    suites = value.get("suites")
    if not isinstance(suites, list) or len(suites) != len(AGENTDOJO_SUITES):
        raise ConformanceError("compatibility report suite census is invalid")
    for name, row in zip(AGENTDOJO_SUITES, suites, strict=True):
        counts = EXPECTED_RELEASE_COUNTS[name]
        expected = {
            "name": name,
            "benchmark_version": list(EXPECTED_INTERNAL_BENCHMARK_VERSIONS[name]),
            "user_task_count": counts["user_tasks"],
            "injection_task_count": counts["injection_tasks"],
            "tool_count": counts["tools"],
            "injection_vector_count": counts["vectors"],
        }
        if row != expected:
            raise ConformanceError("compatibility report suite census drifted")


def _validate_source_provenance(value: Any, *, source_tree_hash: str) -> None:
    expected_fields = {
        "code_revision",
        "code_dirty",
        "source_tree_hash",
        "package_version",
        "python_implementation",
        "python_version",
        "platform",
        "scheduler",
        "gpu_environment",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ConformanceError("source provenance fields are invalid")
    python_version = value.get("python_version")
    if (
        value.get("source_tree_hash") != source_tree_hash
        or not _is_raw_sha256(source_tree_hash)
        or not isinstance(value.get("code_revision"), str)
        or len(str(value.get("code_revision"))) != 40
        or any(
            character not in _HEX_SHA256
            for character in str(value.get("code_revision"))
        )
        or value.get("code_dirty") is not False
        or not isinstance(value.get("package_version"), str)
        or not value.get("package_version")
        or value.get("python_implementation") != "CPython"
        or not isinstance(python_version, str)
        or not python_version.startswith("3.11.")
        or not isinstance(value.get("platform"), str)
        or not value.get("platform")
    ):
        raise ConformanceError("source provenance does not match the frozen execution")
    scheduler = value.get("scheduler")
    scheduler_fields = {
        "kind",
        "job_id",
        "array_job_id",
        "array_task_id",
        "partition",
        "queue",
        "node_list",
        "node_file",
        "cpus_per_task",
        "job_gpus",
        "slurm_job_id",
        "pbs_job_id",
    }
    if (
        not isinstance(scheduler, Mapping)
        or set(scheduler) != scheduler_fields
        or scheduler.get("kind") not in {"pbs", "slurm"}
        or not isinstance(scheduler.get("job_id"), str)
        or not scheduler.get("job_id")
        or scheduler.get("array_task_id") is not None
    ):
        raise ConformanceError("source provenance lacks a non-array scheduler job")
    gpu = value.get("gpu_environment")
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "cuda_visible_devices",
        "nvidia_visible_devices",
    }:
        raise ConformanceError("source provenance GPU environment is invalid")


def _validate_memory_evidence(
    value: Any, *, profile_ids: tuple[str, str]
) -> None:
    stages = (
        "before_model_load",
        "after_attacker_load",
        f"after_monitor_load:{profile_ids[0]}",
        f"after_monitor_load:{profile_ids[1]}",
        "after_protocol_checks",
    )
    if not isinstance(value, list) or len(value) != len(stages):
        raise ConformanceError("passed report lacks five CUDA memory stages")
    previous_allocated: int | None = None
    previous_peak: int | None = None
    expected_name: str | None = None
    expected_total: int | None = None
    for index, (row, stage) in enumerate(zip(value, stages, strict=True)):
        if not isinstance(row, Mapping) or set(row) != {"stage", "devices"}:
            raise ConformanceError("CUDA memory stage fields are invalid")
        devices = row.get("devices")
        if row.get("stage") != stage or not isinstance(devices, list) or len(devices) != 1:
            raise ConformanceError("CUDA memory stages are incomplete or out of order")
        device = devices[0]
        expected_fields = {
            "device",
            "device_name",
            "total_memory_bytes",
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        }
        if not isinstance(device, Mapping) or set(device) != expected_fields:
            raise ConformanceError("CUDA memory device fields are invalid")
        name = device.get("device_name")
        total = device.get("total_memory_bytes")
        metrics = [
            device.get("allocated_bytes"),
            device.get("reserved_bytes"),
            device.get("peak_allocated_bytes"),
            device.get("peak_reserved_bytes"),
        ]
        if (
            device.get("device") != "cuda:0"
            or not isinstance(name, str)
            or "H200" not in name.upper()
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total < 100 * 1024**3
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in metrics
            )
        ):
            raise ConformanceError("CUDA memory evidence is not an H200 allocation")
        allocated, reserved, peak_allocated, peak_reserved = metrics
        if reserved < allocated or peak_allocated < allocated or peak_reserved < reserved:
            raise ConformanceError("CUDA allocation and peak counters are inconsistent")
        if expected_name is None:
            expected_name, expected_total = name, total
        elif name != expected_name or total != expected_total:
            raise ConformanceError("CUDA device identity changed during conformance")
        if previous_allocated is not None:
            if index <= 3 and allocated <= previous_allocated:
                raise ConformanceError("CUDA allocation did not increase after model load")
            if index == 4 and allocated < previous_allocated:
                raise ConformanceError("retained model allocation fell before final probe")
            if previous_peak is not None and peak_allocated < previous_peak:
                raise ConformanceError("CUDA peak allocation decreased")
        previous_allocated = allocated
        previous_peak = peak_allocated


def _validate_local_call_metadata(
    value: Any,
    *,
    identity: ModelIdentity,
    role: str,
    structured_chat: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise ConformanceError(f"{role} call lacks local-model metadata")
    if (
        value.get("client") != "local_transformers"
        or value.get("model_role") != role
        or value.get("model_id") != identity.model_id
        or value.get("requested_model_revision") != identity.model_revision
        or value.get("model_revision") != identity.model_revision
        or value.get("requested_tokenizer_revision") != identity.tokenizer_revision
        or value.get("tokenizer_revision") != identity.tokenizer_revision
        or value.get("local_checkpoint_fingerprint")
        != identity.checkpoint_fingerprint
        or value.get("local_checkpoint_verification_mode")
        not in {"full_tree_sha256_audit", "full_tree_sha256_initialization"}
        or not _is_raw_sha256(value.get("local_checkpoint_manifest_hash"))
        or not isinstance(value.get("local_checkpoint_path"), str)
        or not value.get("local_checkpoint_path")
        or value.get("dtype") != identity.dtype
        or value.get("device") != "cuda:0"
        or float(value.get("temperature", -1.0)) != float(identity.temperature)
        or float(value.get("top_p", -1.0)) != float(identity.top_p)
        or value.get("batch_size") != 1
        or value.get("external_api_calls") != 0
        or value.get("local_files_only") is not True
        or not isinstance(value.get("gpu_name"), str)
        or "H200" not in str(value.get("gpu_name")).upper()
    ):
        raise ConformanceError(f"{role} call metadata is not the approved local runtime")
    if structured_chat and (
        value.get("input_mode") != "structured_chat"
        or not isinstance(value.get("input_messages"), list)
        or not _is_raw_sha256(value.get("input_messages_hash"))
    ):
        raise ConformanceError("monitor call lacks exact structured-chat metadata")


def validate_conformance_report(document: Mapping[str, Any]) -> None:
    """Validate the strict claim boundary and self-hash of a report."""

    _exact_fields(document, _REPORT_FIELDS, label="conformance report")
    if document.get("schema_version") != CONFORMANCE_REPORT_SCHEMA_VERSION:
        raise ConformanceError("unsupported conformance report schema")
    fixed = {
        "artifact_class": CONFORMANCE_ARTIFACT_CLASS,
        "evidence_class": CONFORMANCE_EVIDENCE_CLASS,
        "scientific_evidence_eligible": False,
        "claim_boundary": CONFORMANCE_CLAIM_BOUNDARY,
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "dataset_split": "development",
        "test_outcomes_inspected": False,
        "pair_selection_eligible": False,
        "development_outcomes_generated": True,
        "external_api_calls": 0,
    }
    for field, expected in fixed.items():
        if document.get(field) != expected:
            raise ConformanceError(f"conformance report {field} is invalid")
    if document.get("status") not in {"passed", "failed"}:
        raise ConformanceError("conformance report status is invalid")
    checks = document.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != {
        "attacker",
        "monitor",
        "lifecycle",
    }:
        raise ConformanceError("conformance report checks are invalid")
    if any(not isinstance(checks[key], list) for key in checks):
        raise ConformanceError("conformance report check groups must be lists")
    flattened = [item for key in checks for item in checks[key]]
    if any(
        not isinstance(item, Mapping)
        or item.get("status") not in {"passed", "failed"}
        for item in flattened
    ):
        raise ConformanceError("conformance report contains an invalid check")
    passed = sum(item["status"] == "passed" for item in flattened)
    failed = len(flattened) - passed
    expected_summary = {
        "total_checks": len(flattened),
        "passed_checks": passed,
        "failed_checks": failed,
    }
    if document.get("summary") != expected_summary:
        raise ConformanceError("conformance report summary is inconsistent")
    errors = document.get("errors")
    if not isinstance(errors, list):
        raise ConformanceError("conformance report errors must be a list")
    should_pass = failed == 0 and not errors
    if (document.get("status") == "passed") != should_pass:
        raise ConformanceError("conformance report status is inconsistent")
    if document.get("status") == "passed":
        if not _is_raw_sha256(document.get("conformance_spec_hash")):
            raise ConformanceError("passed report lacks a conformance spec hash")
        if not _is_raw_sha256(document.get("source_tree_hash")):
            raise ConformanceError("passed report lacks its frozen source-tree hash")
        upstream = document.get("upstream_artifacts")
        if (
            not isinstance(upstream, Mapping)
            or set(upstream)
            != {
                "catalog_hash",
                "split_manifest_hash",
                "candidate_strategy_catalog_hash",
            }
            or any(not _is_raw_sha256(value) for value in upstream.values())
        ):
            raise ConformanceError("passed report upstream artifacts are invalid")
        scenario = document.get("scenario")
        if (
            not isinstance(scenario, Mapping)
            or set(scenario)
            != {
                "scenario_id",
                "suite",
                "structural_group_id",
                "dataset_split",
                "public_scenario_hash",
            }
            or not isinstance(scenario.get("scenario_id"), str)
            or not scenario.get("scenario_id")
            or scenario.get("suite") not in AGENTDOJO_SUITES
            or not isinstance(scenario.get("structural_group_id"), str)
            or not scenario.get("structural_group_id")
            or scenario.get("dataset_split") != "development"
            or not _is_raw_sha256(scenario.get("public_scenario_hash"))
        ):
            raise ConformanceError("passed report scenario identity is invalid")
        strategies = document.get("selected_strategies")
        if not isinstance(strategies, list) or len(strategies) != 2:
            raise ConformanceError("passed report requires exactly two strategies")
        strategy_ids: list[str] = []
        for strategy in strategies:
            if (
                not isinstance(strategy, Mapping)
                or set(strategy)
                != {
                    "strategy_id",
                    "strategy_row_hash",
                    "call_count",
                    "call_sequence_hash",
                }
                or not isinstance(strategy.get("strategy_id"), str)
                or not strategy.get("strategy_id")
                or not _is_raw_sha256(strategy.get("strategy_row_hash"))
                or isinstance(strategy.get("call_count"), bool)
                or not isinstance(strategy.get("call_count"), int)
                or strategy["call_count"] <= 0
                or not _is_raw_sha256(strategy.get("call_sequence_hash"))
            ):
                raise ConformanceError("passed report strategy binding is invalid")
            strategy_ids.append(str(strategy["strategy_id"]))
        if len(set(strategy_ids)) != 2:
            raise ConformanceError("passed report strategy IDs are not distinct")

        attacker_identity = _report_identity(
            document.get("attacker_identity"), label="report attacker_identity"
        )
        _validate_approved_attacker_identity(attacker_identity)
        profiles = document.get("selected_monitor_profiles")
        if not isinstance(profiles, list) or len(profiles) != 2:
            raise ConformanceError("passed report requires exactly two monitor profiles")
        profile_ids: list[str] = []
        monitor_identities: dict[str, ModelIdentity] = {}
        for profile in profiles:
            if (
                not isinstance(profile, Mapping)
                or set(profile)
                != {
                    "profile_id",
                    "profile_hash",
                    "profile_row_hash",
                    "family",
                    "monitor_identity",
                }
                or not isinstance(profile.get("profile_id"), str)
                or not profile.get("profile_id")
                or not _is_raw_sha256(profile.get("profile_hash"))
                or not _is_raw_sha256(profile.get("profile_row_hash"))
                or profile.get("family") != GRANITE_MONITOR_FAMILY
            ):
                raise ConformanceError("passed report monitor binding is invalid")
            profile_id = str(profile["profile_id"])
            identity = _report_identity(
                profile.get("monitor_identity"),
                label=f"monitor identity {profile_id}",
            )
            _validate_approved_monitor_identity(identity)
            profile_ids.append(profile_id)
            monitor_identities[profile_id] = identity
        if len(set(profile_ids)) != 2:
            raise ConformanceError("passed report monitor profile IDs are not distinct")
        runtime_fingerprints = {
            attacker_identity.runtime_fingerprint,
            *(identity.runtime_fingerprint for identity in monitor_identities.values()),
        }
        learned_runtime = document.get("learned_runtime")
        if not isinstance(learned_runtime, Mapping):
            raise ConformanceError("passed report lacks learned-runtime provenance")
        try:
            validate_learned_runtime_provenance(
                learned_runtime,
                expected_runtime_fingerprints=runtime_fingerprints,
            )
        except RuntimeIntegrityError as exc:
            raise ConformanceError(f"passed report runtime provenance is invalid: {exc}") from exc
        _validate_compatibility_report(document.get("compatibility"))
        _validate_source_provenance(
            document.get("source_provenance"),
            source_tree_hash=str(document["source_tree_hash"]),
        )
        topology = document.get("client_topology")
        expected_topology = {
            "process_count": 1,
            "attacker_client_count": 1,
            "monitor_client_count": 2,
            "shared_monitor_client": False,
            "monitor_profile_count": 2,
            "attacker_device": "cuda:0",
            "monitor_device": "cuda:0",
            "gpu_device_count": 1,
            "load_order": [
                "attacker",
                f"monitor:{profile_ids[0]}",
                f"monitor:{profile_ids[1]}",
            ],
            "simultaneously_retained_client_count": 3,
        }
        if topology != expected_topology:
            raise ConformanceError(
                "passed conformance report has the wrong client topology"
            )
        _validate_memory_evidence(
            document.get("memory_evidence"),
            profile_ids=(profile_ids[0], profile_ids[1]),
        )

        attacker_checks = checks["attacker"]
        expected_attacker = (
            ("attacker:probe_selection", "probe_selection", 0),
            ("attacker:hidden_state_prediction", "hidden_state_prediction", 1),
            ("attacker:final_plan_selection", "final_plan_selection", 2),
        )
        if len(attacker_checks) != len(expected_attacker):
            raise ConformanceError("passed report lacks exact attacker contract coverage")
        for check, (check_id, contract, call_index) in zip(
            attacker_checks, expected_attacker, strict=True
        ):
            if (
                set(check)
                != {"check_id", "contract", "model_call_index", "status", "output"}
                or check.get("check_id") != check_id
                or check.get("contract") != contract
                or check.get("model_call_index") != call_index
                or check.get("status") != "passed"
                or not isinstance(check.get("output"), Mapping)
                or check["output"].get("valid") is not True
            ):
                raise ConformanceError("passed report has an invalid attacker contract check")

        monitor_checks = checks["monitor"]
        expected_monitor_cells = [
            (profile_id, strategy_id, call_index)
            for profile_id in profile_ids
            for strategy_id, strategy in zip(strategy_ids, strategies, strict=True)
            for call_index in range(int(strategy["call_count"]))
        ]
        if len(monitor_checks) != len(expected_monitor_cells):
            raise ConformanceError("passed report lacks full monitor cell coverage")
        for check, (profile_id, strategy_id, call_index) in zip(
            monitor_checks, expected_monitor_cells, strict=True
        ):
            expected_id = f"monitor:{profile_id}:{strategy_id}:call_{call_index}"
            if (
                set(check)
                != {
                    "check_id",
                    "profile_id",
                    "strategy_id",
                    "call_index",
                    "seed",
                    "status",
                    "evaluation",
                    "failure_provenance",
                }
                or check.get("check_id") != expected_id
                or check.get("profile_id") != profile_id
                or check.get("strategy_id") != strategy_id
                or check.get("call_index") != call_index
                or check.get("status") != "passed"
                or check.get("failure_provenance") is not None
                or not isinstance(check.get("evaluation"), Mapping)
            ):
                raise ConformanceError("passed report has an invalid monitor cell")
            evaluation = check["evaluation"]
            if (
                set(evaluation)
                != {
                    "plan_id",
                    "profile_id",
                    "monitor_family",
                    "decision",
                    "score",
                    "policy_epoch",
                    "provenance",
                }
                or evaluation.get("plan_id") != strategy_id
                or evaluation.get("profile_id") != profile_id
                or evaluation.get("monitor_family") != GRANITE_MONITOR_FAMILY
                or evaluation.get("decision") not in {"allow", "block"}
                or evaluation.get("score") not in {0.0, 1.0}
                or evaluation.get("policy_epoch") != 0
                or not isinstance(evaluation.get("provenance"), Mapping)
            ):
                raise ConformanceError("passed report monitor evaluation is invalid")
            provenance = evaluation["provenance"]
            if set(provenance) != {
                "monitor_input_hash",
                "monitor_profile_hash",
                "frozen_profile_hash",
                "rendered_prompt_hash",
                "raw_response_hash",
                "seed",
                "model_metadata",
                "model_call",
            }:
                raise ConformanceError("passed report monitor provenance fields are invalid")
            model_call = provenance.get("model_call")
            selected_profile = next(
                row for row in profiles if row["profile_id"] == profile_id
            )
            if (
                not isinstance(model_call, Mapping)
                or not _is_raw_sha256(provenance.get("monitor_input_hash"))
                or not _is_raw_sha256(provenance.get("monitor_profile_hash"))
                or provenance.get("frozen_profile_hash")
                != selected_profile["profile_hash"]
                or not _is_raw_sha256(provenance.get("rendered_prompt_hash"))
                or provenance.get("raw_response_hash")
                != model_call.get("raw_response_hash")
                or provenance.get("seed") != check.get("seed")
                or provenance.get("model_metadata") != model_call.get("metadata")
            ):
                raise ConformanceError("passed report monitor provenance binding is invalid")
            try:
                _validate_monitor_call_record(
                    model_call, label=f"conformance {expected_id}"
                )
            except ResultValidationError as exc:
                raise ConformanceError(
                    f"passed report monitor provenance is invalid: {exc}"
                ) from exc
            _validate_local_call_metadata(
                model_call.get("metadata"),
                identity=monitor_identities[profile_id],
                role="monitor",
                structured_chat=True,
            )

        lifecycle = checks["lifecycle"]
        if lifecycle != [{"check_id": "attacker:retire", "status": "passed"}]:
            raise ConformanceError("passed report lacks exact attacker retirement")
        check_ids = [str(check.get("check_id")) for check in flattened]
        if len(check_ids) != len(set(check_ids)):
            raise ConformanceError("passed report repeats check IDs")
        attacker_provenance = document.get("attacker_provenance")
        if (
            not isinstance(attacker_provenance, Mapping)
            or set(attacker_provenance)
            != {
                "protocol_revision",
                "immutable_model_revision",
                "calls",
                "retired",
            }
            or attacker_provenance.get("protocol_revision")
            != StructuredControlledAttacker.protocol_revision
            or attacker_provenance.get("immutable_model_revision")
            != attacker_identity.model_revision
            or attacker_provenance.get("retired") is not True
            or not isinstance(attacker_provenance.get("calls"), list)
            or len(attacker_provenance["calls"]) != 3
        ):
            raise ConformanceError("passed report attacker provenance is invalid")
        expected_phases = ("probe", "prediction", "final")
        for index, (call, phase) in enumerate(
            zip(attacker_provenance["calls"], expected_phases, strict=True)
        ):
            try:
                _validate_model_call_record(call, label=f"conformance attacker call {index}")
            except ResultValidationError as exc:
                raise ConformanceError(
                    f"passed report attacker provenance is invalid: {exc}"
                ) from exc
            if (
                call.get("phase") != phase
                or call.get("call_index") != index
                or call.get("error") is not None
            ):
                raise ConformanceError("passed report attacker call order is invalid")
            _validate_local_call_metadata(
                call.get("metadata"),
                identity=attacker_identity,
                role="attacker",
                structured_chat=False,
            )
    recorded_hash = document.get("conformance_report_hash")
    if not isinstance(recorded_hash, str) or recorded_hash != stable_hash(
        _without_hash(document, "conformance_report_hash")
    ):
        raise ConformanceError("conformance_report_hash is invalid")


def _by_id(
    rows: Sequence[Mapping[str, Any]], field: str, identifier: str
) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get(field)) == identifier]
    if len(matches) != 1:
        raise ConformanceError(
            f"expected exactly one {field}={identifier!r}, found {len(matches)}"
        )
    return matches[0]


def _compatibility_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return dict(converted)
    raise ConformanceError("compatibility report cannot be serialized")


def _error(
    *, stage: str, check_id: str | None, error: BaseException | str
) -> dict[str, Any]:
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        message = str(error)
    else:
        error_type = "ProtocolFailure"
        message = str(error)
    return {
        "stage": stage,
        "check_id": check_id,
        "type": error_type,
        "message": message,
    }


def _finalize_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = {**dict(payload), "conformance_report_hash": stable_hash(payload)}
    validate_conformance_report(report)
    return report


def _base_report(
    *,
    conformance_spec_hash: str | None,
    source_tree_hash: str | None,
    upstream_artifacts: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONFORMANCE_REPORT_SCHEMA_VERSION,
        "artifact_class": CONFORMANCE_ARTIFACT_CLASS,
        "evidence_class": CONFORMANCE_EVIDENCE_CLASS,
        "scientific_evidence_eligible": False,
        "claim_boundary": CONFORMANCE_CLAIM_BOUNDARY,
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "dataset_split": "development",
        "test_outcomes_inspected": False,
        "pair_selection_eligible": False,
        "development_outcomes_generated": True,
        "conformance_spec_hash": conformance_spec_hash,
        "source_tree_hash": source_tree_hash,
        "upstream_artifacts": dict(upstream_artifacts),
        "scenario": None,
        "selected_strategies": [],
        "selected_monitor_profiles": [],
        "attacker_identity": None,
        "learned_runtime": None,
        "compatibility": None,
        "source_provenance": dict(source_provenance),
        "client_topology": {
            "process_count": 1,
            "attacker_client_count": 0,
            "monitor_client_count": 0,
            "shared_monitor_client": False,
            "monitor_profile_count": 0,
            "attacker_device": None,
            "monitor_device": None,
            "gpu_device_count": 0,
        },
        "memory_evidence": [],
        "checks": {"attacker": [], "monitor": [], "lifecycle": []},
        "attacker_provenance": None,
        "errors": [],
        "summary": {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
        },
        "status": "failed",
        "external_api_calls": 0,
    }


def _source_provenance(dependencies: ConformanceDependencies) -> dict[str, Any]:
    try:
        value = dependencies.provenance_factory()
    except Exception as exc:
        raise ConformanceError(f"cannot collect source provenance: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConformanceError("source provenance collector returned a non-object")
    materialized = dict(value)
    if "collection_error" in materialized:
        raise ConformanceError("source provenance collection was incomplete")
    return materialized


def _validate_bindings(
    *,
    spec: Mapping[str, Any],
    catalog: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    strategy_catalog: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], Mapping[str, Any]],
    tuple[Mapping[str, Any], Mapping[str, Any]],
]:
    expected = {
        "catalog_hash": catalog.get("catalog_hash"),
        "split_manifest_hash": split_manifest.get("split_manifest_hash"),
        "candidate_strategy_catalog_hash": strategy_catalog.get(
            "candidate_strategy_catalog_hash"
        ),
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            raise ConformanceError(f"conformance spec {field} binds another artifact")
    if (
        strategy_catalog.get("catalog_hash") != catalog.get("catalog_hash")
        or strategy_catalog.get("split_manifest_hash")
        != split_manifest.get("split_manifest_hash")
    ):
        raise ConformanceError("frozen inputs do not share one upstream chain")

    scenario = _by_id(
        tuple(catalog["scenarios"]),
        "scenario_id",
        str(spec["scenario_id"]),
    )
    development_entry = split_manifest["splits"]["development"]
    development_scenarios = {
        str(item) for item in development_entry.get("scenario_ids", ())
    }
    if (
        scenario.get("dataset_split") != "development"
        or str(scenario["scenario_id"]) not in development_scenarios
    ):
        raise ConformanceError(
            "conformance scenario is not in the frozen development split"
        )

    strategy_ids = _two_distinct_strings(
        spec.get("strategy_ids"), label="strategy_ids"
    )
    strategies = tuple(
        _by_id(tuple(strategy_catalog["strategies"]), "strategy_id", identifier)
        for identifier in strategy_ids
    )
    profile_ids = _two_distinct_strings(
        spec.get("monitor_profile_ids"), label="monitor_profile_ids"
    )
    profiles = tuple(
        _by_id(
            tuple(strategy_catalog["monitor_profiles"]),
            "profile_id",
            identifier,
        )
        for identifier in profile_ids
    )
    for profile in profiles:
        if profile.get("family") != GRANITE_MONITOR_FAMILY:
            raise ConformanceError(
                "both selected monitor profiles must use Granite Guardian 4.1"
            )
        if profile.get("runtime_fingerprint") != spec.get("runtime_fingerprint"):
            raise ConformanceError(
                "selected monitor profile runtime differs from the conformance spec"
            )
        _validate_approved_monitor_identity(_monitor_identity(profile))
    compatible, reason = _monitor_pair_compatibility(profiles[0], profiles[1])
    if not compatible:
        raise ConformanceError(
            f"selected monitor profiles cannot share one client: {reason}"
        )
    return scenario, strategies, profiles  # type: ignore[return-value]


def _monitor_identity(profile: Mapping[str, Any]) -> ModelIdentity:
    decoding = profile["decoding"]
    return ModelIdentity(
        role="monitor",
        implementation=str(profile["implementation"]),
        model_id=str(profile["model_id"]),
        model_revision=str(profile["model_revision"]),
        tokenizer_revision=str(profile["tokenizer_revision"]),
        checkpoint_fingerprint=str(profile["checkpoint_fingerprint"]),
        runtime_fingerprint=str(profile["runtime_fingerprint"]),
        prompt_hash=str(profile["prompt_hash"]),
        policy_hash=str(profile["policy_hash"]),
        threshold=float(profile["threshold"]),
        reasoning_mode=str(profile["reasoning_mode"]),
        dtype=str(profile["dtype"]),
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        max_new_tokens=int(decoding["max_new_tokens"]),
    )


def _public_scenario(
    *,
    scenario: Mapping[str, Any],
    strategies: Sequence[Mapping[str, Any]],
    suite: Any,
    compat: Any,
) -> tuple[
    PublicScenario,
    tuple[Any, Any],
    dict[str, str],
]:
    user_task = compat.get_user_task(suite, str(scenario["user_task_id"]))
    injection_task = compat.get_injection_task(
        suite, str(scenario["injection_task_id"])
    )
    user_prompt = str(getattr(user_task, "PROMPT", ""))
    injection_goal = str(getattr(injection_task, "GOAL", ""))
    if not user_prompt or not injection_goal:
        raise ConformanceError("frozen tasks expose an empty prompt or injection goal")
    label_to_strategy = {
        f"candidate_{index}": str(strategy["strategy_id"])
        for index, strategy in enumerate(strategies)
    }
    plans = tuple(
        _trusted_plan(
            plan_id=f"candidate_{index}",
            strategy=strategy,
            scenario=scenario,
            user_prompt=user_prompt,
            injection_goal=injection_goal,
        )
        for index, strategy in enumerate(strategies)
    )
    schemas = canonicalize_tool_schemas(compat.canonical_tool_schemas(suite))
    schema_names = {schema.name for schema in schemas}
    for plan in plans:
        unknown = sorted(
            {call.function for call in plan.calls if call.function not in schema_names}
        )
        if unknown:
            raise ConformanceError(
                f"strategy {label_to_strategy[plan.plan_id]!r} references "
                f"unknown tools {unknown}"
            )
    candidate_public_plans: list[PublicPlan] = []
    probe_public_plans: list[PublicPlan] = []
    for index, (strategy, plan) in enumerate(zip(strategies, plans, strict=True)):
        description = str(
            strategy.get("public_description", f"candidate strategy {index}")
        )
        steps = tuple(
            call.public_dict(protect_arguments=True) for call in plan.calls
        )
        candidate_public_plans.append(
            PublicPlan(
                f"candidate_{index}",
                f"candidate_{index}",
                description,
                "candidate",
                steps,
            )
        )
        probe_public_plans.append(
            PublicPlan(
                f"probe_candidate_{index}",
                f"candidate_{index}",
                description,
                "probe",
                steps,
            )
        )
    public_scenario = PublicScenario(
        scenario_id=str(scenario["scenario_id"]),
        suite=str(scenario["suite"]),
        user_prompt=user_prompt,
        tool_schemas=schemas,
        candidate_plans=tuple(candidate_public_plans + probe_public_plans),
        structural_group_id=str(scenario["structural_group_id"]),
        dataset_split="development",
        public_environment=public_value(
            {
                "initial_environment_hash": scenario["initial_environment_hash"],
                "injection_vector_id": scenario["injection_vector_id"],
                "candidate_mapping_revision": stable_hash(label_to_strategy),
            }
        ),
    )
    return public_scenario, plans, label_to_strategy  # type: ignore[return-value]


def _attacker_checks(
    attacker: StructuredControlledAttacker,
    scenario: PublicScenario,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schedule = SeedSchedule(
        scenario.scenario_id, attacker.immutable_model_revision, 0
    )
    operations: tuple[tuple[str, Callable[[], Any]], ...] = (
        (
            "probe_selection",
            lambda: attacker.choose_probe(
                scenario, (), probe_index=0, seed=schedule.probe(0)
            ),
        ),
        (
            "hidden_state_prediction",
            lambda: attacker.predict_hidden_state(
                scenario, (), seed=schedule.prediction()
            ),
        ),
        (
            "final_plan_selection",
            lambda: attacker.choose_final(scenario, (), seed=schedule.final()),
        ),
    )
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for call_index, (contract, operation) in enumerate(operations):
        check_id = f"attacker:{contract}"
        try:
            result = operation()
            valid = bool(getattr(result, "valid", False))
            output = (
                {
                    "plan_id": getattr(result, "plan_id", None),
                    "valid": valid,
                    "error": getattr(result, "error", None),
                }
                if contract != "hidden_state_prediction"
                else {
                    "prediction": getattr(result, "prediction", None),
                    "posterior": (
                        dict(result.posterior)
                        if getattr(result, "posterior", None) is not None
                        else None
                    ),
                    "valid": valid,
                    "error": getattr(result, "error", None),
                }
            )
            check = {
                "check_id": check_id,
                "contract": contract,
                "model_call_index": call_index,
                "status": "passed" if valid else "failed",
                "output": output,
            }
            checks.append(check)
            if not valid:
                errors.append(
                    _error(
                        stage="attacker_protocol",
                        check_id=check_id,
                        error=str(getattr(result, "error", "invalid output")),
                    )
                )
        except Exception as exc:  # retain an unexpected adapter failure too
            checks.append(
                {
                    "check_id": check_id,
                    "contract": contract,
                    "model_call_index": call_index,
                    "status": "failed",
                    "output": None,
                }
            )
            errors.append(
                _error(stage="attacker_protocol", check_id=check_id, error=exc)
            )
    return checks, errors


def _monitor_checks(
    *,
    profiles: Sequence[Mapping[str, Any]],
    strategies: Sequence[Mapping[str, Any]],
    plans: Sequence[Any],
    public_scenario: PublicScenario,
    label_to_strategy: Mapping[str, str],
    monitor_clients: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schema_by_name = {schema.name: schema for schema in public_scenario.tool_schemas}
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        monitor = _monitor(
            profile,
            scenario_id=public_scenario.scenario_id,
            label_to_strategy=label_to_strategy,
            model_client=monitor_clients[profile_id],
        )
        schedule = SeedSchedule(
            public_scenario.scenario_id, str(profile["model_revision"]), 0
        )
        for strategy, plan in zip(strategies, plans, strict=True):
            strategy_id = str(strategy["strategy_id"])
            for call_index, call in enumerate(plan.calls):
                check_id = (
                    f"monitor:{profile_id}:{strategy_id}:call_{call_index}"
                )
                monitor_input = MonitorInput(
                    frozen_user_goal=public_scenario.user_prompt,
                    public_conversation_summary=public_value(
                        {
                            "scenario_id": public_scenario.scenario_id,
                            "summary": "development checkpoint conformance",
                        }
                    ),
                    proposed_call=call,
                    tool_schema=schema_by_name[call.function],
                )
                seed = schedule.monitor(call_index)
                try:
                    evaluation = monitor.evaluate(
                        monitor_input, plan_id=strategy_id, seed=seed
                    )
                    binary = evaluation.decision in {"allow", "block"}
                    checks.append(
                        {
                            "check_id": check_id,
                            "profile_id": profile_id,
                            "strategy_id": strategy_id,
                            "call_index": call_index,
                            "seed": seed,
                            "status": "passed" if binary else "failed",
                            "evaluation": evaluation.to_trusted_dict(),
                            "failure_provenance": None,
                        }
                    )
                    if not binary:
                        errors.append(
                            _error(
                                stage="monitor_protocol",
                                check_id=check_id,
                                error="monitor returned a nonbinary decision",
                            )
                        )
                except Exception as exc:
                    failure = getattr(monitor, "last_failure_provenance", None)
                    checks.append(
                        {
                            "check_id": check_id,
                            "profile_id": profile_id,
                            "strategy_id": strategy_id,
                            "call_index": call_index,
                            "seed": seed,
                            "status": "failed",
                            "evaluation": None,
                            "failure_provenance": (
                                dict(failure)
                                if isinstance(failure, Mapping)
                                else failure
                            ),
                        }
                    )
                    errors.append(
                        _error(stage="monitor_protocol", check_id=check_id, error=exc)
                    )
    return checks, errors


def execute_controlled_conformance(
    *,
    spec: Mapping[str, Any],
    catalog: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    strategy_catalog: Mapping[str, Any],
    dependency_lock_path: Path | str,
    attacker_checkpoint: Path | str,
    monitor_checkpoint: Path | str,
    model_cache: Path | str | None = None,
    attacker_device: str = "cuda",
    monitor_device: str = "cuda",
    dependencies: ConformanceDependencies | None = None,
) -> dict[str, Any]:
    """Execute the controlled conformance matrix without running a benchmark."""

    deps = dependencies or ConformanceDependencies()
    if (
        attacker_device != monitor_device
        or not attacker_device.startswith("cuda")
    ):
        raise ConformanceError(
            "controlled conformance must retain all three clients on one CUDA device"
        )
    attacker_identity = validate_conformance_spec(spec)
    validate_catalog(catalog)
    validate_split_manifest(split_manifest, catalog=catalog)
    validate_candidate_strategy_catalog(strategy_catalog)
    scenario, strategies, profiles = _validate_bindings(
        spec=spec,
        catalog=catalog,
        split_manifest=split_manifest,
        strategy_catalog=strategy_catalog,
    )
    learned_runtime = deps.runtime_validator(
        dependency_lock_path=dependency_lock_path,
        fixture_mode=False,
        runtime_fingerprints={str(spec["runtime_fingerprint"])},
    )
    source_provenance = _source_provenance(deps)
    if source_provenance.get("source_tree_hash") != spec.get("source_tree_hash"):
        raise ConformanceError(
            "observed source-tree hash differs from the frozen conformance spec"
        )
    adapter = deps.compat
    if adapter is None:
        from . import compat as adapter

    compatibility = adapter.assert_compatible(
        AGENTDOJO_SOURCE_REVISION, AGENTDOJO_BENCHMARK_VERSION
    )
    suite = adapter.load_suite(
        str(scenario["suite"]),
        deployment_source_revision=AGENTDOJO_SOURCE_REVISION,
        benchmark_version=AGENTDOJO_BENCHMARK_VERSION,
    )
    adapter.validate_frozen_scenario_row(suite, scenario)
    public_scenario, plans, label_to_strategy = _public_scenario(
        scenario=scenario,
        strategies=strategies,
        suite=suite,
        compat=adapter,
    )

    memory_probe = deps.memory_probe or _cuda_memory_probe
    memory_evidence = [
        dict(
            memory_probe(
                stage="before_model_load",
                devices=(attacker_device, monitor_device),
            )
        )
    ]
    attacker_client = deps.model_client_factory(
        attacker_identity,
        checkpoint_path=attacker_checkpoint,
        cache_dir=model_cache,
        device=attacker_device,
    )
    memory_evidence.append(
        dict(
            memory_probe(
                stage="after_attacker_load",
                devices=(attacker_device, monitor_device),
            )
        )
    )
    monitor_clients: dict[str, Any] = {}
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        monitor_clients[profile_id] = deps.model_client_factory(
            _monitor_identity(profile),
            checkpoint_path=monitor_checkpoint,
            cache_dir=model_cache,
            device=monitor_device,
        )
        memory_evidence.append(
            dict(
                memory_probe(
                    stage=f"after_monitor_load:{profile_id}",
                    devices=(attacker_device, monitor_device),
                )
            )
        )
    if monitor_clients[str(profiles[0]["profile_id"])] is monitor_clients[
        str(profiles[1]["profile_id"])
    ]:
        raise ConformanceError(
            "pair-observation conformance requires two distinct retained "
            "Granite client instances"
        )
    if any(attacker_client is client for client in monitor_clients.values()):
        raise ConformanceError(
            "Qwen and Granite must be distinct retained client instances"
        )
    attacker = StructuredControlledAttacker(
        attacker_client,
        immutable_model_revision=attacker_identity.model_revision,
        max_tokens=attacker_identity.max_new_tokens,
    )
    attacker_checks, attacker_errors = _attacker_checks(
        attacker, public_scenario
    )
    monitor_checks, monitor_errors = _monitor_checks(
        profiles=profiles,
        strategies=strategies,
        plans=plans,
        public_scenario=public_scenario,
        label_to_strategy=label_to_strategy,
        monitor_clients=monitor_clients,
    )
    memory_evidence.append(
        dict(
            memory_probe(
                stage="after_protocol_checks",
                devices=(attacker_device, monitor_device),
            )
        )
    )
    lifecycle_checks: list[dict[str, Any]] = []
    lifecycle_errors: list[dict[str, Any]] = []
    try:
        attacker.retire()
        lifecycle_checks.append(
            {"check_id": "attacker:retire", "status": "passed"}
        )
    except Exception as exc:
        lifecycle_checks.append(
            {"check_id": "attacker:retire", "status": "failed"}
        )
        lifecycle_errors.append(
            _error(stage="model_lifecycle", check_id="attacker:retire", error=exc)
        )
    errors = attacker_errors + monitor_errors + lifecycle_errors
    all_checks = attacker_checks + monitor_checks + lifecycle_checks
    passed_checks = sum(check["status"] == "passed" for check in all_checks)
    payload = _base_report(
        conformance_spec_hash=str(spec["conformance_spec_hash"]),
        source_tree_hash=str(spec["source_tree_hash"]),
        upstream_artifacts={
            "catalog_hash": catalog["catalog_hash"],
            "split_manifest_hash": split_manifest["split_manifest_hash"],
            "candidate_strategy_catalog_hash": strategy_catalog[
                "candidate_strategy_catalog_hash"
            ],
        },
        source_provenance=source_provenance,
    )
    payload.update(
        {
            "scenario": {
                "scenario_id": scenario["scenario_id"],
                "suite": scenario["suite"],
                "structural_group_id": scenario["structural_group_id"],
                "dataset_split": scenario["dataset_split"],
                "public_scenario_hash": public_scenario.public_hash,
            },
            "selected_strategies": [
                {
                    "strategy_id": strategy["strategy_id"],
                    "strategy_row_hash": stable_hash(strategy),
                    "call_count": len(plan.calls),
                    "call_sequence_hash": plan.call_sequence_hash,
                }
                for strategy, plan in zip(strategies, plans, strict=True)
            ],
            "selected_monitor_profiles": [
                {
                    "profile_id": profile["profile_id"],
                    "profile_hash": profile["profile_hash"],
                    "profile_row_hash": stable_hash(profile),
                    "family": profile["family"],
                    "monitor_identity": _monitor_identity(profile).scientific_dict(),
                }
                for profile in profiles
            ],
            "attacker_identity": attacker_identity.scientific_dict(),
            "learned_runtime": dict(learned_runtime),
            "compatibility": _compatibility_dict(compatibility),
            "client_topology": {
                "process_count": 1,
                "attacker_client_count": 1,
                "monitor_client_count": 2,
                "shared_monitor_client": False,
                "monitor_profile_count": 2,
                "attacker_device": attacker_device,
                "monitor_device": monitor_device,
                "gpu_device_count": 1,
                "load_order": [
                    "attacker",
                    f"monitor:{profiles[0]['profile_id']}",
                    f"monitor:{profiles[1]['profile_id']}",
                ],
                "simultaneously_retained_client_count": 3,
            },
            "memory_evidence": memory_evidence,
            "checks": {
                "attacker": attacker_checks,
                "monitor": monitor_checks,
                "lifecycle": lifecycle_checks,
            },
            "attacker_provenance": dict(attacker.provenance),
            "errors": errors,
            "summary": {
                "total_checks": len(all_checks),
                "passed_checks": passed_checks,
                "failed_checks": len(all_checks) - passed_checks,
            },
            "status": "passed" if not errors else "failed",
        }
    )
    return _finalize_report(payload)


def _fatal_report(
    *,
    spec: Mapping[str, Any] | None,
    dependencies: ConformanceDependencies,
    error: BaseException,
) -> dict[str, Any]:
    raw = spec or {}
    try:
        source_provenance = _source_provenance(dependencies)
    except Exception as provenance_error:
        source_provenance = {
            "collection_error": _error(
                stage="provenance", check_id=None, error=provenance_error
            )
        }
    payload = _base_report(
        conformance_spec_hash=(
            str(raw["conformance_spec_hash"])
            if isinstance(raw.get("conformance_spec_hash"), str)
            else None
        ),
        source_tree_hash=(
            str(raw["source_tree_hash"])
            if isinstance(raw.get("source_tree_hash"), str)
            else None
        ),
        upstream_artifacts={
            field: raw.get(field)
            for field in (
                "catalog_hash",
                "split_manifest_hash",
                "candidate_strategy_catalog_hash",
            )
        },
        source_provenance=source_provenance,
    )
    payload["errors"] = [
        _error(stage="preflight", check_id=None, error=error)
    ]
    return _finalize_report(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the development-only controlled checkpoint protocol "
            "conformance gate."
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--strategy-catalog", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--attacker-checkpoint", type=Path, required=True)
    parser.add_argument("--monitor-checkpoint", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--attacker-device", default="cuda")
    parser.add_argument("--monitor-device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: ConformanceDependencies | None = None,
) -> int:
    """CLI entry point; any preflight or protocol failure returns nonzero."""

    args = _parser().parse_args(argv)
    deps = dependencies or ConformanceDependencies()
    if args.output.exists():
        print(
            f"conformance error: refusing to replace existing output {args.output}",
            file=sys.stderr,
        )
        return 2
    spec: Mapping[str, Any] | None = None
    try:
        spec = load_json_object(args.spec, label="conformance spec")
        catalog = load_json_object(args.catalog, label="catalog")
        splits = load_json_object(args.splits, label="split manifest")
        strategies = load_json_object(
            args.strategy_catalog, label="candidate-strategy catalog"
        )
        report = execute_controlled_conformance(
            spec=spec,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategies,
            dependency_lock_path=args.dependency_lock,
            attacker_checkpoint=args.attacker_checkpoint,
            monitor_checkpoint=args.monitor_checkpoint,
            model_cache=args.model_cache,
            attacker_device=args.attacker_device,
            monitor_device=args.monitor_device,
            dependencies=deps,
        )
    except Exception as exc:
        report = _fatal_report(spec=spec, dependencies=deps, error=exc)
    try:
        atomic_create_json(args.output, report)
    except ResultValidationError as exc:
        print(f"conformance error: {exc}", file=sys.stderr)
        return 2
    print(
        f"conformance_status={report['status']} "
        f"report={args.output} hash={report['conformance_report_hash']}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main
    sys.exit(main())


__all__ = [
    "CONFORMANCE_ARTIFACT_CLASS",
    "CONFORMANCE_CLAIM_BOUNDARY",
    "CONFORMANCE_EVIDENCE_CLASS",
    "CONFORMANCE_REPORT_SCHEMA_VERSION",
    "CONFORMANCE_SPEC_SCHEMA_VERSION",
    "ConformanceDependencies",
    "ConformanceError",
    "execute_controlled_conformance",
    "main",
    "validate_conformance_report",
    "validate_conformance_spec",
]
