"""Atomic, hash-bound persistence for AgentDojo experiment shards.

This module deliberately has no AgentDojo, torch, or transformers imports.  A
worker can therefore inspect or resume a shard before any model is loaded.
Each completed trial is first published as an independent checkpoint; the
final JSONL and manifest are produced only after the complete expected cohort
has been validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
import os
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from silenttwin.io.jsonl import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_objects_jsonl,
    read_jsonl,
    sha256_file,
)
from silenttwin.io.provenance import collect_provenance
from silenttwin.schemas import stable_digest

from .config import (
    AGENTDOJO_MANIFEST_SCHEMA,
    AGENTDOJO_RESULT_SCHEMA,
    AgentDojoExperimentConfig,
    canonical_json,
    stable_hash,
)
from .runtime_integrity import (
    RuntimeIntegrityError,
    not_applicable_learned_runtime_provenance,
    validate_learned_runtime_provenance,
)


CHECKPOINT_SCHEMA = "silenttwin.agentdojo.checkpoint.v1"
RESULT_FILENAME = "result.jsonl"
FAILURES_FILENAME = "failures.jsonl"
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
CHECKPOINT_DIRECTORY = "checkpoints"
LOG_FILENAME = "run.log"


def _learned_runtime_fingerprints(
    config: AgentDojoExperimentConfig,
) -> set[str]:
    return {
        model.runtime_fingerprint
        for model in config.models
        if model.implementation in {
            "local_transformers",
            "transformers_pi_detector",
        }
    }


def _validated_provenance(
    provenance: Mapping[str, Any],
    *,
    config: AgentDojoExperimentConfig,
) -> dict[str, Any]:
    """Require retained runtime identity for learned runs and an explicit sentinel otherwise."""

    value = dict(provenance)
    expected = _learned_runtime_fingerprints(config)
    learned_runtime = value.get("learned_runtime")
    if learned_runtime is None and not expected:
        learned_runtime = not_applicable_learned_runtime_provenance()
        value["learned_runtime"] = learned_runtime
    if not isinstance(learned_runtime, Mapping):
        raise ResultValidationError(
            "AgentDojo provenance lacks learned-runtime provenance"
        )
    try:
        validate_learned_runtime_provenance(
            learned_runtime,
            expected_runtime_fingerprints=expected,
        )
    except RuntimeIntegrityError as exc:
        raise ResultValidationError(
            f"AgentDojo learned-runtime provenance is invalid: {exc}"
        ) from exc
    return value


ENGINEERING_SMOKE_EVIDENCE_CLASS = "engineering_smoke_only"
BENCHMARK_EVIDENCE_CLASS = "agentdojo_benchmark_execution"
_RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRUSTED_ENVELOPE_SCHEMA = "silenttwin.agentdojo.trusted.v1"
_PUBLIC_ENVELOPE_SCHEMAS = frozenset(
    {
        "silenttwin.agentdojo.public.v1",
        "silenttwin.agentdojo.probe-feedback.v1",
    }
)


def _raw_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_raw_sha256(value: Any, *, label: str, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or _RAW_SHA256_RE.fullmatch(value) is None:
        raise ResultValidationError(f"{label} is not an exact raw SHA-256 digest")


def _validate_durable_namespaces(row: Mapping[str, Any]) -> None:
    """Require the exact typed envelopes promised by result schema v1."""

    trusted = row.get("trusted_evaluation")
    if (
        not isinstance(trusted, Mapping)
        or set(trusted) != {"schema", "value"}
        or trusted.get("schema") != _TRUSTED_ENVELOPE_SCHEMA
        or not isinstance(trusted.get("value"), Mapping)
    ):
        raise ResultValidationError(
            "AgentDojo v1 result requires the exact trusted-evaluation envelope schema"
        )
    if "public_transcript" in row:
        raise ResultValidationError(
            "AgentDojo v1 result forbids the legacy public_transcript fallback"
        )
    for field in ("agent_visible_transcript", "postselection_output"):
        envelopes = row.get(field)
        if not isinstance(envelopes, list):
            raise ResultValidationError(
                f"AgentDojo v1 result requires {field} as typed public envelopes"
            )
        for index, envelope in enumerate(envelopes):
            if not isinstance(envelope, Mapping) or set(envelope) != {
                "schema",
                "value",
            }:
                raise ResultValidationError(
                    f"{field}[{index}] is not an exact public envelope"
                )
            schema = envelope.get("schema")
            if schema not in _PUBLIC_ENVELOPE_SCHEMAS:
                raise ResultValidationError(
                    f"{field}[{index}] does not use a public AgentDojo v1 schema"
                )


def _validate_model_call_record(call: Any, *, label: str) -> None:
    if not isinstance(call, Mapping):
        raise ResultValidationError(f"{label} is not a model-call object")
    required = {
        "phase",
        "call_index",
        "seed",
        "canonical_input_hash",
        "rendered_input_hash",
        "raw_response_hash",
        "parsed_output_hash",
        "canonical_input",
        "protocol_prompt",
        "rendered_chat_template_input",
        "raw_response",
        "parsed_output",
        "latency_ms",
        "failure_metadata",
        "metadata",
        "error",
    }
    missing = required - set(call)
    if missing:
        raise ResultValidationError(
            f"{label} lacks required provenance fields {sorted(missing)}"
        )
    if not isinstance(call["phase"], str) or not call["phase"]:
        raise ResultValidationError(f"{label} has an invalid phase")
    for field in ("call_index", "seed"):
        value = call[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResultValidationError(f"{label} has an invalid {field}")
    _require_raw_sha256(call["canonical_input_hash"], label=f"{label} canonical hash")
    # Model-visible AgentDojo messages/tools use the shared canonical protocol
    # serializer (UTF-8 JSON with non-ASCII text preserved). Artifact/config
    # self-hashes use ``config.stable_hash`` instead; they differ for Unicode.
    if call["canonical_input_hash"] != stable_digest(call["canonical_input"]):
        raise ResultValidationError(f"{label} canonical input hash is inconsistent")
    prompt = call["protocol_prompt"]
    rendered = call["rendered_chat_template_input"]
    if not isinstance(prompt, str) or not isinstance(rendered, str):
        raise ResultValidationError(f"{label} lacks exact prompt/rendered material")
    _require_raw_sha256(call["rendered_input_hash"], label=f"{label} rendered hash")
    if call["rendered_input_hash"] != _raw_text_sha256(rendered):
        raise ResultValidationError(f"{label} rendered input hash is inconsistent")
    raw_response = call["raw_response"]
    if raw_response is not None and not isinstance(raw_response, str):
        raise ResultValidationError(f"{label} raw response must be text or null")
    _require_raw_sha256(
        call["raw_response_hash"], label=f"{label} raw response hash", nullable=True
    )
    expected_raw_hash = (
        _raw_text_sha256(raw_response) if raw_response is not None else None
    )
    if call["raw_response_hash"] != expected_raw_hash:
        raise ResultValidationError(f"{label} raw response hash is inconsistent")
    parsed = call.get("parsed_output")
    _require_raw_sha256(
        call["parsed_output_hash"],
        label=f"{label} parsed output hash",
        nullable=True,
    )
    expected_parsed_hash = stable_digest(parsed) if parsed is not None else None
    if call["parsed_output_hash"] != expected_parsed_hash:
        raise ResultValidationError(f"{label} parsed output hash is inconsistent")
    latency = call["latency_ms"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or float(latency) < 0.0
    ):
        raise ResultValidationError(f"{label} has invalid wrapper latency")
    if not isinstance(call["failure_metadata"], Mapping) or not isinstance(
        call["metadata"], Mapping
    ):
        raise ResultValidationError(f"{label} has invalid response/failure metadata")
    error = call["error"]
    if error is not None and (not isinstance(error, str) or not error):
        raise ResultValidationError(f"{label} has invalid failure status")
    if error is None and (raw_response is None or parsed is None):
        raise ResultValidationError(
            f"{label} success lacks raw response or parsed output"
        )
    if error is not None and (
        parsed is not None or call["parsed_output_hash"] is not None
    ):
        raise ResultValidationError(
            f"{label} failure improperly claims parsed output"
        )


def _validate_monitor_call_record(call: Any, *, label: str) -> None:
    """Validate LocalActionMonitor success or failure call material."""

    if not isinstance(call, Mapping):
        raise ResultValidationError(f"{label} is not a monitor-call object")
    required = {
        "canonical_monitor_input",
        "protocol_prompt",
        "rendered_chat_template_input",
        "rendered_input_hash",
        "raw_response",
        "raw_response_hash",
        "parsed_output",
        "seed",
        "latency_ms",
        "metadata",
        "failure",
    }
    missing = required - set(call)
    if missing:
        raise ResultValidationError(
            f"{label} lacks required monitor provenance {sorted(missing)}"
        )
    if not isinstance(call["canonical_monitor_input"], Mapping) or not isinstance(
        call["protocol_prompt"], str
    ):
        raise ResultValidationError(f"{label} lacks canonical monitor input/prompt")
    rendered = call["rendered_chat_template_input"]
    raw = call["raw_response"]
    if (
        not isinstance(call["protocol_prompt"], str)
        or not isinstance(rendered, str)
        or (raw is not None and not isinstance(raw, str))
    ):
        raise ResultValidationError(f"{label} lacks exact rendered/raw material")
    if call["rendered_input_hash"] != _raw_text_sha256(rendered):
        raise ResultValidationError(f"{label} rendered input hash is inconsistent")
    if call["raw_response_hash"] != (
        _raw_text_sha256(raw) if raw is not None else None
    ):
        raise ResultValidationError(f"{label} raw response hash is inconsistent")
    seed = call["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ResultValidationError(f"{label} has an invalid seed")
    latency = call["latency_ms"]
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ResultValidationError(f"{label} has invalid wrapper latency")
    if not isinstance(call["metadata"], Mapping):
        raise ResultValidationError(f"{label} has invalid response metadata")
    failure = call["failure"]
    if failure is not None and not isinstance(failure, Mapping):
        raise ResultValidationError(f"{label} has invalid failure provenance")
    if failure is not None and not isinstance(call.get("failure_metadata"), Mapping):
        raise ResultValidationError(f"{label} lacks provider failure metadata")
    if failure is None and (raw is None or call["parsed_output"] is None):
        raise ResultValidationError(
            f"{label} success lacks raw response or parsed output"
        )
    if failure is not None and call["parsed_output"] is not None:
        raise ResultValidationError(
            f"{label} failure improperly claims parsed output"
        )


def _validate_nested_monitor_calls(value: Any, *, label: str) -> int:
    count = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "model_call" and child is not None:
                _validate_monitor_call_record(child, label=f"{label}.model_call")
                count += 1
            elif key == "failed_model_call" and child is not None:
                _validate_monitor_call_record(
                    child, label=f"{label}.failed_model_call"
                )
                count += 1
            else:
                count += _validate_nested_monitor_calls(
                    child, label=f"{label}.{key}"
                )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            count += _validate_nested_monitor_calls(
                child, label=f"{label}[{index}]"
            )
    return count


def _validate_auxiliary_defense_call(call: Any, *, label: str) -> None:
    if not isinstance(call, Mapping):
        raise ResultValidationError(f"{label} is not a call-provenance object")
    for field in (
        "canonical_input",
        "canonical_input_hash",
        "protocol_prompt",
        "protocol_prompt_hash",
        "rendered_chat_template_input",
        "raw_response",
        "raw_response_serialization",
        "rendered_input_hash",
        "raw_response_hash",
        "usage",
        "latency_ms",
        "response_metadata",
        "failure_metadata",
        "failure",
    ):
        if field not in call:
            raise ResultValidationError(f"{label} lacks {field}")
    rendered = call["rendered_chat_template_input"]
    raw = call["raw_response"]
    if (
        not isinstance(call["protocol_prompt"], str)
        or not isinstance(rendered, str)
        or (raw is not None and not isinstance(raw, str))
    ):
        raise ResultValidationError(f"{label} lacks exact rendered/raw material")
    if call["canonical_input_hash"] != stable_digest(call["canonical_input"]):
        raise ResultValidationError(f"{label} canonical input hash is inconsistent")
    if call["protocol_prompt_hash"] != _raw_text_sha256(call["protocol_prompt"]):
        raise ResultValidationError(f"{label} protocol prompt hash is inconsistent")
    if call["rendered_input_hash"] != _raw_text_sha256(rendered):
        raise ResultValidationError(f"{label} rendered input hash is inconsistent")
    if call["raw_response_hash"] != (
        _raw_text_sha256(raw) if raw is not None else None
    ):
        raise ResultValidationError(f"{label} raw response hash is inconsistent")
    if not isinstance(call["raw_response_serialization"], Mapping) or not isinstance(
        call["usage"], Mapping
    ):
        raise ResultValidationError(f"{label} has invalid usage provenance")
    latency = call["latency_ms"]
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ResultValidationError(f"{label} has invalid wrapper latency")
    if call["failure"] is not None and not isinstance(call["failure"], Mapping):
        raise ResultValidationError(f"{label} has invalid failure provenance")
    if not isinstance(call["response_metadata"], Mapping) or not isinstance(
        call["failure_metadata"], Mapping
    ):
        raise ResultValidationError(f"{label} has invalid transport metadata")


def _validate_model_provenance(
    row: Mapping[str, Any], *, config: AgentDojoExperimentConfig
) -> None:
    """Validate retained generation/defense call material before persistence."""

    if row.get("experiment_id") not in {
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "ecological",
    }:
        return
    provenance = row.get("model_provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ResultValidationError("AgentDojo result lacks model_provenance")
    learned_action_monitor = any(
        model.role == "monitor" and model.implementation == "local_transformers"
        for model in config.models
    ) or config.monitor_family not in {
        "deterministic_task_policy",
        "decision_independent_semantic_twin",
        "ecological_no_action_monitor",
    }
    if "attacker" in provenance:
        attacker = provenance["attacker"]
        if not isinstance(attacker, Mapping) or not isinstance(
            attacker.get("calls"), list
        ):
            raise ResultValidationError(
                "controlled result lacks its attacker call ledger"
            )
        calls = attacker["calls"]
        if not calls:
            raise ResultValidationError("controlled attacker call ledger is empty")
        for index, call in enumerate(calls):
            _validate_model_call_record(call, label=f"attacker call {index}")
        indices = [call["call_index"] for call in calls]
        if indices != list(range(len(calls))):
            raise ResultValidationError(
                "controlled attacker call indices are not unique and sequential"
            )
        phases = [call["phase"] for call in calls]
        if "prediction" not in phases:
            raise ResultValidationError(
                "controlled attacker ledger lacks its mandatory prediction call"
            )
        allowed_phases = {
            "probe",
            "prediction",
            "final",
            "postselection_observation",
        }
        if any(phase not in allowed_phases for phase in phases):
            raise ResultValidationError("controlled attacker ledger has an unknown phase")
        prediction_index = phases.index("prediction")
        if (
            any(phase != "probe" for phase in phases[:prediction_index])
            or phases.count("prediction") != 1
            or any(
                phase not in {"final", "postselection_observation"}
                for phase in phases[prediction_index + 1 :]
            )
            or phases[prediction_index + 1 :]
            != [
                *(["final"] * phases.count("final")),
                *(
                    ["postselection_observation"]
                    * phases.count("postselection_observation")
                ),
            ]
        ):
            raise ResultValidationError("controlled attacker call phase order is invalid")
        phase_counts = {phase: phases.count(phase) for phase in allowed_phases}
        if (
            phase_counts["probe"] > config.query_budget
            or phase_counts["final"] > 1
            or phase_counts["postselection_observation"] > 1
        ):
            raise ResultValidationError(
                "controlled attacker call counts exceed the frozen protocol"
            )
        termination_comparator = row.get("condition") == (
            "verdict_dependent_termination"
        )
        if (
            row.get("run_valid") is True
            and not termination_comparator
            and phase_counts["probe"] != config.query_budget
        ):
            raise ResultValidationError(
                "valid controlled row did not consume the frozen probe budget"
            )
        experiment_id = str(row.get("experiment_id"))
        if experiment_id == "e1" and phase_counts["final"] != 0:
            raise ResultValidationError("E1 attacker ledger contains a final call")
        if experiment_id in {"e2", "e3", "e5"}:
            final_is_runner_selected = (
                experiment_id == "e2"
                and row.get("condition") in {"oracle", "random_final"}
            )
            expected_final_calls = 0 if final_is_runner_selected else 1
            if phase_counts["final"] != expected_final_calls:
                raise ResultValidationError(
                    "controlled attacker final-call count differs from its condition"
                )
        expected_postselection_calls = (
            1 if row.get("postselection_delivered_to_agent") is True else 0
        )
        if phase_counts["postselection_observation"] != expected_postselection_calls:
            raise ResultValidationError(
                "controlled postselection call count differs from its delivery flag"
            )
        monitor_calls = _validate_nested_monitor_calls(
            row.get("trusted_evaluation"), label="trusted_evaluation"
        )
        learned_monitor_required = learned_action_monitor and (
            experiment_id in {"e2", "e3", "e5"}
            or (
                experiment_id == "e1"
                and config.query_budget > 0
                and row.get("condition") in {"genuine", "matched_shuffled"}
            )
        )
        if learned_monitor_required and monitor_calls == 0:
            raise ResultValidationError(
                "learned controlled row lacks monitor model-call provenance"
            )
        return
    if row.get("tier2_track") == "ecological":
        calls = provenance.get("calls")
        if not isinstance(calls, list):
            raise ResultValidationError("ecological result lacks its victim call ledger")
        if not calls and row.get("error_stage") != "setup":
            raise ResultValidationError(
                "ecological victim call ledger is empty without an explicit setup failure"
            )
        for index, call in enumerate(calls):
            _validate_model_call_record(call, label=f"victim call {index}")
        indices = [call["call_index"] for call in calls]
        if indices != list(range(len(calls))) or any(
            call["phase"] != "ecological_tool_turn" for call in calls
        ):
            raise ResultValidationError(
                "ecological victim call indices/phases are inconsistent"
            )
        filter_provenance = provenance.get("tool_filter_provenance")
        if isinstance(filter_provenance, Mapping):
            filter_call = filter_provenance.get("call")
            if filter_call is not None:
                _validate_auxiliary_defense_call(
                    filter_call, label="adapted tool-filter call"
                )
        detector = provenance.get("content_detector_provenance")
        if isinstance(detector, Mapping):
            detector_calls = detector.get("calls")
            if not isinstance(detector_calls, list):
                raise ResultValidationError(
                    "content-detector provenance lacks its call ledger"
                )
            for index, call in enumerate(detector_calls):
                if not isinstance(call, Mapping):
                    raise ResultValidationError(
                        f"content-detector call {index} is invalid"
                    )
                for field in (
                    "detector_input_text",
                    "detector_input_hash",
                    "rendered_chat_template_input",
                    "tokenized_input",
                    "raw_output",
                    "raw_output_hash",
                    "raw_output_serialization",
                    "usage",
                    "latency_ms",
                    "failure",
                ):
                    if field not in call:
                        raise ResultValidationError(
                            f"content-detector call {index} lacks {field}"
                        )
                detector_input = call["detector_input_text"]
                raw_output = call["raw_output"]
                if not isinstance(detector_input, str) or (
                    raw_output is not None and not isinstance(raw_output, str)
                ):
                    raise ResultValidationError(
                        f"content-detector call {index} lacks exact input/output material"
                    )
                if call["detector_input_hash"] != _raw_text_sha256(detector_input):
                    raise ResultValidationError(
                        f"content-detector call {index} input hash is inconsistent"
                    )
                if call["raw_output_hash"] != (
                    _raw_text_sha256(raw_output) if raw_output is not None else None
                ):
                    raise ResultValidationError(
                        f"content-detector call {index} output hash is inconsistent"
                    )
                if not all(
                    isinstance(call[field], Mapping)
                    for field in (
                        "rendered_chat_template_input",
                        "tokenized_input",
                        "raw_output_serialization",
                        "usage",
                    )
                ):
                    raise ResultValidationError(
                        f"content-detector call {index} has invalid availability metadata"
                    )
                latency = call["latency_ms"]
                if (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or latency < 0
                ):
                    raise ResultValidationError(
                        f"content-detector call {index} has invalid latency"
                    )
                if call["failure"] is not None and not isinstance(
                    call["failure"], Mapping
                ):
                    raise ResultValidationError(
                        f"content-detector call {index} has invalid failure provenance"
                    )
        return
    # Authored E4/E5 rows deliberately have no generation model.  They must
    # state that boundary instead of silently omitting provenance.
    if provenance.get("generation_calls") != 0:
        raise ResultValidationError(
            "model-free authored row lacks generation_calls=0 provenance"
        )
    trusted = row.get("trusted_evaluation")
    monitor_calls = _validate_nested_monitor_calls(
        trusted, label="trusted_evaluation"
    )
    if learned_action_monitor and monitor_calls == 0:
        raise ResultValidationError(
            "learned authored row lacks monitor model-call provenance"
        )


def evidence_boundary_for_config(
    config: AgentDojoExperimentConfig,
) -> dict[str, bool | str]:
    """Return the exact evidence labels implied by one scientific config."""

    fixture_mode = config.fixture_mode
    return {
        "fixture_mode": fixture_mode,
        "evidence_class": (
            ENGINEERING_SMOKE_EVIDENCE_CLASS
            if fixture_mode
            else BENCHMARK_EVIDENCE_CLASS
        ),
        "scientific_evidence_eligible": not fixture_mode,
    }


def _validate_evidence_boundary(
    document: Mapping[str, Any],
    *,
    config: AgentDojoExperimentConfig,
    label: str,
) -> None:
    expected = evidence_boundary_for_config(config)
    for field, expected_value in expected.items():
        observed = document.get(field)
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ResultValidationError(
                f"{label} {field} differs from its frozen fixture-mode evidence boundary"
            )


def bind_evidence_boundary(
    sample: Mapping[str, Any],
    *,
    config: AgentDojoExperimentConfig,
) -> dict[str, Any]:
    """Attach config-derived evidence labels without masking contradictions."""

    row = dict(sample)
    expected = evidence_boundary_for_config(config)
    for field, expected_value in expected.items():
        if field not in row:
            continue
        observed = row[field]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ResultValidationError(
                "AgentDojo runner result "
                f"{field} contradicts its frozen fixture-mode evidence boundary"
            )
    row.update(expected)
    return row


def trial_checkpoint_id(configuration_hash: str, trial_id: str) -> str:
    """Return the immutable filename identity for one expected trial."""

    return stable_hash(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "configuration_hash": configuration_hash,
            "trial_id": trial_id,
        }
    )


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultValidationError(f"missing {label}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultValidationError(f"{label} must be one JSON object: {path}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_sample(
    sample: Mapping[str, Any],
    *,
    config: AgentDojoExperimentConfig,
    trial_id: str,
) -> dict[str, Any]:
    row = json.loads(canonical_json(dict(sample)))
    if row.get("schema_version") != AGENTDOJO_RESULT_SCHEMA:
        raise ResultValidationError("incompatible AgentDojo result schema")
    if row.get("record_type") != "sample":
        raise ResultValidationError("AgentDojo checkpoint is not a sample record")
    _validate_durable_namespaces(row)
    if row.get("trial_id") != trial_id:
        raise ResultValidationError("AgentDojo checkpoint trial ID is inconsistent")
    if row.get("environment_backend") != "agentdojo":
        raise ResultValidationError("AgentDojo result has another backend identity")
    if row.get("experiment_id") != config.experiment_id:
        raise ResultValidationError("AgentDojo result belongs to another experiment")
    if row.get("tier2_track") != config.tier2_track:
        raise ResultValidationError("AgentDojo result belongs to another Tier-2 track")
    if row.get("agentdojo_suite") != config.agentdojo_suite:
        raise ResultValidationError("AgentDojo result belongs to another suite")
    _validate_evidence_boundary(row, config=config, label="AgentDojo result")
    if row.get("scenario_id") not in config.scenario_ids:
        raise ResultValidationError("AgentDojo result is outside the frozen scenario bundle")
    if row.get("structural_group_id") not in config.structural_group_ids:
        raise ResultValidationError("AgentDojo result is outside the frozen structural bundle")
    if config.experiment_id in {"e1", "e2", "e3", "e4", "e5", "ecological"}:
        expected_condition = (
            config.feedback_source
            if config.experiment_id == "e1"
            else config.condition
            if config.experiment_id == "e2"
            else config.closure_channel
            if config.experiment_id == "e3"
            else f"{config.workflow}:{config.settlement_runtime}"
            if config.experiment_id == "e4"
            else config.ablation
            if config.experiment_id == "e5"
            else f"{config.threat_mode}:{config.ecological_defense}"
        )
        bindings = {
            "dataset_split": config.dataset_split,
            "threat_mode": config.threat_mode,
            "feedback_policy": config.feedback_policy,
            "feedback_source": config.feedback_source,
            "query_budget": config.query_budget,
            "condition": expected_condition,
        }
        for field, expected in bindings.items():
            if row.get(field) != expected:
                raise ResultValidationError(
                    f"AgentDojo result {field} differs from its frozen configuration"
                )
        specialized = (
            {"closure_channel": config.closure_channel}
            if config.experiment_id == "e3"
            else {
                "workflow": config.workflow,
                "settlement_runtime": config.settlement_runtime,
            }
            if config.experiment_id == "e4"
            else {"ablation": config.ablation}
            if config.experiment_id == "e5"
            else {
                "ecological_attack": config.ecological_attack,
                "ecological_defense": config.ecological_defense,
                "released_attack_name": config.released_attack_name,
                "released_attack_target_pipeline": (
                    config.released_attack_target_pipeline
                ),
                "settlement_runtime": config.settlement_runtime,
            }
            if config.experiment_id == "ecological"
            else {}
        )
        for field, expected in specialized.items():
            if row.get(field) != expected:
                raise ResultValidationError(
                    f"AgentDojo result {field} differs from its frozen configuration"
                )
        trusted = row.get("trusted_evaluation")
        if not isinstance(trusted, Mapping) or not isinstance(
            trusted.get("value"), Mapping
        ):
            raise ResultValidationError(
                "AgentDojo result lacks its typed trusted-evaluation namespace"
            )
        trusted_value = trusted["value"]
        scenario_id = str(row["scenario_id"])
        if config.experiment_id in {"e1", "e2"}:
            actual = trusted_value.get("actual_hidden_state")
            donor = trusted_value.get("donor_state")
            expected_trial_id = stable_hash(
                {
                    "protocol": "silenttwin.agentdojo.controlled.v1",
                    "configuration_hash": config.configuration_hash,
                    "scenario_id": scenario_id,
                    "actual_state": actual,
                    "donor_state": donor,
                    "replicate": config.replicate,
                }
            )
        elif config.experiment_id in {"e3", "e5"}:
            actual = trusted_value.get("actual_hidden_state")
            donor = trusted_value.get("donor_state")
            expected_trial_id = stable_hash(
                {
                    "protocol": "silenttwin.agentdojo.advanced.v1",
                    "configuration_hash": config.configuration_hash,
                    "scenario_id": scenario_id,
                    "actual_state": actual,
                    "donor_state": donor,
                    "replicate": config.replicate,
                }
            )
        elif config.experiment_id == "e4":
            actual = trusted_value.get("actual_hidden_state")
            expected_trial_id = stable_hash(
                {
                    "protocol": "silenttwin.agentdojo.useful-work.v1",
                    "configuration_hash": config.configuration_hash,
                    "scenario_id": scenario_id,
                    "actual_state": actual,
                    "workflow": config.workflow,
                    "settlement_runtime": config.settlement_runtime,
                    "replicate": config.replicate,
                }
            )
        else:
            expected_trial_id = stable_hash(
                {
                    "protocol": "silenttwin.agentdojo.tool-loop.v1",
                    "configuration_hash": config.configuration_hash,
                    "scenario_id": scenario_id,
                    "threat_mode": config.threat_mode,
                    "defense": config.ecological_defense,
                    "replicate": config.replicate,
                }
            )
        if trial_id != expected_trial_id:
            raise ResultValidationError(
                "AgentDojo result trusted assignment does not match its trial ID"
            )
    existing_hash = row.get("configuration_hash")
    if existing_hash is not None and existing_hash != config.configuration_hash:
        raise ResultValidationError("AgentDojo result configuration hash is inconsistent")
    _validate_model_provenance(row, config=config)
    row["configuration_hash"] = config.configuration_hash
    return row


def _checkpoint_document(
    *,
    config: AgentDojoExperimentConfig,
    checkpoint_id: str,
    trial_id: str,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "configuration_hash": config.configuration_hash,
        "checkpoint_id": checkpoint_id,
        "trial_id": trial_id,
        "sample": dict(sample),
    }
    return {**payload, "checkpoint_hash": stable_hash(payload)}


def _validate_checkpoint_document(
    checkpoint: Mapping[str, Any],
    *,
    config: AgentDojoExperimentConfig,
    checkpoint_id: str,
    trial_id: str,
    label: str,
) -> dict[str, Any]:
    payload = dict(checkpoint)
    recorded_hash = payload.pop("checkpoint_hash", None)
    if recorded_hash != stable_hash(payload):
        raise ResultValidationError(f"{label} self-hash is invalid")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ResultValidationError(f"incompatible {label}")
    if checkpoint.get("configuration_hash") != config.configuration_hash:
        raise ResultValidationError(f"{label} configuration hash is inconsistent")
    if checkpoint.get("checkpoint_id") != checkpoint_id:
        raise ResultValidationError(f"{label} checkpoint ID is inconsistent")
    if checkpoint.get("trial_id") != trial_id:
        raise ResultValidationError(f"{label} trial ID is inconsistent")
    sample = checkpoint.get("sample")
    if not isinstance(sample, Mapping):
        raise ResultValidationError(f"{label} has no sample object")
    return _validate_sample(sample, config=config, trial_id=trial_id)


def _validate_checkpoint_manifest_document(
    manifest: Mapping[str, Any],
    *,
    configuration_hash: str,
    source_tree_hash: str,
    expected_trial_ids: Sequence[str],
) -> None:
    payload = dict(manifest)
    recorded_hash = payload.pop("checkpoint_manifest_hash", None)
    if recorded_hash != stable_hash(payload):
        raise ResultValidationError("AgentDojo checkpoint manifest hash is invalid")
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ResultValidationError("incompatible AgentDojo checkpoint schema")
    if manifest.get("status") not in {"running", "complete"}:
        raise ResultValidationError("AgentDojo checkpoint manifest has an invalid status")
    if manifest.get("configuration_hash") != configuration_hash:
        raise ResultValidationError("checkpoint belongs to another configuration")
    if manifest.get("source_tree_hash") != source_tree_hash:
        raise ResultValidationError("checkpoint belongs to another source tree")
    expected_trials = [str(item) for item in expected_trial_ids]
    if len(expected_trials) != len(set(expected_trials)):
        raise ResultValidationError("checkpoint expected trial IDs are not unique")
    if manifest.get("expected_trial_ids") != expected_trials:
        raise ResultValidationError("checkpoint expected trial order changed")
    expected_checkpoints = sorted(
        trial_checkpoint_id(configuration_hash, trial_id)
        for trial_id in expected_trials
    )
    if manifest.get("expected_checkpoint_ids") != expected_checkpoints:
        raise ResultValidationError("checkpoint expected cohort changed")
    completed_raw = manifest.get("completed_checkpoint_ids")
    if not isinstance(completed_raw, list):
        raise ResultValidationError("checkpoint completed cohort is not a list")
    completed = [str(item) for item in completed_raw]
    if completed != sorted(completed) or len(completed) != len(set(completed)):
        raise ResultValidationError("checkpoint completed cohort is not unique and sorted")
    if not set(completed) <= set(expected_checkpoints):
        raise ResultValidationError("checkpoint completed cohort contains unexpected IDs")
    if manifest.get("status") == "complete" and completed != expected_checkpoints:
        raise ResultValidationError("complete checkpoint manifest has an incomplete cohort")


def _load_published_checkpoint_samples(
    directory: Path,
    *,
    config: AgentDojoExperimentConfig,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_trials = [str(item) for item in manifest["expected_trial_ids"]]
    expected = {
        trial_checkpoint_id(config.configuration_hash, trial_id): trial_id
        for trial_id in expected_trials
    }
    checkpoint_dir = directory / CHECKPOINT_DIRECTORY
    actual_paths = {
        path.stem: path for path in checkpoint_dir.glob("*.json") if path.is_file()
    }
    if set(actual_paths) != set(expected):
        raise ResultValidationError(
            "published checkpoint files do not match the complete expected cohort"
        )
    samples: dict[str, dict[str, Any]] = {}
    for checkpoint_id, trial_id in expected.items():
        checkpoint = _read_object(
            actual_paths[checkpoint_id], label="AgentDojo trial checkpoint"
        )
        samples[trial_id] = _validate_checkpoint_document(
            checkpoint,
            config=config,
            checkpoint_id=checkpoint_id,
            trial_id=trial_id,
            label="AgentDojo trial checkpoint",
        )
    return samples


@dataclass(slots=True)
class AgentDojoCheckpointStore:
    """Crash-safe checkpoint store for one exact scientific configuration."""

    output_dir: Path
    config: AgentDojoExperimentConfig
    expected_trial_ids: tuple[str, ...]
    provenance_hash: str

    def __init__(
        self,
        output_dir: Path | str,
        config: AgentDojoExperimentConfig,
        expected_trial_ids: Iterable[str],
        *,
        provenance_hash: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.expected_trial_ids = tuple(str(item) for item in expected_trial_ids)
        self.provenance_hash = str(provenance_hash)
        if not self.expected_trial_ids:
            raise ValueError("an AgentDojo shard must expect at least one trial")
        if len(set(self.expected_trial_ids)) != len(self.expected_trial_ids):
            raise ValueError("expected AgentDojo trial IDs must be unique")
        if not self.provenance_hash:
            raise ValueError("AgentDojo checkpoints require a source provenance hash")

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / CHECKPOINT_DIRECTORY

    @property
    def checkpoint_manifest_path(self) -> Path:
        return self.output_dir / CHECKPOINT_MANIFEST_FILENAME

    @property
    def expected(self) -> dict[str, str]:
        return {
            trial_checkpoint_id(self.config.configuration_hash, trial_id): trial_id
            for trial_id in self.expected_trial_ids
        }

    def _manifest(self, *, status: str, completed: Iterable[str]) -> dict[str, Any]:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": status,
            "configuration_hash": self.config.configuration_hash,
            "source_tree_hash": self.provenance_hash,
            "expected_trial_ids": list(self.expected_trial_ids),
            "expected_checkpoint_ids": sorted(self.expected),
            "completed_checkpoint_ids": sorted(completed),
        }
        return {**payload, "checkpoint_manifest_hash": stable_hash(payload)}

    def _validate_manifest(self) -> dict[str, Any]:
        manifest = _read_object(
            self.checkpoint_manifest_path, label="AgentDojo checkpoint manifest"
        )
        _validate_checkpoint_manifest_document(
            manifest,
            configuration_hash=self.config.configuration_hash,
            source_tree_hash=self.provenance_hash,
            expected_trial_ids=self.expected_trial_ids,
        )
        return manifest

    def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_manifest_path.exists():
            self._validate_manifest()
            return
        atomic_write_json(
            self.checkpoint_manifest_path,
            self._manifest(status="running", completed=()),
        )

    def load(self) -> dict[str, dict[str, Any]]:
        initial = self._validate_manifest()
        completed: dict[str, dict[str, Any]] = {}
        checkpoint_ids: set[str] = set()
        for path in sorted(self.checkpoint_dir.glob("*.json")):
            checkpoint = _read_object(path, label="AgentDojo trial checkpoint")
            identifier = str(checkpoint.get("checkpoint_id", ""))
            if identifier not in self.expected or path.name != f"{identifier}.json":
                raise ResultValidationError(f"unexpected AgentDojo checkpoint: {path}")
            trial_id = self.expected[identifier]
            if trial_id in completed:
                raise ResultValidationError(f"checkpoint trial identity mismatch: {path}")
            completed[trial_id] = _validate_checkpoint_document(
                checkpoint,
                config=self.config,
                checkpoint_id=identifier,
                trial_id=trial_id,
                label=f"AgentDojo trial checkpoint {path}",
            )
            checkpoint_ids.add(identifier)
        declared = set(initial.get("completed_checkpoint_ids", ()))
        if declared != checkpoint_ids:
            if initial.get("status") == "running" and declared < checkpoint_ids:
                atomic_write_json(
                    self.checkpoint_manifest_path,
                    self._manifest(status="running", completed=checkpoint_ids),
                )
            else:
                raise ResultValidationError(
                    "AgentDojo checkpoint files and manifest disagree"
                )
        return {
            trial_id: completed[trial_id]
            for trial_id in self.expected_trial_ids
            if trial_id in completed
        }

    def save(self, sample: Mapping[str, Any]) -> None:
        self._validate_manifest()
        trial_id = str(sample.get("trial_id", ""))
        identifier = trial_checkpoint_id(self.config.configuration_hash, trial_id)
        if identifier not in self.expected:
            raise ResultValidationError(f"unexpected AgentDojo trial ID {trial_id!r}")
        row = _validate_sample(sample, config=self.config, trial_id=trial_id)
        path = self.checkpoint_dir / f"{identifier}.json"
        if path.exists():
            existing = _read_object(path, label="AgentDojo trial checkpoint")
            existing_sample = _validate_checkpoint_document(
                existing,
                config=self.config,
                checkpoint_id=identifier,
                trial_id=trial_id,
                label="AgentDojo existing trial checkpoint",
            )
            if existing_sample == row:
                return
            raise ResultValidationError(f"checkpoint collision for trial {trial_id!r}")
        atomic_write_json(
            path,
            _checkpoint_document(
                config=self.config,
                checkpoint_id=identifier,
                trial_id=trial_id,
                sample=row,
            ),
        )
        completed = {
            candidate.stem
            for candidate in self.checkpoint_dir.glob("*.json")
            if candidate.is_file()
        }
        if not completed <= set(self.expected):
            raise ResultValidationError("checkpoint directory contains an unexpected file")
        atomic_write_json(
            self.checkpoint_manifest_path,
            self._manifest(status="running", completed=completed),
        )

    def mark_complete(self) -> dict[str, dict[str, Any]]:
        rows = self.load()
        if tuple(rows) != self.expected_trial_ids:
            missing = [item for item in self.expected_trial_ids if item not in rows]
            raise ResultValidationError(
                f"cannot complete AgentDojo checkpoint; missing trial IDs: {missing}"
            )
        atomic_write_json(
            self.checkpoint_manifest_path,
            self._manifest(status="complete", completed=self.expected),
        )
        return rows


def make_run_manifest(
    config: AgentDojoExperimentConfig,
    *,
    output_dir: Path | str,
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    grid_hash: str,
    grid_task_id: int,
    shard_id: str,
    grid_batch_hash: str | None = None,
) -> dict[str, Any]:
    """Build the complete manifest after result and failure files exist."""

    directory = Path(output_dir)
    result_path = directory / RESULT_FILENAME
    failures_path = directory / FAILURES_FILENAME
    checkpoint_path = directory / CHECKPOINT_MANIFEST_FILENAME
    checkpoint = _read_object(checkpoint_path, label="AgentDojo checkpoint manifest")
    if len(records) != len({str(item.get("trial_id")) for item in records}):
        raise ResultValidationError("AgentDojo final records repeat trial IDs")
    trial_ids = [str(item.get("trial_id", "")) for item in records]
    if not all(trial_ids):
        raise ResultValidationError("AgentDojo final record has no trial ID")
    resolved_provenance = _validated_provenance(provenance, config=config)
    source_tree_hash = resolved_provenance.get("source_tree_hash")
    if not isinstance(source_tree_hash, str) or not source_tree_hash:
        raise ResultValidationError("AgentDojo provenance lacks a source-tree hash")
    _validate_checkpoint_manifest_document(
        checkpoint,
        configuration_hash=config.configuration_hash,
        source_tree_hash=source_tree_hash,
        expected_trial_ids=trial_ids,
    )
    if checkpoint.get("status") != "complete":
        raise ResultValidationError("cannot publish a run with incomplete checkpoints")
    checkpoint_samples = _load_published_checkpoint_samples(
        directory, config=config, manifest=checkpoint
    )
    normalized_records = [
        _validate_sample(record, config=config, trial_id=trial_id)
        for trial_id, record in zip(trial_ids, records, strict=True)
    ]
    if any(
        checkpoint_samples[trial_id] != record
        for trial_id, record in zip(trial_ids, normalized_records, strict=True)
    ):
        raise ResultValidationError("published result differs from its trial checkpoint")
    for index, failure in enumerate(failures):
        _validate_evidence_boundary(
            failure,
            config=config,
            label=f"AgentDojo failure row {index}",
        )
    generation = {
        model.role: {
            **model.scientific_dict(),
            "operational": model.operational_dict(),
        }
        for model in config.models
    }
    return {
        "manifest_schema_version": AGENTDOJO_MANIFEST_SCHEMA,
        "result_schema_version": AGENTDOJO_RESULT_SCHEMA,
        "status": "complete",
        "environment_backend": "agentdojo",
        "tier2_track": config.tier2_track,
        "experiment_id": config.experiment_id,
        **evidence_boundary_for_config(config),
        "configuration": config.scientific_dict(),
        "configuration_hash": config.configuration_hash,
        "operational_configuration": config.operational_dict(),
        "expected_trial_count": len(normalized_records),
        "actual_trial_count": len(normalized_records),
        "resolved_scenario_ids": list(config.scenario_ids),
        "resolved_scenario_count": len(config.scenario_ids),
        "structural_group_ids": list(config.structural_group_ids),
        "result_file": RESULT_FILENAME,
        "result_sha256": sha256_file(result_path),
        "failures_file": FAILURES_FILENAME,
        "failures_sha256": sha256_file(failures_path),
        "failure_count": len(failures),
        "checkpoint_manifest": CHECKPOINT_MANIFEST_FILENAME,
        "checkpoint_manifest_hash": stable_hash(checkpoint),
        "started_at": started_at,
        "completed_at": completed_at,
        "agentdojo": {
            "package_version": config.agentdojo_package_version,
            "source_revision": config.agentdojo_source_revision,
            "benchmark_version": config.agentdojo_benchmark_version,
            "catalog_hash": config.agentdojo_catalog_hash,
            "scenario_registry_revision": config.scenario_registry_revision,
            "scenario_registry_hash": config.scenario_registry_hash,
            "split_manifest_hash": config.split_manifest_hash,
            "candidate_strategy_catalog_hash": config.candidate_strategy_catalog_hash,
            "pair_registry_hash": config.pair_registry_hash,
        },
        "scientific_bindings": {
            "analysis_plan_hash": config.analysis_plan_hash,
            "dependency_lock_hash": config.dependency_lock_hash,
            "sample_size_freeze_hash": config.sample_size_freeze_hash,
            "development_evidence_hash": config.development_evidence_hash,
            "primary_contrast_id": config.primary_contrast_id,
            "tool_protocol_revision": config.tool_protocol_revision,
            "system_prompt_hash": config.system_prompt_hash,
            "monitor_profile_hash": config.monitor_profile_hash,
        },
        "generation_provenance": generation,
        "provenance": resolved_provenance,
        "orchestration": {
            "grid_hash": grid_hash,
            "grid_task_id": int(grid_task_id),
            "grid_batch_hash": grid_batch_hash,
            "shard_id": shard_id,
            "scheduler": dict(resolved_provenance.get("scheduler", {})),
        },
        "aggregation_revision": "silenttwin.agentdojo.aggregate.v1",
    }


def publish_completed_run(
    *,
    store: AgentDojoCheckpointStore,
    failures: Sequence[Mapping[str, Any]],
    started_at: str,
    completed_at: str,
    grid_hash: str,
    grid_task_id: int,
    shard_id: str,
    grid_batch_hash: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish one complete shard from validated checkpoints."""

    rows_by_id = store.mark_complete()
    records = [rows_by_id[item] for item in store.expected_trial_ids]
    materialized_failures = [json.loads(canonical_json(dict(item))) for item in failures]
    atomic_write_objects_jsonl(store.output_dir / RESULT_FILENAME, records)
    atomic_write_objects_jsonl(store.output_dir / FAILURES_FILENAME, materialized_failures)
    resolved_provenance = dict(provenance or collect_provenance())
    manifest = make_run_manifest(
        store.config,
        output_dir=store.output_dir,
        records=records,
        failures=materialized_failures,
        provenance=resolved_provenance,
        started_at=started_at,
        completed_at=completed_at,
        grid_hash=grid_hash,
        grid_task_id=grid_task_id,
        shard_id=shard_id,
        grid_batch_hash=grid_batch_hash,
    )
    _atomic_write_text(
        store.output_dir / LOG_FILENAME,
        "\n".join(
            (
                "environment_backend=agentdojo",
                f"configuration_hash={store.config.configuration_hash}",
                f"grid_hash={grid_hash}",
                f"grid_task_id={grid_task_id}",
                f"shard_id={shard_id}",
                "status=complete",
                "",
            )
        ),
    )
    # The complete manifest is the publication marker and is therefore always
    # renamed last, after checkpoints, result/failure streams, and the log.
    atomic_write_json(store.output_dir / MANIFEST_FILENAME, manifest)
    return manifest


def validate_completed_run(
    output_dir: Path | str,
    *,
    expected_config: AgentDojoExperimentConfig | None = None,
    expected_grid_hash: str | None = None,
    expected_shard_id: str | None = None,
    expected_source_tree_hash: str | None = None,
) -> dict[str, Any]:
    """Validate a completed shard for safe idempotent reuse."""

    directory = Path(output_dir)
    manifest = _read_object(directory / MANIFEST_FILENAME, label="AgentDojo run manifest")
    if manifest.get("manifest_schema_version") != AGENTDOJO_MANIFEST_SCHEMA:
        raise ResultValidationError("incompatible AgentDojo run-manifest schema")
    if manifest.get("result_schema_version") != AGENTDOJO_RESULT_SCHEMA:
        raise ResultValidationError("incompatible AgentDojo result schema")
    if manifest.get("status") != "complete" or manifest.get(
        "environment_backend"
    ) != "agentdojo":
        raise ResultValidationError("AgentDojo run is not complete")
    scientific = manifest.get("configuration")
    if not isinstance(scientific, Mapping):
        raise ResultValidationError("AgentDojo manifest lacks scientific configuration")
    config = AgentDojoExperimentConfig.from_mapping(scientific)
    if dict(scientific) != config.scientific_dict():
        raise ResultValidationError("AgentDojo manifest scientific configuration is not canonical")
    if manifest.get("configuration_hash") != config.configuration_hash:
        raise ResultValidationError("AgentDojo manifest configuration hash is invalid")
    if manifest.get("experiment_id") != config.experiment_id or manifest.get(
        "tier2_track"
    ) != config.tier2_track:
        raise ResultValidationError("AgentDojo manifest identity differs from its configuration")
    _validate_evidence_boundary(manifest, config=config, label="AgentDojo manifest")
    if expected_config is not None and config.configuration_hash != expected_config.configuration_hash:
        raise ResultValidationError("existing AgentDojo run is a configuration collision")
    orchestration = manifest.get("orchestration")
    if not isinstance(orchestration, Mapping):
        raise ResultValidationError("AgentDojo manifest lacks orchestration binding")
    if expected_grid_hash is not None and orchestration.get("grid_hash") != expected_grid_hash:
        raise ResultValidationError("existing AgentDojo run belongs to another grid")
    if expected_shard_id is not None and orchestration.get("shard_id") != expected_shard_id:
        raise ResultValidationError("existing AgentDojo run belongs to another shard")
    if manifest.get("result_file") != RESULT_FILENAME or manifest.get(
        "failures_file"
    ) != FAILURES_FILENAME:
        raise ResultValidationError("AgentDojo manifest references non-canonical stream names")
    if manifest.get("checkpoint_manifest") != CHECKPOINT_MANIFEST_FILENAME:
        raise ResultValidationError("AgentDojo manifest references another checkpoint manifest")
    if manifest.get("resolved_scenario_ids") != list(config.scenario_ids) or manifest.get(
        "resolved_scenario_count"
    ) != len(config.scenario_ids):
        raise ResultValidationError("AgentDojo manifest scenario binding is inconsistent")
    if manifest.get("structural_group_ids") != list(config.structural_group_ids):
        raise ResultValidationError("AgentDojo manifest structural binding is inconsistent")
    expected_agentdojo = {
        "package_version": config.agentdojo_package_version,
        "source_revision": config.agentdojo_source_revision,
        "benchmark_version": config.agentdojo_benchmark_version,
        "catalog_hash": config.agentdojo_catalog_hash,
        "scenario_registry_revision": config.scenario_registry_revision,
        "scenario_registry_hash": config.scenario_registry_hash,
        "split_manifest_hash": config.split_manifest_hash,
        "candidate_strategy_catalog_hash": config.candidate_strategy_catalog_hash,
        "pair_registry_hash": config.pair_registry_hash,
    }
    if manifest.get("agentdojo") != expected_agentdojo:
        raise ResultValidationError("AgentDojo manifest release/upstream binding is inconsistent")
    expected_scientific_bindings = {
        "analysis_plan_hash": config.analysis_plan_hash,
        "dependency_lock_hash": config.dependency_lock_hash,
        "sample_size_freeze_hash": config.sample_size_freeze_hash,
        "development_evidence_hash": config.development_evidence_hash,
        "primary_contrast_id": config.primary_contrast_id,
        "tool_protocol_revision": config.tool_protocol_revision,
        "system_prompt_hash": config.system_prompt_hash,
        "monitor_profile_hash": config.monitor_profile_hash,
    }
    if manifest.get("scientific_bindings") != expected_scientific_bindings:
        raise ResultValidationError("AgentDojo manifest scientific bindings are inconsistent")
    result_path = directory / RESULT_FILENAME
    failures_path = directory / FAILURES_FILENAME
    if manifest.get("result_sha256") != sha256_file(result_path):
        raise ResultValidationError("AgentDojo result digest mismatch")
    if manifest.get("failures_sha256") != sha256_file(failures_path):
        raise ResultValidationError("AgentDojo failure-ledger digest mismatch")
    records = read_jsonl(result_path)
    if manifest.get("actual_trial_count") != len(records) or manifest.get(
        "expected_trial_count"
    ) != len(records):
        raise ResultValidationError("AgentDojo result count differs from manifest")
    trial_ids: set[str] = set()
    normalized_records: dict[str, dict[str, Any]] = {}
    for row in records:
        trial_id = str(row.get("trial_id", ""))
        if not trial_id or trial_id in trial_ids:
            raise ResultValidationError("AgentDojo result has invalid trial IDs")
        trial_ids.add(trial_id)
        normalized_records[trial_id] = _validate_sample(
            row, config=config, trial_id=trial_id
        )
    failures = read_jsonl(failures_path)
    if manifest.get("failure_count") != len(failures):
        raise ResultValidationError("AgentDojo failure count differs from manifest")
    for index, failure in enumerate(failures):
        _validate_evidence_boundary(
            failure,
            config=config,
            label=f"AgentDojo failure row {index}",
        )
    checkpoint = _read_object(
        directory / CHECKPOINT_MANIFEST_FILENAME,
        label="AgentDojo checkpoint manifest",
    )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ResultValidationError("AgentDojo manifest lacks provenance")
    validated_provenance = _validated_provenance(provenance, config=config)
    if dict(provenance) != validated_provenance:
        raise ResultValidationError(
            "AgentDojo manifest lacks explicit model-free runtime provenance"
        )
    source_tree_hash = provenance.get("source_tree_hash")
    if not isinstance(source_tree_hash, str) or not source_tree_hash:
        raise ResultValidationError("AgentDojo manifest provenance lacks a source-tree hash")
    if (
        expected_source_tree_hash is not None
        and source_tree_hash != expected_source_tree_hash
    ):
        raise ResultValidationError(
            "existing AgentDojo run belongs to another source tree"
        )
    expected_trials = checkpoint.get("expected_trial_ids")
    if not isinstance(expected_trials, list):
        raise ResultValidationError("AgentDojo checkpoint manifest lacks expected trials")
    _validate_checkpoint_manifest_document(
        checkpoint,
        configuration_hash=config.configuration_hash,
        source_tree_hash=source_tree_hash,
        expected_trial_ids=[str(item) for item in expected_trials],
    )
    if checkpoint.get("status") != "complete":
        raise ResultValidationError("AgentDojo completed run has invalid checkpoints")
    if manifest.get("checkpoint_manifest_hash") != stable_hash(checkpoint):
        raise ResultValidationError("AgentDojo run/checkpoint manifest binding is invalid")
    if [str(item) for item in expected_trials] != [
        str(row.get("trial_id")) for row in records
    ]:
        raise ResultValidationError("AgentDojo result order/cohort differs from checkpoints")
    checkpoint_samples = _load_published_checkpoint_samples(
        directory, config=config, manifest=checkpoint
    )
    if checkpoint_samples != normalized_records:
        raise ResultValidationError("AgentDojo results differ from completed checkpoints")
    log = (directory / LOG_FILENAME).read_text(encoding="utf-8")
    log_lines = set(log.splitlines())
    required_log_lines = {
        "status=complete",
        f"configuration_hash={config.configuration_hash}",
        f"grid_hash={orchestration.get('grid_hash')}",
        f"grid_task_id={orchestration.get('grid_task_id')}",
        f"shard_id={orchestration.get('shard_id')}",
    }
    if not required_log_lines <= log_lines:
        raise ResultValidationError("AgentDojo run log does not record completion")
    return manifest


__all__ = [
    "AgentDojoCheckpointStore",
    "BENCHMARK_EVIDENCE_CLASS",
    "CHECKPOINT_DIRECTORY",
    "CHECKPOINT_MANIFEST_FILENAME",
    "CHECKPOINT_SCHEMA",
    "ENGINEERING_SMOKE_EVIDENCE_CLASS",
    "FAILURES_FILENAME",
    "LOG_FILENAME",
    "MANIFEST_FILENAME",
    "RESULT_FILENAME",
    "bind_evidence_boundary",
    "evidence_boundary_for_config",
    "make_run_manifest",
    "publish_completed_run",
    "trial_checkpoint_id",
    "validate_completed_run",
]
