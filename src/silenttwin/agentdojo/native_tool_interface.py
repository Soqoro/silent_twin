"""Adaptive train-only qualification of Qwen's native tool-call interface.

The preceding clean-repair component run is immutable and remains scored under
its failed strict-JSON contract.  This module issues new calls through the
exact native tool template embedded in the pinned Qwen tokenizer.  Its primary
endpoint is interface realization, not utility.  Development and test are
never available to this workflow.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from silenttwin.agentdojo.canonical import (
    CanonicalToolCall,
    CanonicalToolSchema,
    canonicalize_tool_call,
    strict_json_loads,
)
from silenttwin.agentdojo.clean_repair import (
    EXPECTED_SUITE_GROUP_COUNTS,
    EXPECTED_TASK_COUNT,
    QWEN_REPAIR_IDENTITY,
    REPAIR_SYSTEM_PROMPT,
    SANITIZED_HANDOFF_RECORD,
    _call_semantics,
    _cell_summary,
    _execute_calls,
    _json_projection,
    _live_task,
    _model_provenance_valid,
    load_inputs as load_clean_repair_inputs,
    validate_task_records,
)
from silenttwin.agentdojo.config import (
    AGENTDOJO_SUITES,
    ModelIdentity,
    canonical_json,
    require_hash,
    stable_hash,
)
from silenttwin.agentdojo.pipeline import ModelCallRecord, _response_metadata
from silenttwin.agentdojo.runtime_integrity import (
    capture_learned_runtime_provenance,
)
from silenttwin.agentdojo.visibility import PublicEnvelope, public_value
from silenttwin.io.jsonl import (
    atomic_write_json,
    atomic_write_objects_jsonl,
    read_jsonl,
    sha256_file,
)
from silenttwin.io.provenance import collect_provenance
from silenttwin.schemas import stable_digest


PROTOCOL_SCHEMA = "silenttwin.agentdojo.native_tool_interface_protocol.v1"
INPUT_SCHEMA = "silenttwin.agentdojo.native_tool_interface_inputs.v1"
CHECKPOINT_SCHEMA = "silenttwin.agentdojo.native_tool_interface_checkpoint.v1"
RUN_MANIFEST_SCHEMA = "silenttwin.agentdojo.native_tool_interface_run.v1"
ANALYSIS_SCHEMA = "silenttwin.agentdojo.native_tool_interface_analysis.v1"
PROTOCOL_REVISION = "scientific-v6-native-tool-interface-train-v1"
ARM = "fresh_repair_restricted_native_tool_chat"
NATIVE_PROTOCOL = "qwen-native-tool-chat.v1"
QWEN_CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
LEARNED_RUNTIME_FINGERPRINT = (
    "sha256:75aab14cb46e74baf8c16308069f99e1d4df497915450f47fc4a1846889f965c"
)
NATIVE_PROMPT_BINDING_HASH = stable_hash(
    {
        "system_prompt": REPAIR_SYSTEM_PROMPT,
        "sanitized_handoff": SANITIZED_HANDOFF_RECORD,
        "interface_protocol": NATIVE_PROTOCOL,
        "chat_template_sha256": QWEN_CHAT_TEMPLATE_SHA256,
        "tool_definition_format": "openai_function_v1",
        "tool_response_format": "qwen_xml_tool_call_v1",
    }
)
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


class NativeToolInterfaceError(RuntimeError):
    """A native-interface artifact or execution boundary is inconsistent."""


def _raw_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_object(path: Path | str, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NativeToolInterfaceError(f"invalid {label} {candidate}: {exc}") from exc
    if not isinstance(value, dict):
        raise NativeToolInterfaceError(f"{label} is not one JSON object: {candidate}")
    return value


def _self_hash(
    document: Mapping[str, Any], *, field: str, label: str
) -> str:
    payload = dict(document)
    recorded = payload.pop(field, None)
    if not isinstance(recorded, str):
        raise NativeToolInterfaceError(f"{label} lacks {field}")
    try:
        require_hash(field, recorded)
    except ValueError as exc:
        raise NativeToolInterfaceError(f"{label} has invalid {field}") from exc
    if recorded != stable_hash(payload):
        raise NativeToolInterfaceError(f"{label} {field} mismatch")
    return recorded


def validate_protocol(document: Mapping[str, Any]) -> str:
    """Validate the immutable adaptive train-only interface protocol."""

    if (
        document.get("schema_version") != PROTOCOL_SCHEMA
        or document.get("protocol_revision") != PROTOCOL_REVISION
        or document.get("design_phase")
        != "post_clean_repair_adaptive_train_only_interface_qualification"
        or document.get("environment_backend") != "agentdojo"
    ):
        raise NativeToolInterfaceError("unsupported native-tool protocol identity")
    access = document.get("access_policy")
    if not isinstance(access, Mapping) or dict(access) != {
        "execution_permitted_splits": ["train"],
        "development_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "development_submission_permitted": False,
        "held_out_evaluation_permitted": False,
        "confirmatory_claim_permitted": False,
    }:
        raise NativeToolInterfaceError("native-tool protocol crosses its train boundary")
    design = document.get("design")
    if not isinstance(design, Mapping) or (
        design.get("arm") != ARM
        or design.get("expected_task_count") != EXPECTED_TASK_COUNT
        or design.get("expected_checkpoint_count") != EXPECTED_TASK_COUNT
        or design.get("max_turns") != 12
        or design.get("max_calls_per_turn") != 8
        or design.get("pilot_max_new_tasks") != 1
        or design.get("pilot_continuation_rule")
        != "identity_boundary_and_replay_integrity_only"
    ):
        raise NativeToolInterfaceError("native-tool design differs from executable code")
    analysis = document.get("analysis")
    if not isinstance(analysis, Mapping) or (
        analysis.get("independent_unit") != "structural_group_id"
        or analysis.get("suite_weighting") != "equal_suite"
        or analysis.get("confidence_level") != 0.95
        or analysis.get("bootstrap_resamples") != 5000
        or analysis.get("minimum_episode_valid_rate") != 0.9
        or analysis.get("minimum_episode_valid_ci_lower") != 0.8
        or analysis.get("utility_is_progression_gate") is not False
    ):
        raise NativeToolInterfaceError("native-tool analysis plan drifted")
    interface = document.get("native_interface")
    if not isinstance(interface, Mapping) or dict(interface) != {
        "protocol": NATIVE_PROTOCOL,
        "tokenizer_chat_template_sha256": QWEN_CHAT_TEMPLATE_SHA256,
        "tool_definition_format": "openai_function_v1",
        "tool_response_format": "qwen_xml_tool_call_v1",
        "assistant_preamble_before_tool_calls_permitted": True,
        "markdown_or_json_repair_permitted": False,
        "retry_on_contract_failure": False,
    }:
        raise NativeToolInterfaceError("native-tool interface contract drifted")
    model = document.get("model")
    expected_model = {
        **dict(QWEN_REPAIR_IDENTITY),
        "scientific_function": "native_tool_interface_qualification",
        "prompt_hashes": {ARM: NATIVE_PROMPT_BINDING_HASH},
        "tokenizer_chat_template_sha256": QWEN_CHAT_TEMPLATE_SHA256,
        "runtime_fingerprint": LEARNED_RUNTIME_FINGERPRINT,
    }
    if not isinstance(model, Mapping) or dict(model) != expected_model:
        raise NativeToolInterfaceError("native-tool model identity drifted")
    upstream = document.get("upstream_bindings")
    required_upstream = {
        "clean_repair_input_file_sha256",
        "clean_repair_input_metadata_hash",
        "clean_repair_task_records_hash",
        "failed_analysis_file_sha256",
        "failed_analysis_hash",
        "failed_result_sha256",
    }
    if not isinstance(upstream, Mapping) or set(upstream) != required_upstream:
        raise NativeToolInterfaceError("native-tool protocol lacks exact upstream bindings")
    for field in required_upstream:
        try:
            require_hash(field, str(upstream[field]))
        except ValueError as exc:
            raise NativeToolInterfaceError(f"invalid upstream binding {field}") from exc
    return _self_hash(document, field="protocol_hash", label="native-tool protocol")


def _native_tool_definitions(
    schemas: Sequence[CanonicalToolSchema],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": _json_projection(schema.parameters),
            },
        }
        for schema in schemas
    )


def _initial_messages(user_prompt: str) -> tuple[dict[str, Any], ...]:
    return (
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": canonical_json(
                {
                    "record_type": "sanitized_repair_handoff",
                    **dict(SANITIZED_HANDOFF_RECORD),
                }
            ),
        },
        {"role": "user", "content": user_prompt},
    )


def parse_native_response(
    text: str, *, turn: int, max_calls_per_turn: int
) -> tuple[str, tuple[CanonicalToolCall, ...]]:
    """Parse the exact tool syntax documented by the pinned Qwen template."""

    if not isinstance(text, str) or not text.strip():
        raise NativeToolInterfaceError("native tool response is empty")
    matches = tuple(_TOOL_CALL_PATTERN.finditer(text))
    open_count = text.count("<tool_call>")
    close_count = text.count("</tool_call>")
    if open_count != close_count or open_count != len(matches):
        raise NativeToolInterfaceError("native tool response has unmatched tags")
    if not matches:
        if "<tool_call" in text or "</tool_call" in text:
            raise NativeToolInterfaceError("native tool response has malformed tags")
        return text.strip(), ()
    if len(matches) > max_calls_per_turn:
        raise NativeToolInterfaceError("native tool response exceeds the call bound")
    remainder_parts: list[str] = []
    cursor = 0
    calls: list[CanonicalToolCall] = []
    for index, match in enumerate(matches):
        remainder_parts.append(text[cursor : match.start()])
        cursor = match.end()
        try:
            value = strict_json_loads(match.group(1))
        except (json.JSONDecodeError, ValueError) as exc:
            raise NativeToolInterfaceError(
                "native tool-call payload is not strict JSON"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"name", "arguments"}
            or not isinstance(value.get("name"), str)
            or not value.get("name")
            or not isinstance(value.get("arguments"), Mapping)
        ):
            raise NativeToolInterfaceError(
                'native tool-call payload must be exactly {"name","arguments"}'
            )
        calls.append(
            canonicalize_tool_call(
                {
                    "function": value["name"],
                    "arguments": value["arguments"],
                },
                default_id=f"turn-{turn}-call-{index}",
            )
        )
    remainder_parts.append(text[cursor:])
    preamble = "".join(remainder_parts).strip()
    if "<tool_call" in preamble or "</tool_call" in preamble:
        raise NativeToolInterfaceError("native tool response has stray tags")
    return preamble, tuple(calls)


@dataclass(frozen=True, slots=True)
class NativeToolLoopResult:
    messages: tuple[dict[str, Any], ...]
    output_text: str
    traces: tuple[CanonicalToolCall, ...]
    model_calls: tuple[ModelCallRecord, ...]
    terminated: bool
    error: str | None = None


def _assistant_message(
    content: str, calls: Sequence[CanonicalToolCall]
) -> dict[str, Any]:
    if not calls:
        return {"role": "assistant", "content": content}
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": call.function,
                    "arguments": dict(call.arguments),
                },
            }
            for call in calls
        ],
    }


def run_native_tool_loop(
    *,
    model_client: Any,
    initial_messages: Sequence[Mapping[str, Any]],
    tool_schemas: Sequence[CanonicalToolSchema],
    execute_call: Callable[[CanonicalToolCall], Any],
    seed_for_turn: Callable[[int], int],
    max_turns: int,
    max_calls_per_turn: int,
    max_tokens: int,
) -> NativeToolLoopResult:
    """Run a bounded native Qwen tool loop with every transport byte retained."""

    if not callable(getattr(model_client, "complete_tool_chat", None)):
        raise TypeError("native tool loop requires complete_tool_chat()")
    if max_turns <= 0 or max_calls_per_turn <= 0 or max_tokens <= 0:
        raise ValueError("native tool-loop bounds must be positive")
    messages = [deepcopy(dict(message)) for message in initial_messages]
    tools = list(_native_tool_definitions(tool_schemas))
    known_tools = {schema.name for schema in tool_schemas}
    traces: list[CanonicalToolCall] = []
    records: list[ModelCallRecord] = []
    for turn in range(max_turns):
        request = {
            "protocol": NATIVE_PROTOCOL,
            "messages": deepcopy(messages),
            "tools": deepcopy(tools),
        }
        protocol_prompt = canonical_json(request)
        client_input = canonical_json({"messages": messages, "tools": tools})
        seed = int(seed_for_turn(turn))
        started = time.perf_counter()
        response: Any | None = None
        text: str | None = None
        parsed_output: dict[str, Any] | None = None
        metadata: dict[str, Any] = {}
        try:
            response = model_client.complete_tool_chat(
                messages, tools, seed=seed, max_tokens=max_tokens
            )
            raw = getattr(response, "text", response)
            if not isinstance(raw, str):
                raise NativeToolInterfaceError("native model response has no text")
            text = raw
            metadata = _response_metadata(response, role="victim")
            content, calls = parse_native_response(
                text, turn=turn, max_calls_per_turn=max_calls_per_turn
            )
            parsed_output = {
                "assistant_preamble": content if calls else "",
                "content": content if not calls else None,
                "tool_calls": [call.to_dict() for call in calls],
            }
            unknown = {call.function for call in calls} - known_tools
            if unknown:
                raise NativeToolInterfaceError(
                    f"native response names unknown tools: {sorted(unknown)}"
                )
            rendered = str(metadata.get("rendered_input", client_input))
            records.append(
                ModelCallRecord(
                    phase="native_tool_turn",
                    call_index=turn,
                    seed=seed,
                    canonical_input_hash=stable_digest(request),
                    rendered_input_hash=_raw_sha256(rendered),
                    raw_response_hash=_raw_sha256(text),
                    parsed_output_hash=stable_digest(parsed_output),
                    canonical_input=request,
                    protocol_prompt=protocol_prompt,
                    rendered_chat_template_input=rendered,
                    raw_response=text,
                    parsed_output=parsed_output,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    metadata=metadata,
                )
            )
        except Exception as exc:
            failure_provider = getattr(model_client, "failure_metadata", None)
            failure = dict(failure_provider()) if callable(failure_provider) else {}
            if response is not None and not metadata:
                metadata = _response_metadata(response, role="victim")
            rendered = str(
                metadata.get("rendered_input", failure.get("rendered_input", client_input))
            )
            records.append(
                ModelCallRecord(
                    phase="native_tool_turn",
                    call_index=turn,
                    seed=seed,
                    canonical_input_hash=stable_digest(request),
                    rendered_input_hash=_raw_sha256(rendered),
                    raw_response_hash=_raw_sha256(text) if text is not None else None,
                    parsed_output_hash=(
                        stable_digest(parsed_output)
                        if parsed_output is not None
                        else None
                    ),
                    canonical_input=request,
                    protocol_prompt=protocol_prompt,
                    rendered_chat_template_input=rendered,
                    raw_response=text,
                    parsed_output=parsed_output,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    failure_metadata=failure,
                    metadata=metadata,
                    error=f"{type(exc).__name__}:{exc}",
                )
            )
            return NativeToolLoopResult(
                tuple(messages), "", tuple(traces), tuple(records), False,
                f"{type(exc).__name__}:{exc}",
            )
        messages.append(_assistant_message(content, calls))
        if not calls:
            return NativeToolLoopResult(
                tuple(messages), content, tuple(traces), tuple(records), True
            )
        try:
            for call in calls:
                result = execute_call(call)
                envelope = result if isinstance(result, PublicEnvelope) else public_value(result)
                messages.append(
                    {"role": "tool", "content": canonical_json(envelope.unwrap())}
                )
                traces.append(call)
        except Exception as exc:
            return NativeToolLoopResult(
                tuple(messages), "", tuple(traces), tuple(records), False,
                f"{type(exc).__name__}:{exc}",
            )
    return NativeToolLoopResult(
        tuple(messages), "", tuple(traces), tuple(records), False,
        "maximum_native_tool_turns_exceeded",
    )


class _FreshNativeSession:
    def __init__(self, client: Any, *, context_id: str) -> None:
        self._client: Any | None = client
        self.context_id = context_id
        self.retired = False

    def complete_tool_chat(self, *args: Any, **kwargs: Any) -> Any:
        if self.retired or self._client is None:
            raise NativeToolInterfaceError("native repair context is retired")
        return self._client.complete_tool_chat(*args, **kwargs)

    def failure_metadata(self) -> dict[str, Any]:
        provider = getattr(self._client, "failure_metadata", None)
        value = dict(provider()) if callable(provider) else {}
        value.update(
            {
                "repair_context_id": self.context_id,
                "fresh_message_state": True,
                "shared_read_only_model_weights": True,
            }
        )
        return value

    def retire(self) -> None:
        self._client = None
        self.retired = True


def _native_prompt_binding_valid(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> bool:
    if not records:
        return False
    expected_prefix = [deepcopy(dict(message)) for message in initial_messages]
    expected_tools = [deepcopy(dict(tool)) for tool in tools]
    for index, record in enumerate(records):
        request = record.get("canonical_input")
        metadata = record.get("metadata")
        failure = record.get("failure_metadata")
        combined = {
            **(dict(failure) if isinstance(failure, Mapping) else {}),
            **(dict(metadata) if isinstance(metadata, Mapping) else {}),
        }
        if (
            not isinstance(request, Mapping)
            or set(request) != {"protocol", "messages", "tools"}
            or request.get("protocol") != NATIVE_PROTOCOL
            or request.get("tools") != expected_tools
            or not isinstance(request.get("messages"), list)
            or request["messages"][: len(expected_prefix)] != expected_prefix
            or record.get("call_index") != index
            or record.get("canonical_input_hash") != stable_digest(request)
            or record.get("protocol_prompt") != canonical_json(request)
            or combined.get("input_mode") != "native_tool_chat"
            or combined.get("input_messages") != request["messages"]
            or combined.get("input_tools") != expected_tools
            or combined.get("chat_template_hash") != QWEN_CHAT_TEMPLATE_SHA256
        ):
            return False
        client_input = canonical_json(
            {"messages": request["messages"], "tools": expected_tools}
        )
        rendered = record.get("rendered_chat_template_input")
        if (
            not isinstance(rendered, str)
            or record.get("rendered_input_hash") != _raw_sha256(rendered)
            or combined.get("input_prompt_hash") != _raw_sha256(client_input)
            or combined.get("native_tool_chat_input_hash")
            != _raw_sha256(client_input)
            or combined.get("input_messages_hash")
            != _raw_sha256(canonical_json(request["messages"]))
            or combined.get("input_tools_hash")
            != _raw_sha256(canonical_json(expected_tools))
            or combined.get("rendered_input_hash") != _raw_sha256(rendered)
        ):
            return False
    return True


def freeze_inputs(
    *,
    protocol_path: Path,
    upstream_input_path: Path,
    failed_analysis_path: Path,
    dependency_lock_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind the failed train result and clean task corpus before new calls."""

    if output_path.exists():
        raise NativeToolInterfaceError(
            f"refusing to overwrite frozen native-tool inputs: {output_path}"
        )
    protocol = _load_object(protocol_path, label="native-tool protocol")
    protocol_hash = validate_protocol(protocol)
    upstream = protocol["upstream_bindings"]
    if sha256_file(upstream_input_path) != upstream["clean_repair_input_file_sha256"]:
        raise NativeToolInterfaceError("clean-repair input bytes changed")
    clean_metadata, tasks = load_clean_repair_inputs(upstream_input_path)
    if (
        clean_metadata.get("metadata_hash")
        != upstream["clean_repair_input_metadata_hash"]
        or clean_metadata.get("task_records_hash")
        != upstream["clean_repair_task_records_hash"]
        or clean_metadata.get("development_outcomes_inspected") is not False
        or clean_metadata.get("test_outcomes_inspected") is not False
    ):
        raise NativeToolInterfaceError("clean-repair input metadata binding changed")
    if sha256_file(failed_analysis_path) != upstream["failed_analysis_file_sha256"]:
        raise NativeToolInterfaceError("failed analysis bytes changed")
    failed_analysis = _load_object(failed_analysis_path, label="failed analysis")
    if (
        _self_hash(failed_analysis, field="analysis_hash", label="failed analysis")
        != upstream["failed_analysis_hash"]
        or failed_analysis.get("result_sha256") != upstream["failed_result_sha256"]
        or failed_analysis.get("train_component_feasibility_supported") is not False
        or failed_analysis.get("development_submission_permitted") is not False
        or failed_analysis.get("held_out_evaluation_permitted") is not False
        or failed_analysis.get("development_outcomes_inspected") is not False
        or failed_analysis.get("test_outcomes_inspected") is not False
    ):
        raise NativeToolInterfaceError("failed analysis does not authorize this train gate")
    provenance = collect_provenance()
    if provenance.get("code_dirty") is not False:
        raise NativeToolInterfaceError("native-tool input freeze requires clean Git")
    runtime = capture_learned_runtime_provenance(
        dependency_lock_path,
        expected_runtime_fingerprints={str(protocol["model"]["runtime_fingerprint"])},
    )
    metadata_payload = {
        "schema_version": INPUT_SCHEMA,
        "record_type": "metadata",
        "protocol_hash": protocol_hash,
        "protocol_file_sha256": sha256_file(protocol_path),
        "source_tree_hash": provenance["source_tree_hash"],
        "code_revision": provenance["code_revision"],
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "learned_runtime_provenance": runtime,
        "model": deepcopy(protocol["model"]),
        "upstream_input_file_sha256": sha256_file(upstream_input_path),
        "upstream_input_metadata_hash": clean_metadata["metadata_hash"],
        "upstream_protocol_hash": clean_metadata["protocol_hash"],
        "task_records_hash": clean_metadata["task_records_hash"],
        "failed_analysis_file_sha256": sha256_file(failed_analysis_path),
        "failed_analysis_hash": failed_analysis["analysis_hash"],
        "failed_result_sha256": failed_analysis["result_sha256"],
        "task_count": len(tasks),
        "expected_checkpoint_count": len(tasks),
        "maximum_model_call_count": len(tasks) * int(protocol["design"]["max_turns"]),
        "suite_task_counts": deepcopy(clean_metadata["suite_task_counts"]),
        "oracle_strict_utility_rate": clean_metadata["oracle_strict_utility_rate"],
        "oracle_tool_error_count": clean_metadata["oracle_tool_error_count"],
        "private_or_adversarial_fields_present": False,
        "development_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "confirmatory_claim_permitted": False,
        "external_api_calls": 0,
        "model_inference_calls": 0,
    }
    metadata = {
        **metadata_payload,
        "metadata_hash": stable_hash(metadata_payload),
    }
    atomic_write_objects_jsonl(output_path, [metadata, *tasks])
    output_path.chmod(0o444)
    return {
        "output": str(output_path),
        "file_sha256": sha256_file(output_path),
        "metadata_hash": metadata["metadata_hash"],
        "task_records_hash": metadata["task_records_hash"],
        "task_count": len(tasks),
        "expected_checkpoint_count": len(tasks),
        "development_outcomes_inspected": False,
        "test_outcomes_inspected": False,
    }


def load_inputs(path: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    if not rows or rows[0].get("record_type") != "metadata":
        raise NativeToolInterfaceError("native-tool inputs lack metadata")
    metadata = dict(rows[0])
    payload = dict(metadata)
    recorded = payload.pop("metadata_hash", None)
    if recorded != stable_hash(payload):
        raise NativeToolInterfaceError("native-tool input metadata hash mismatch")
    tasks = [dict(row) for row in rows[1:]]
    if (
        metadata.get("schema_version") != INPUT_SCHEMA
        or metadata.get("task_count") != EXPECTED_TASK_COUNT
        or metadata.get("expected_checkpoint_count") != EXPECTED_TASK_COUNT
        or len(tasks) != EXPECTED_TASK_COUNT
        or metadata.get("task_records_hash") != stable_hash(tasks)
        or metadata.get("private_or_adversarial_fields_present") is not False
        or metadata.get("development_outcomes_inspected") is not False
        or metadata.get("test_outcomes_inspected") is not False
    ):
        raise NativeToolInterfaceError("native-tool input corpus is incomplete")
    validate_task_records(
        tasks, protocol_hash=str(metadata.get("upstream_protocol_hash", ""))
    )
    return metadata, tasks


def _checkpoint_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    return {**value, "checkpoint_hash": stable_hash(value)}


def _run_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    return {**value, "run_manifest_hash": stable_hash(value)}


def _model_identity(protocol: Mapping[str, Any], runtime_fingerprint: str) -> ModelIdentity:
    model = protocol["model"]
    return ModelIdentity(
        role="victim",
        implementation=str(model["implementation"]),
        model_id=str(model["model_id"]),
        model_revision=str(model["model_revision"]),
        tokenizer_revision=str(model["tokenizer_revision"]),
        checkpoint_fingerprint=str(model["checkpoint_fingerprint"]),
        runtime_fingerprint=runtime_fingerprint,
        prompt_hash=str(model["prompt_hashes"][ARM]),
        dtype=str(model["dtype"]),
        temperature=float(model["temperature"]),
        top_p=float(model["top_p"]),
        max_new_tokens=int(model["max_new_tokens"]),
    )


def _usage_total(call_records: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for call in call_records:
        metadata = call.get("metadata")
        usage = metadata.get("usage") if isinstance(metadata, Mapping) else None
        value = usage.get("total_tokens") if isinstance(usage, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            total += value
    return total


def run_trial(
    *,
    protocol: Mapping[str, Any],
    task_record: Mapping[str, Any],
    model_client: Any,
    compat: Any,
    task_records_hash: str,
) -> dict[str, Any]:
    """Run one restricted native-tool episode and an atomic final replay."""

    protocol_hash = validate_protocol(protocol)
    try:
        require_hash("task_records_hash", task_records_hash)
    except ValueError as exc:
        raise NativeToolInterfaceError("trial lacks its task-corpus binding") from exc
    if task_record.get("dataset_split") != "train":
        raise NativeToolInterfaceError("native-tool trial is not train-only")
    suite, task, initial_environment, full_schemas = _live_task(
        task_record, compat=compat
    )
    initial_hash = str(compat.environment_hash(initial_environment))
    permitted = set(task_record["restricted_tool_names"])
    visible_schemas = tuple(
        schema for schema in full_schemas if schema.name in permitted
    )
    if not visible_schemas or {schema.name for schema in visible_schemas} != permitted:
        raise NativeToolInterfaceError("native-tool restricted schema is incomplete")
    trial_id = stable_hash(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "protocol_hash": protocol_hash,
            "upstream_task_id": task_record["task_id"],
            "arm": ARM,
        }
    )
    context_id = stable_hash([trial_id, "fresh-native-context"])
    speculative = compat.clone_environment(initial_environment)
    speculative_audit: list[dict[str, Any]] = []
    active = True

    def execute_speculative(call: CanonicalToolCall) -> PublicEnvelope:
        nonlocal active
        if not active:
            raise NativeToolInterfaceError("native speculative callback is retired")
        pre_hash = str(compat.environment_hash(speculative))
        try:
            outcome = compat.execute_call(suite, speculative, call.to_dict())
            error = getattr(outcome, "error", None)
            result = getattr(outcome, "result", None)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            result = None
        post_hash = str(compat.environment_hash(speculative))
        speculative_audit.append(
            {
                "call_index": len(speculative_audit),
                "call": call.to_dict(),
                "pre_environment_hash": pre_hash,
                "post_environment_hash": post_hash,
                "status": "error" if error else "ok",
                "error": str(error) if error else None,
            }
        )
        if error:
            raise NativeToolInterfaceError(f"native speculative tool error: {error}")
        return public_value(
            {
                "function": call.function,
                "status": "ok",
                "value": _json_projection(result),
                "error": None,
            }
        )

    user_prompt = str(getattr(task, "PROMPT", ""))
    initial_messages = _initial_messages(user_prompt)
    native_tools = _native_tool_definitions(visible_schemas)
    session = _FreshNativeSession(model_client, context_id=context_id)
    seed_base = int(
        stable_hash([task_record["task_id"], "native-tool-interface-seed"])[:15], 16
    )
    loop = run_native_tool_loop(
        model_client=session,
        initial_messages=initial_messages,
        tool_schemas=visible_schemas,
        execute_call=execute_speculative,
        seed_for_turn=lambda turn: seed_base + int(turn),
        max_turns=int(protocol["design"]["max_turns"]),
        max_calls_per_turn=int(protocol["design"]["max_calls_per_turn"]),
        max_tokens=int(protocol["model"]["max_new_tokens"]),
    )
    active = False
    session.retire()

    final_target = compat.load_environment(suite, task, {})
    final_start_hash = str(compat.environment_hash(final_target))
    if final_start_hash != initial_hash:
        raise NativeToolInterfaceError("native final target differs from initial state")
    locked_calls = tuple(loop.traces) if loop.error is None and loop.terminated else ()
    output_text = loop.output_text if loop.error is None and loop.terminated else ""
    working = compat.clone_environment(final_target)
    replay_traces, replay_audit = _execute_calls(
        suite=suite,
        environment=working,
        calls=locked_calls,
        compat=compat,
    )
    replay_error = any(row["status"] != "ok" for row in replay_audit) or len(
        replay_audit
    ) != len(locked_calls)
    final_environment = final_target if replay_error else working
    committed_call_count = 0 if replay_error else len(locked_calls)
    call_records = [call.to_trusted_dict() for call in loop.model_calls]
    model_protocol_valid = loop.error is None and loop.terminated
    model_provenance_valid = _model_provenance_valid(
        call_records, model=protocol["model"]
    )
    prompt_binding_valid = _native_prompt_binding_valid(
        call_records,
        initial_messages=initial_messages,
        tools=native_tools,
    )
    run_valid = bool(
        model_protocol_valid
        and model_provenance_valid
        and prompt_binding_valid
        and not replay_error
        and session.retired
    )
    raw_utility = bool(
        compat.grade_utility(
            task,
            output_text,
            compat.clone_environment(final_target),
            final_environment,
            tuple(replay_traces) if not replay_error else (),
            strict=True,
        )
    )
    utility = bool(raw_utility and run_valid)
    oracle_calls = tuple(
        canonicalize_tool_call(call, default_id=f"oracle-{index}")
        for index, call in enumerate(task_record["oracle_calls"])
    )
    attempted_function_names = [
        str(call["function"])
        for record in call_records
        if isinstance(record.get("parsed_output"), Mapping)
        for call in record["parsed_output"].get("tool_calls", [])
        if isinstance(call, Mapping) and isinstance(call.get("function"), str)
    ]
    unauthorized = sum(name not in permitted for name in attempted_function_names)
    errors = [str(loop.error)] if loop.error is not None else []
    if replay_error:
        errors.append("atomic_final_replay_error")
    if not model_provenance_valid:
        errors.append("invalid_model_provenance")
    if not prompt_binding_valid:
        errors.append("invalid_native_prompt_binding")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "trial_id": trial_id,
        "protocol_hash": protocol_hash,
        "task_records_hash": task_records_hash,
        "upstream_task_id": task_record["task_id"],
        "source_scenario_id": task_record["source_scenario_id"],
        "agentdojo_suite": task_record["suite"],
        "user_task_id": task_record["user_task_id"],
        "structural_group_id": task_record["structural_group_id"],
        "dataset_split": "train",
        "arm": ARM,
        "utility": int(utility),
        "raw_strict_utility": int(raw_utility),
        "run_valid": run_valid,
        "model_protocol_valid": model_protocol_valid,
        "model_provenance_valid": model_provenance_valid,
        "prompt_binding_valid": prompt_binding_valid,
        "tool_loop_terminated": bool(loop.terminated),
        "tool_loop_error": loop.error,
        "initial_environment_hash": initial_hash,
        "final_start_environment_hash": final_start_hash,
        "final_environment_hash": str(compat.environment_hash(final_environment)),
        "visible_tool_schema_hash": stable_hash(
            [schema.to_dict() for schema in visible_schemas]
        ),
        "visible_tool_names": [schema.name for schema in visible_schemas],
        "restricted_tool_names": list(task_record["restricted_tool_names"]),
        "locked_calls": [call.to_dict() for call in locked_calls],
        "locked_call_count": len(locked_calls),
        "committed_call_count": committed_call_count,
        "unauthorized_function_count": unauthorized,
        "restricted_function_scope_compliant": unauthorized == 0,
        "exact_oracle_call_sequence": _call_semantics(locked_calls)
        == _call_semantics(oracle_calls),
        "output_text": output_text,
        "output_hash": stable_digest(output_text),
        "native_conversation": [deepcopy(message) for message in loop.messages],
        "speculative_execution_audit": speculative_audit,
        "atomic_replay_audit": replay_audit,
        "atomic_final_replay_succeeded": not replay_error,
        "repair_context_id": context_id,
        "repair_context_retired_before_final_replay": session.retired,
        "fresh_message_state": bool(call_records),
        "fresh_kv_cache_per_model_call": True,
        "shared_read_only_model_weights": True,
        "sanitized_handoff_delivered": True,
        "prior_context_present": False,
        "prior_transcript_present": False,
        "prior_draft_present": False,
        "injection_payload_present": False,
        "private_verdict_present": False,
        "private_reason_present": False,
        "model_calls": call_records,
        "model_call_count": len(call_records),
        "parsed_model_call_count": sum(
            call.get("parsed_output") is not None for call in call_records
        ),
        "token_count": _usage_total(call_records),
        "tool_call_count": len(loop.traces),
        "attempted_function_names": attempted_function_names,
        "errors": errors,
        "external_api_calls": 0,
        "development_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "confirmatory_claim_permitted": False,
    }
    return _checkpoint_document(payload)


def expanded_jobs(
    protocol_hash: str, tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "job_id": stable_hash(
                {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "protocol_hash": protocol_hash,
                    "upstream_task_id": task["task_id"],
                    "arm": ARM,
                }
            ),
            "upstream_task_id": task["task_id"],
            "task": task,
        }
        for task in tasks
    ]


def _validate_checkpoint(
    document: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    protocol_hash: str,
    task_records_hash: str,
) -> dict[str, Any]:
    payload = dict(document)
    recorded = payload.pop("checkpoint_hash", None)
    if (
        document.get("schema_version") != CHECKPOINT_SCHEMA
        or document.get("trial_id") != job["job_id"]
        or document.get("protocol_hash") != protocol_hash
        or document.get("task_records_hash") != task_records_hash
        or document.get("upstream_task_id") != job["upstream_task_id"]
        or document.get("arm") != ARM
        or document.get("dataset_split") != "train"
        or document.get("development_outcomes_inspected") is not False
        or document.get("test_outcomes_inspected") is not False
        or document.get("external_api_calls") != 0
        or recorded != stable_hash(payload)
    ):
        raise NativeToolInterfaceError(
            f"invalid native-tool checkpoint {job['job_id']}"
        )
    return dict(document)


def run_benchmark(
    *,
    protocol_path: Path,
    input_path: Path,
    dependency_lock_path: Path,
    checkpoint_path: Path,
    model_cache_path: Path,
    output_directory: Path,
    device: str,
    max_new_tasks: int | None = None,
) -> dict[str, Any]:
    """Run or resume the scalar H200 native-interface qualification."""

    protocol = _load_object(protocol_path, label="native-tool protocol")
    protocol_hash = validate_protocol(protocol)
    metadata, tasks = load_inputs(input_path)
    if (
        metadata.get("protocol_hash") != protocol_hash
        or metadata.get("protocol_file_sha256") != sha256_file(protocol_path)
        or metadata.get("model") != protocol.get("model")
        or metadata.get("upstream_input_file_sha256")
        != protocol["upstream_bindings"]["clean_repair_input_file_sha256"]
        or metadata.get("failed_analysis_hash")
        != protocol["upstream_bindings"]["failed_analysis_hash"]
    ):
        raise NativeToolInterfaceError("native-tool inputs belong to another freeze")
    provenance = collect_provenance()
    if provenance.get("code_dirty") is not False:
        raise NativeToolInterfaceError("native-tool GPU run requires clean Git")
    for field in ("source_tree_hash", "code_revision"):
        if provenance.get(field) != metadata.get(field):
            raise NativeToolInterfaceError(
                f"native-tool GPU {field} differs from input freeze"
            )
    runtime = capture_learned_runtime_provenance(
        dependency_lock_path,
        expected_runtime_fingerprints={str(metadata["runtime_fingerprint"])},
    )
    if runtime != metadata.get("learned_runtime_provenance"):
        raise NativeToolInterfaceError("native-tool learned runtime changed")
    if not os.environ.get("PBS_JOBID") and not os.environ.get("SLURM_JOB_ID"):
        raise NativeToolInterfaceError("native-tool GPU run requires a scheduler job")
    if os.environ.get("PBS_JOBID") and os.environ.get("PBS_ENVIRONMENT") != "PBS_BATCH":
        raise NativeToolInterfaceError("native-tool PBS run requires PBS_BATCH")
    if not checkpoint_path.is_dir() or not model_cache_path.is_dir():
        raise NativeToolInterfaceError("native-tool checkpoint/cache is unavailable")
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise NativeToolInterfaceError("max_new_tasks must be positive")

    jobs = expanded_jobs(protocol_hash, tasks)
    if len(jobs) != EXPECTED_TASK_COUNT:
        raise NativeToolInterfaceError("native-tool job expansion changed")
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    expected_ids = set(jobs_by_id)
    checkpoint_directory = output_directory / "checkpoints"
    manifest_path = output_directory / "run_manifest.json"
    result_path = output_directory / "result.jsonl"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    observed = {
        path.stem for path in checkpoint_directory.glob("*.json") if path.is_file()
    }
    if observed - expected_ids:
        raise NativeToolInterfaceError("native-tool checkpoint directory has unknown jobs")
    completed: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoint_directory.glob("*.json")):
        value = _load_object(path, label="native-tool checkpoint")
        completed[path.stem] = _validate_checkpoint(
            value,
            job=jobs_by_id[path.stem],
            protocol_hash=protocol_hash,
            task_records_hash=str(metadata["task_records_hash"]),
        )
    if result_path.exists():
        if len(completed) != len(jobs):
            raise NativeToolInterfaceError("published native-tool result is incomplete")
        manifest = _load_object(manifest_path, label="native-tool run manifest")
        manifest_hash = _self_hash(
            manifest, field="run_manifest_hash", label="native-tool run manifest"
        )
        if (
            manifest.get("status") != "complete"
            or manifest.get("result_sha256") != sha256_file(result_path)
        ):
            raise NativeToolInterfaceError("published native-tool result is invalid")
        return {
            "status": "complete",
            "completed_task_count": len(completed),
            "result": str(result_path),
            "result_sha256": manifest["result_sha256"],
            "run_manifest_hash": manifest_hash,
            "reused_existing_run": True,
        }

    immutable_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "protocol_hash": protocol_hash,
        "input_file_sha256": sha256_file(input_path),
        "input_metadata_hash": metadata["metadata_hash"],
        "task_records_hash": metadata["task_records_hash"],
        "source_tree_hash": provenance["source_tree_hash"],
        "code_revision": provenance["code_revision"],
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "expected_checkpoint_count": len(jobs),
        "expected_job_ids_hash": stable_hash([job["job_id"] for job in jobs]),
        "model": deepcopy(protocol["model"]),
        "native_interface": deepcopy(protocol["native_interface"]),
    }
    if manifest_path.exists():
        existing = _load_object(manifest_path, label="native-tool run manifest")
        _self_hash(existing, field="run_manifest_hash", label="native-tool run manifest")
        if any(existing.get(key) != value for key, value in immutable_manifest.items()):
            raise NativeToolInterfaceError("existing native-tool run has another identity")
    running_payload = {
        **immutable_manifest,
        "status": "running",
        "completed_task_count": len(completed),
        "model_call_count": sum(
            int(row.get("model_call_count", 0)) for row in completed.values()
        ),
        "scheduler": provenance["scheduler"],
        "result_file": None,
        "result_sha256": None,
        "development_outcomes_inspected": False,
        "test_outcomes_inspected": False,
        "external_api_calls": 0,
    }
    atomic_write_json(manifest_path, _run_manifest(running_payload))

    incomplete = [
        str(job["upstream_task_id"])
        for job in jobs
        if str(job["job_id"]) not in completed
    ]
    if max_new_tasks is not None:
        incomplete = incomplete[:max_new_tasks]
    selected = set(incomplete)
    from . import compat
    from .assembly import model_client_from_identity

    client = None
    if selected:
        client = model_client_from_identity(
            _model_identity(protocol, str(runtime["runtime_fingerprint"])),
            checkpoint_path=checkpoint_path,
            cache_dir=model_cache_path,
            device=device,
        )
        gpu_name = str(client.failure_metadata().get("gpu_name", ""))
        if "H200" not in gpu_name.upper():
            raise NativeToolInterfaceError(
                f"native-tool gate requires NVIDIA H200, observed {gpu_name!r}"
            )
        tokenizer = client._tokenizer
        template = getattr(tokenizer, "chat_template", "") or ""
        if _raw_sha256(template) != QWEN_CHAT_TEMPLATE_SHA256:
            raise NativeToolInterfaceError("pinned Qwen chat template bytes changed")
    for job in jobs:
        job_id = str(job["job_id"])
        if job_id in completed or str(job["upstream_task_id"]) not in selected:
            continue
        assert client is not None
        result = run_trial(
            protocol=protocol,
            task_record=job["task"],
            model_client=client,
            compat=compat,
            task_records_hash=str(metadata["task_records_hash"]),
        )
        if result.get("trial_id") != job_id:
            raise NativeToolInterfaceError("native-tool trial identity changed")
        destination = checkpoint_directory / f"{job_id}.json"
        if destination.exists():
            raise NativeToolInterfaceError("refusing to overwrite native checkpoint")
        atomic_write_json(destination, result)
        destination.chmod(0o444)
        completed[job_id] = result
        running_payload.update(
            {
                "completed_task_count": len(completed),
                "model_call_count": sum(
                    int(row["model_call_count"]) for row in completed.values()
                ),
            }
        )
        atomic_write_json(manifest_path, _run_manifest(running_payload))

    if len(completed) != len(jobs):
        partial = {
            **running_payload,
            "status": "partial_integrity_pilot"
            if max_new_tasks is not None
            else "incomplete",
            "completed_task_count": len(completed),
        }
        atomic_write_json(manifest_path, _run_manifest(partial))
        return {
            "status": partial["status"],
            "completed_task_count": len(completed),
            "expected_task_count": len(jobs),
            "model_call_count": partial["model_call_count"],
            "result": None,
            "reused_existing_run": False,
        }

    ordered = [completed[str(job["job_id"])] for job in jobs]
    atomic_write_objects_jsonl(result_path, ordered)
    result_path.chmod(0o444)
    final_payload = {
        **running_payload,
        "status": "complete",
        "completed_task_count": len(ordered),
        "model_call_count": sum(int(row["model_call_count"]) for row in ordered),
        "protocol_valid_task_count": sum(
            bool(row["model_protocol_valid"]) for row in ordered
        ),
        "run_valid_task_count": sum(bool(row["run_valid"]) for row in ordered),
        "utility_success_count": sum(int(row["utility"]) for row in ordered),
        "result_file": result_path.name,
        "result_sha256": sha256_file(result_path),
    }
    final_manifest = _run_manifest(final_payload)
    atomic_write_json(manifest_path, final_manifest)
    return {
        "status": "complete",
        "completed_task_count": len(ordered),
        "model_call_count": final_payload["model_call_count"],
        "protocol_valid_task_count": final_payload["protocol_valid_task_count"],
        "run_valid_task_count": final_payload["run_valid_task_count"],
        "utility_success_count": final_payload["utility_success_count"],
        "result": str(result_path),
        "result_sha256": final_payload["result_sha256"],
        "run_manifest_hash": final_manifest["run_manifest_hash"],
        "reused_existing_run": False,
    }


def _validate_result_call_records(
    row: Mapping[str, Any], *, protocol: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    calls = row.get("model_calls")
    if (
        not isinstance(calls, list)
        or not calls
        or len(calls) > int(protocol["design"]["max_turns"])
        or row.get("model_call_count") != len(calls)
        or row.get("parsed_model_call_count")
        != sum(call.get("parsed_output") is not None for call in calls)
    ):
        raise NativeToolInterfaceError("native result has invalid model-call accounting")
    first_request = calls[0].get("canonical_input")
    if not isinstance(first_request, Mapping):
        raise NativeToolInterfaceError("native result lacks its first request")
    messages = first_request.get("messages")
    tools = first_request.get("tools")
    if (
        not isinstance(messages, list)
        or len(messages) < 3
        or not isinstance(messages[2], Mapping)
        or not isinstance(messages[2].get("content"), str)
        or stable_digest(messages[2]["content"]) != task.get("user_prompt_hash")
        or messages[:3] != list(_initial_messages(messages[2]["content"]))
        or not isinstance(tools, list)
        or not tools
    ):
        raise NativeToolInterfaceError("native first request changed its clean handoff")
    tool_names: list[str] = []
    canonical_schemas: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if (
            not isinstance(tool, Mapping)
            or set(tool) != {"type", "function"}
            or tool.get("type") != "function"
            or not isinstance(function, Mapping)
            or set(function) != {"name", "description", "parameters"}
            or not isinstance(function.get("name"), str)
            or not isinstance(function.get("description"), str)
            or not isinstance(function.get("parameters"), Mapping)
        ):
            raise NativeToolInterfaceError("native result has malformed tool definitions")
        tool_names.append(str(function["name"]))
        canonical_schemas.append(
            {
                "name": function["name"],
                "description": function["description"],
                "parameters": dict(function["parameters"]),
            }
        )
    if (
        tool_names != row.get("visible_tool_names")
        or sorted(tool_names) != task.get("restricted_tool_names")
        or stable_hash(canonical_schemas) != row.get("visible_tool_schema_hash")
    ):
        raise NativeToolInterfaceError("native result provenance/prompt binding failed")
    observed_prompt_binding = _native_prompt_binding_valid(
        calls,
        initial_messages=_initial_messages(messages[2]["content"]),
        tools=tools,
    )
    observed_model_provenance = _model_provenance_valid(
        calls, model=protocol["model"]
    )
    if (
        row.get("prompt_binding_valid") is not observed_prompt_binding
        or row.get("model_provenance_valid") is not observed_model_provenance
    ):
        raise NativeToolInterfaceError(
            "native trusted provenance flags disagree with their records"
        )
    for index, call in enumerate(calls):
        raw = call.get("raw_response")
        parsed = call.get("parsed_output")
        if (
            call.get("phase") != "native_tool_turn"
            or call.get("call_index") != index
            or (raw is not None and call.get("raw_response_hash") != _raw_sha256(raw))
            or (
                parsed is not None
                and call.get("parsed_output_hash") != stable_digest(parsed)
            )
            or (parsed is None and call.get("parsed_output_hash") is not None)
        ):
            raise NativeToolInterfaceError("native model-call hash chain failed")


def analyze_benchmark(
    *,
    protocol_path: Path,
    input_path: Path,
    run_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Analyze the immutable train gate without invoking a learned model."""

    if output_path.exists():
        raise NativeToolInterfaceError(
            f"refusing to overwrite native-tool analysis: {output_path}"
        )
    protocol = _load_object(protocol_path, label="native-tool protocol")
    protocol_hash = validate_protocol(protocol)
    metadata, tasks = load_inputs(input_path)
    if (
        metadata.get("protocol_hash") != protocol_hash
        or metadata.get("protocol_file_sha256") != sha256_file(protocol_path)
    ):
        raise NativeToolInterfaceError("native analysis input belongs to another protocol")
    manifest_path = run_directory / "run_manifest.json"
    result_path = run_directory / "result.jsonl"
    manifest = _load_object(manifest_path, label="native-tool run manifest")
    manifest_hash = _self_hash(
        manifest, field="run_manifest_hash", label="native-tool run manifest"
    )
    provenance = collect_provenance()
    if provenance.get("code_dirty") is not False:
        raise NativeToolInterfaceError("native-tool analysis requires clean Git")
    if any(
        provenance.get(field) != manifest.get(field)
        for field in ("code_revision", "source_tree_hash")
    ):
        raise NativeToolInterfaceError(
            "native-tool analysis must use the exact run implementation"
        )
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol_hash") != protocol_hash
        or manifest.get("input_file_sha256") != sha256_file(input_path)
        or manifest.get("input_metadata_hash") != metadata["metadata_hash"]
        or manifest.get("task_records_hash") != metadata["task_records_hash"]
        or manifest.get("expected_checkpoint_count") != EXPECTED_TASK_COUNT
        or manifest.get("result_file") != result_path.name
        or manifest.get("result_sha256") != sha256_file(result_path)
        or manifest.get("development_outcomes_inspected") is not False
        or manifest.get("test_outcomes_inspected") is not False
        or manifest.get("external_api_calls") != 0
    ):
        raise NativeToolInterfaceError("native-tool complete manifest is invalid")
    jobs = expanded_jobs(protocol_hash, tasks)
    rows = read_jsonl(result_path)
    if len(rows) != len(jobs):
        raise NativeToolInterfaceError("native-tool result row count is incomplete")
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_paths = sorted(checkpoint_directory.glob("*.json"))
    if len(checkpoint_paths) != len(jobs):
        raise NativeToolInterfaceError("native-tool checkpoint count is incomplete")
    checkpoints = {path.stem: path for path in checkpoint_paths}
    if set(checkpoints) != {str(job["job_id"]) for job in jobs}:
        raise NativeToolInterfaceError("native-tool checkpoint identities changed")
    boundary_failures: list[str] = []
    for row, job in zip(rows, jobs, strict=True):
        validated = _validate_checkpoint(
            row,
            job=job,
            protocol_hash=protocol_hash,
            task_records_hash=str(metadata["task_records_hash"]),
        )
        checkpoint = _load_object(
            checkpoints[str(job["job_id"])], label="native-tool checkpoint"
        )
        if checkpoint != validated:
            raise NativeToolInterfaceError("native checkpoint/result bytes disagree")
        _validate_result_call_records(
            row, protocol=protocol, task=job["task"]
        )
        boundary_ok = (
            row.get("sanitized_handoff_delivered") is True
            and row.get("prior_context_present") is False
            and row.get("prior_transcript_present") is False
            and row.get("prior_draft_present") is False
            and row.get("injection_payload_present") is False
            and row.get("private_verdict_present") is False
            and row.get("private_reason_present") is False
            and row.get("repair_context_retired_before_final_replay") is True
            and row.get("external_api_calls") == 0
            and row.get("development_outcomes_inspected") is False
            and row.get("test_outcomes_inspected") is False
        )
        if not boundary_ok:
            boundary_failures.append(str(row["trial_id"]))

    analysis_plan = protocol["analysis"]
    summaries = {
        metric: _cell_summary(
            rows,
            metric=metric,
            resamples=int(analysis_plan["bootstrap_resamples"]),
            seed=int(analysis_plan["bootstrap_seed"]) + index,
            confidence=float(analysis_plan["confidence_level"]),
        )
        for index, metric in enumerate(
            (
                "model_protocol_valid",
                "run_valid",
                "utility",
                "exact_oracle_call_sequence",
                "model_call_count",
                "tool_call_count",
                "token_count",
            )
        )
    }
    model_call_count = sum(int(row["model_call_count"]) for row in rows)
    parsed_call_count = sum(int(row["parsed_model_call_count"]) for row in rows)
    error_counts = Counter(
        str(call.get("error"))
        for row in rows
        for call in row["model_calls"]
        if call.get("error") is not None
    )
    episode = summaries["model_protocol_valid"]
    criteria = {
        "episode_valid_rate_at_least_0_90": float(episode["estimate"])
        >= float(analysis_plan["minimum_episode_valid_rate"]),
        "episode_valid_ci_lower_at_least_0_80": float(episode["ci_lower"])
        >= float(analysis_plan["minimum_episode_valid_ci_lower"]),
        "model_provenance_is_exact": all(
            row["model_provenance_valid"] is True for row in rows
        ),
        "native_prompt_binding_is_exact": all(
            row["prompt_binding_valid"] is True for row in rows
        ),
        "restricted_scope_has_no_unknown_function": sum(
            int(row["unauthorized_function_count"]) for row in rows
        )
        == 0,
        "sanitization_boundary_has_no_failures": not boundary_failures,
        "atomic_final_replay_has_no_failures": all(
            row["atomic_final_replay_succeeded"] is True for row in rows
        ),
        "upstream_oracle_strict_utility_is_one": metadata.get(
            "oracle_strict_utility_rate"
        )
        == 1.0
        and metadata.get("oracle_tool_error_count") == 0,
    }
    supported = all(criteria.values())
    payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "protocol_hash": protocol_hash,
        "input_file_sha256": sha256_file(input_path),
        "input_metadata_hash": metadata["metadata_hash"],
        "task_records_hash": metadata["task_records_hash"],
        "run_manifest_hash": manifest_hash,
        "result_sha256": manifest["result_sha256"],
        "run_code_revision": manifest["code_revision"],
        "run_source_tree_hash": manifest["source_tree_hash"],
        "analysis_code_revision": provenance["code_revision"],
        "analysis_source_tree_hash": provenance["source_tree_hash"],
        "analysis_implementation_relation": "same_as_run",
        "claim_scope": "adaptive_train_only_native_tool_interface_qualification",
        "independent_unit": "structural_group_id",
        "independent_unit_count": EXPECTED_TASK_COUNT,
        "suite_weighting": "equal_suite",
        "suite_independent_unit_counts": dict(EXPECTED_SUITE_GROUP_COUNTS),
        "task_count": len(rows),
        "model_call_count": model_call_count,
        "parsed_model_call_count": parsed_call_count,
        "turn_parse_rate": parsed_call_count / model_call_count,
        "model_call_error_counts": dict(sorted(error_counts.items())),
        "metrics": summaries,
        "boundary_failure_trial_ids": boundary_failures,
        "preregistered_interface_criteria": criteria,
        "interface_qualification_supported": supported,
        "new_clean_repair_protocol_permitted": supported,
        "utility_used_as_progression_gate": False,
        "development_submission_permitted": False,
        "held_out_evaluation_permitted": False,
        "confirmatory_claim_permitted": False,
        "development_outcomes_inspected": False,
        "test_outcomes_inspected": False,
    }
    analysis = {**payload, "analysis_hash": stable_hash(payload)}
    atomic_write_json(output_path, analysis)
    output_path.chmod(0o444)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-inputs")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--upstream-inputs", type=Path, required=True)
    freeze.add_argument("--failed-analysis", type=Path, required=True)
    freeze.add_argument("--dependency-lock", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--dependency-lock", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--model-cache", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--max-new-tasks", type=int)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--protocol", type=Path, required=True)
    analyze.add_argument("--inputs", type=Path, required=True)
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-inputs":
        result = freeze_inputs(
            protocol_path=args.protocol,
            upstream_input_path=args.upstream_inputs,
            failed_analysis_path=args.failed_analysis,
            dependency_lock_path=args.dependency_lock,
            output_path=args.output,
        )
    elif args.command == "run":
        result = run_benchmark(
            protocol_path=args.protocol,
            input_path=args.inputs,
            dependency_lock_path=args.dependency_lock,
            checkpoint_path=args.checkpoint,
            model_cache_path=args.model_cache,
            output_directory=args.output_dir,
            device=args.device,
            max_new_tasks=args.max_new_tasks,
        )
    else:
        result = analyze_benchmark(
            protocol_path=args.protocol,
            input_path=args.inputs,
            run_directory=args.run_dir,
            output_path=args.output,
        )
    print(canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "ARM",
    "CHECKPOINT_SCHEMA",
    "INPUT_SCHEMA",
    "NATIVE_PROTOCOL",
    "NativeToolInterfaceError",
    "PROTOCOL_SCHEMA",
    "QWEN_CHAT_TEMPLATE_SHA256",
    "analyze_benchmark",
    "freeze_inputs",
    "load_inputs",
    "main",
    "parse_native_response",
    "run_benchmark",
    "run_native_tool_loop",
    "run_trial",
    "validate_protocol",
]
