from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from silenttwin.agentdojo.canonical import CanonicalToolSchema
from silenttwin.agentdojo.native_tool_interface import (
    ARM,
    NATIVE_PROTOCOL,
    QWEN_CHAT_TEMPLATE_SHA256,
    NativeToolInterfaceError,
    parse_native_response,
    run_native_tool_loop,
    run_trial,
    validate_protocol,
)
from silenttwin.agentdojo.config import stable_hash
from silenttwin.schemas import canonical_json
from tests.unit.test_agentdojo_clean_repair import _RepairCompat, _live_record


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    REPO_ROOT
    / "configs/silenttwin/agentdojo/scientific-v6-native-tool-interface-train-v1.json"
)


def test_native_tool_protocol_is_self_hashed_and_train_only() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert validate_protocol(value) == value["protocol_hash"]

    changed = json.loads(json.dumps(value))
    changed["access_policy"]["execution_permitted_splits"] = ["development"]
    with pytest.raises(NativeToolInterfaceError, match="train boundary"):
        validate_protocol(changed)


def test_parse_native_response_accepts_terminal_and_native_calls() -> None:
    content, calls = parse_native_response(
        "The task is complete.", turn=0, max_calls_per_turn=8
    )
    assert content == "The task is complete."
    assert calls == ()

    content, calls = parse_native_response(
        "I will check both records.\n"
        '<tool_call>{"name":"lookup","arguments":{"q":"a"}}</tool_call>\n'
        '<tool_call>{"name":"lookup","arguments":{"q":"b"}}</tool_call>',
        turn=3,
        max_calls_per_turn=8,
    )
    assert content == "I will check both records."
    assert [call.call_id for call in calls] == [
        "turn-3-call-0",
        "turn-3-call-1",
    ]
    assert [call.function for call in calls] == ["lookup", "lookup"]
    assert [dict(call.arguments) for call in calls] == [{"q": "a"}, {"q": "b"}]


@pytest.mark.parametrize(
    "response,error",
    [
        ("   ", "empty"),
        ("<tool_call>{}</tool_call>", "exactly"),
        (
            '<tool_call>{"name":"x","arguments":{}}</tool_call',
            "unmatched",
        ),
        ("```json\n{}\n```", None),
        (
            '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
            "strict JSON",
        ),
    ],
)
def test_parse_native_response_fails_closed(
    response: str, error: str | None
) -> None:
    if error is None:
        # A fenced object without native tags is terminal text, never silently
        # converted into a tool call.
        content, calls = parse_native_response(
            response, turn=0, max_calls_per_turn=8
        )
        assert content == response
        assert calls == ()
        return
    with pytest.raises(NativeToolInterfaceError, match=error):
        parse_native_response(response, turn=0, max_calls_per_turn=8)


class _NativeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def complete_tool_chat(self, messages, tools, *, seed: int, max_tokens: int):
        copied_messages = json.loads(canonical_json(messages))
        copied_tools = json.loads(canonical_json(tools))
        self.requests.append((copied_messages, copied_tools))
        text = self.responses.pop(0)
        return SimpleNamespace(
            text=text,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            metadata={
                "rendered_input": f"rendered-{len(self.requests)}",
                "input_mode": "native_tool_chat",
            },
        )

    def failure_metadata(self):
        return {"client": "fixture"}


def test_native_tool_loop_preserves_structured_turns_and_tool_results() -> None:
    client = _NativeClient(
        [
            '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>',
            "The value is 1.",
        ]
    )
    schema = CanonicalToolSchema(
        name="lookup",
        description="Look up a value.",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )

    result = run_native_tool_loop(
        model_client=client,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "request"},
        ],
        tool_schemas=[schema],
        execute_call=lambda call: {
            "function": call.function,
            "value": 1,
        },
        seed_for_turn=lambda turn: 100 + turn,
        max_turns=4,
        max_calls_per_turn=2,
        max_tokens=64,
    )

    assert result.terminated is True
    assert result.error is None
    assert result.output_text == "The value is 1."
    assert [call.function for call in result.traces] == ["lookup"]
    assert [record.seed for record in result.model_calls] == [100, 101]
    assert all(
        record.canonical_input["protocol"] == NATIVE_PROTOCOL
        for record in result.model_calls
    )
    assert [message["role"] for message in client.requests[1][0]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert json.loads(client.requests[1][0][-1]["content"])["value"] == 1


def test_native_tool_loop_rejects_unknown_function_without_execution() -> None:
    client = _NativeClient(
        ['<tool_call>{"name":"delete_all","arguments":{}}</tool_call>']
    )
    schema = CanonicalToolSchema(
        name="lookup",
        description="Look up a value.",
        parameters={"type": "object"},
    )
    executed: list[str] = []

    result = run_native_tool_loop(
        model_client=client,
        initial_messages=[{"role": "user", "content": "request"}],
        tool_schemas=[schema],
        execute_call=lambda call: executed.append(call.function),
        seed_for_turn=lambda turn: turn,
        max_turns=2,
        max_calls_per_turn=2,
        max_tokens=32,
    )

    assert result.terminated is False
    assert "unknown tools" in str(result.error)
    assert executed == []
    assert result.model_calls[0].parsed_output["tool_calls"][0]["function"] == "delete_all"


class _NativeRepairModel:
    def __init__(self, model: dict[str, Any]) -> None:
        self.model = model

    def complete_tool_chat(self, messages, tools, *, seed: int, max_tokens: int):
        del seed, max_tokens
        has_tool_result = any(message["role"] == "tool" for message in messages)
        text = (
            "done"
            if has_tool_result
            else '<tool_call>{"name":"candidate_0","arguments":{}}</tool_call>'
        )
        client_input = canonical_json({"messages": messages, "tools": tools})
        rendered = "frozen-native-rendering:" + client_input
        metadata = {
            "client": "local_transformers",
            "model_id": self.model["model_id"],
            "requested_model_revision": self.model["model_revision"],
            "model_revision": self.model["model_revision"],
            "requested_tokenizer_revision": self.model["tokenizer_revision"],
            "tokenizer_revision": self.model["tokenizer_revision"],
            "local_checkpoint_fingerprint": self.model["checkpoint_fingerprint"],
            "local_checkpoint_verification_mode": "full_tree_sha256_audit",
            "local_checkpoint_manifest_hash": "a" * 64,
            "local_checkpoint_path": "/frozen/qwen",
            "dtype": self.model["dtype"],
            "device": "cuda:0",
            "temperature": self.model["temperature"],
            "top_p": self.model["top_p"],
            "batch_size": 1,
            "local_files_only": True,
            "external_api_calls": 0,
            "gpu_name": "NVIDIA H200",
            "input_mode": "native_tool_chat",
            "input_messages": json.loads(canonical_json(messages)),
            "input_messages_hash": hashlib.sha256(
                canonical_json(messages).encode("utf-8")
            ).hexdigest(),
            "input_tools": json.loads(canonical_json(tools)),
            "input_tools_hash": hashlib.sha256(
                canonical_json(tools).encode("utf-8")
            ).hexdigest(),
            "native_tool_chat_input_hash": hashlib.sha256(
                client_input.encode("utf-8")
            ).hexdigest(),
            "input_prompt_hash": hashlib.sha256(
                client_input.encode("utf-8")
            ).hexdigest(),
            "rendered_input": rendered,
            "rendered_input_hash": hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest(),
            "chat_template_hash": QWEN_CHAT_TEMPLATE_SHA256,
        }
        return SimpleNamespace(text=text, usage=None, metadata=metadata)

    def failure_metadata(self):
        return {}


def test_native_trial_uses_exact_prompt_binding_and_atomic_replay() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    upstream_protocol_hash = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/scientific-v6-clean-repair-train-v1.json"
        ).read_text(encoding="utf-8")
    )["protocol_hash"]
    task = _live_record(upstream_protocol_hash)

    result = run_trial(
        protocol=protocol,
        task_record=task,
        model_client=_NativeRepairModel(protocol["model"]),
        compat=_RepairCompat,
        task_records_hash=stable_hash("native-fixture-corpus"),
    )

    assert result["arm"] == ARM
    assert result["model_protocol_valid"] is True
    assert result["model_provenance_valid"] is True
    assert result["prompt_binding_valid"] is True
    assert result["run_valid"] is True
    assert result["utility"] == 1
    assert result["locked_call_count"] == 1
    assert result["committed_call_count"] == 1
    assert result["exact_oracle_call_sequence"] is True
    assert result["unauthorized_function_count"] == 0
    assert result["repair_context_retired_before_final_replay"] is True
    assert len(result["model_calls"]) == 2
