from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from silenttwin.agentdojo.config import ModelIdentity
from silenttwin.agentdojo.ecological_defenses import (
    AdaptedLocalToolFilter,
    EcologicalDefenseUnavailableError,
    RELEASED_TOOL_FILTER_PROMPT,
    RELEASED_TOOL_FILTER_PROMPT_SHA256,
    ReleasedTransformersPIDetectorAdapter,
)
from silenttwin.schemas import stable_digest


class _Detector:
    def __init__(self, value: object) -> None:
        self.value = value

    def detect(self, text: str) -> object:
        assert text == "fixture tool output"
        return self.value


class _FilterResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.metadata = {
            "rendered_input": "tokenizer-rendered-filter-input",
            "external_api_calls": 0,
        }
        self.usage = SimpleNamespace(
            input_tokens=9, output_tokens=2, total_tokens=11
        )


class _FilterClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> _FilterResponse:
        assert seed == 17
        assert max_tokens == 256
        self.prompts.append(prompt)
        return _FilterResponse(self.text)


def _identity() -> ModelIdentity:
    fingerprint = "sha256:" + "a" * 64
    return ModelIdentity(
        role="monitor",
        implementation="transformers_pi_detector",
        model_id="/operator/frozen/pi-detector",
        model_revision="b" * 40,
        tokenizer_revision="c" * 40,
        checkpoint_fingerprint=fingerprint,
        prompt_hash="b" * 64,
        runtime_fingerprint="sha256:" + "c" * 64,
        threshold=0.5,
    )


def _victim_identity() -> ModelIdentity:
    return ModelIdentity(
        role="victim",
        implementation="deterministic_fake",
        model_id="builtin:deterministic-fake-v1",
        model_revision="builtin:deterministic-fake-v1",
        tokenizer_revision="builtin:deterministic-fake-tokenizer-v1",
        checkpoint_fingerprint="builtin:deterministic-fake-checkpoint-v1",
        prompt_hash="d" * 64,
    )


def test_adapted_tool_filter_preserves_pinned_raw_substring_and_history() -> None:
    client = _FilterClient("search_files_by_filename")
    adapter = AdaptedLocalToolFilter(
        client,
        identity=_victim_identity(),
        seed=17,
    )
    schemas = [
        {"name": "search_files", "description": "broad", "parameters": {}},
        {
            "name": "search_files_by_filename",
            "description": "narrow",
            "parameters": {},
        },
        {"name": "send_email", "description": "mail", "parameters": {}},
    ]

    outcome = adapter("find a file", schemas, system_prompt="system")

    # This collision is intentional: the released implementation uses raw
    # substring matching rather than tokenized comma parsing.
    assert outcome["tool_names"] == [
        "search_files",
        "search_files_by_filename",
    ]
    assert outcome["visible_history"] == [
        {"role": "user", "content": RELEASED_TOOL_FILTER_PROMPT},
        {"role": "assistant", "content": "search_files_by_filename"},
    ]
    assert "TOOL_FILTER_INPUT:\n" in client.prompts[0]
    provenance = adapter.provenance
    assert provenance["baseline_label"] == "adapted_local_tool_filter"
    assert provenance["exact_released_transport"] is False
    assert provenance["released_prompt_sha256"] == (
        RELEASED_TOOL_FILTER_PROMPT_SHA256
    )
    call = provenance["call"]
    assert call["canonical_input"]["messages"][2]["content"] == (
        RELEASED_TOOL_FILTER_PROMPT
    )
    assert call["protocol_prompt"] == client.prompts[0]
    assert call["rendered_chat_template_input"] == (
        "tokenizer-rendered-filter-input"
    )
    assert call["rendered_input_hash"] == hashlib.sha256(
        b"tokenizer-rendered-filter-input"
    ).hexdigest()
    assert call["raw_response"] == "search_files_by_filename"
    assert call["usage"]["total_tokens"] == 11
    assert call["latency_ms"] >= 0.0
    assert call["failure"] is None


class _FailingFilterClient(_FilterClient):
    def complete(self, prompt: str, *, seed: int, max_tokens: int):
        del seed, max_tokens
        self.prompts.append(prompt)
        raise RuntimeError("fixture filter failure")

    @staticmethod
    def failure_metadata() -> dict[str, str]:
        return {"rendered_input": "tokenizer-rendered-filter-failure"}


def test_adapted_tool_filter_retains_full_failure_call_provenance() -> None:
    adapter = AdaptedLocalToolFilter(
        _FailingFilterClient("unused"),
        identity=_victim_identity(),
        seed=17,
    )

    with pytest.raises(EcologicalDefenseUnavailableError, match="fixture filter failure"):
        adapter(
            "find a file",
            [{"name": "search_files", "description": "broad", "parameters": {}}],
            system_prompt="system",
        )

    call = adapter.provenance["call"]
    assert call["canonical_input"]["tool_choice"] == "none"
    assert call["rendered_chat_template_input"] == (
        "tokenizer-rendered-filter-failure"
    )
    assert call["raw_response"] is None
    assert call["failure"]["type"] == "RuntimeError"
    assert call["failure_metadata"]["rendered_input"] == (
        "tokenizer-rendered-filter-failure"
    )


class _NonTextFilterClient(_FilterClient):
    def complete(self, prompt: str, *, seed: int, max_tokens: int):
        del seed, max_tokens
        self.prompts.append(prompt)
        return SimpleNamespace(
            text={"unexpected": "structured payload"},
            metadata={"rendered_input": "tokenizer-rendered-filter-nontext"},
        )


def test_adapted_tool_filter_serializes_non_text_failure_payload() -> None:
    adapter = AdaptedLocalToolFilter(
        _NonTextFilterClient("unused"),
        identity=_victim_identity(),
        seed=17,
    )

    with pytest.raises(EcologicalDefenseUnavailableError, match="has no text"):
        adapter(
            "find a file",
            [{"name": "search_files", "description": "broad", "parameters": {}}],
            system_prompt="system",
        )

    call = adapter.provenance["call"]
    assert call["raw_response"] == '{"unexpected":"structured payload"}'
    assert call["raw_response_serialization"] == {
        "status": "canonical_json_non_text",
        "returned_type": "dict",
    }
    assert call["raw_response_hash"] == hashlib.sha256(
        call["raw_response"].encode("utf-8")
    ).hexdigest()
    assert call["response_metadata"]["rendered_input"] == (
        "tokenizer-rendered-filter-nontext"
    )


def test_released_detector_adapter_enforces_threshold_and_records_provenance() -> None:
    identity = _identity()
    adapter = ReleasedTransformersPIDetectorAdapter(
        _Detector((True, 0.2)),
        identity=identity,
        resolved_checkpoint_fingerprint=identity.checkpoint_fingerprint,
    )

    assert adapter("fixture tool output") == (True, 0.2)
    assert adapter.provenance["role"] == "content_input_detector"
    assert adapter.provenance["input_semantics"] == (
        "adapted_from_agentdojo_tool_formatter"
    )
    assert adapter.provenance["model_revision"] == "b" * 40
    assert adapter.provenance["tokenizer_revision"] == "c" * 40
    assert adapter.provenance["checkpoint_fingerprint"] == "sha256:" + "a" * 64
    assert adapter.provenance["external_api_calls"] == 0
    call = adapter.provenance["calls"][0]
    assert call["detector_input_text"] == "fixture tool output"
    assert call["rendered_chat_template_input"]["status"] == "not_available"
    assert call["tokenized_input"]["status"] == "not_available"
    assert call["raw_output_hash"] == hashlib.sha256(
        b"[true,0.2]"
    ).hexdigest()
    assert call["usage"]["status"] == "not_available"
    assert call["latency_ms"] >= 0.0
    assert call["failure"] is None


@pytest.mark.parametrize(
    "value",
    (
        (False, 0.2),  # Boolean disagrees with score < 0.5.
        (True, 1.2),
        True,
        {"detected": True, "score": 0.2},
    ),
)
def test_released_detector_adapter_fails_closed_on_contract_drift(value: object) -> None:
    identity = _identity()
    adapter = ReleasedTransformersPIDetectorAdapter(
        _Detector(value),
        identity=identity,
        resolved_checkpoint_fingerprint=identity.checkpoint_fingerprint,
    )

    with pytest.raises(EcologicalDefenseUnavailableError):
        adapter("fixture tool output")
    call = adapter.provenance["calls"][0]
    assert call["failure"] is not None
    assert call["raw_output_serialization"]["status"] in {
        "canonical_json",
        "repr_fallback",
    }


def test_released_detector_factory_rejects_hub_identifier_before_importing_models() -> None:
    identity = _identity()
    identity = ModelIdentity(
        **{
            **identity.scientific_dict(),
            "model_id": "protectai/deberta-v3-base-prompt-injection-v2",
        }
    )

    with pytest.raises(
        EcologicalDefenseUnavailableError,
        match="explicit operational checkpoint path",
    ):
        ReleasedTransformersPIDetectorAdapter.from_identity(
            identity,
            compat=object(),
        )
