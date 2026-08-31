from __future__ import annotations

from typing import Any

import pytest

from silenttwin.model_clients.local_transformers import (
    LocalModelConfig,
    LocalModelUnavailableError,
    LocalTransformersModelClient,
)
from silenttwin.schemas import canonical_json


class _ChatTokenizer:
    chat_template = "fixture-chat-template"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def apply_chat_template(self, messages, **kwargs):
        copied = [dict(message) for message in messages]
        self.calls.append((copied, dict(kwargs)))
        return "TOKENIZER_RENDERED_CHAT"


def _client() -> LocalTransformersModelClient:
    return LocalTransformersModelClient(
        LocalModelConfig(
            model_id="fixture/model",
            model_revision="a" * 40,
            tokenizer_revision="a" * 40,
            device="cpu",
        )
    )


def test_complete_chat_preserves_messages_and_uses_tokenizer_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    tokenizer = _ChatTokenizer()
    client._tokenizer = tokenizer
    monkeypatch.setattr(client, "_load", lambda: None)
    captured: dict[str, Any] = {}

    def fake_complete_rendered(rendered: str, **kwargs):
        captured["rendered"] = rendered
        captured.update(kwargs)
        return "fixture-response"

    monkeypatch.setattr(client, "_complete_rendered", fake_complete_rendered)
    messages = [
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "candidate action"},
        {"role": "user", "content": "guardian criterion"},
    ]

    response = client.complete_chat(messages, seed=7, max_tokens=64)

    assert response == "fixture-response"
    assert tokenizer.calls == [
        (
            messages,
            {"tokenize": False, "add_generation_prompt": True},
        )
    ]
    assert captured["rendered"] == "TOKENIZER_RENDERED_CHAT"
    assert captured["input_prompt"] == canonical_json(messages)
    assert captured["seed"] == 7
    assert captured["max_tokens"] == 64
    assert captured["input_metadata"]["input_mode"] == "structured_chat"
    assert captured["input_metadata"]["input_messages"] == messages


def test_complete_chat_rejects_missing_template_and_ambiguous_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    tokenizer = _ChatTokenizer()
    tokenizer.chat_template = None
    client._tokenizer = tokenizer
    monkeypatch.setattr(client, "_load", lambda: None)

    with pytest.raises(LocalModelUnavailableError, match="requires a tokenizer"):
        client.complete_chat([{"role": "user", "content": "request"}])

    with pytest.raises(TypeError, match="exactly role and content"):
        client.complete_chat(
            [{"role": "user", "content": "request", "hidden": "forbidden"}]
        )


def test_score_next_tokens_preserves_exact_candidate_ids_and_chat_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    tokenizer = _ChatTokenizer()
    client._tokenizer = tokenizer
    monkeypatch.setattr(client, "_load", lambda: None)
    captured: dict[str, Any] = {}

    def fake_score(rendered: str, **kwargs):
        captured["rendered"] = rendered
        captured.update(kwargs)
        return "fixture-score"

    monkeypatch.setattr(client, "_score_rendered_next_tokens", fake_score)

    score = client.score_next_tokens(
        "forced choice request",
        candidate_token_ids={"A": 32, "B": 33},
    )

    assert score == "fixture-score"
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": "forced choice request"}],
            {"tokenize": False, "add_generation_prompt": True},
        )
    ]
    assert captured == {
        "rendered": "TOKENIZER_RENDERED_CHAT",
        "input_prompt": "forced choice request",
        "candidate_token_ids": {"A": 32, "B": 33},
    }
