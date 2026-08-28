from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import pytest

from silenttwin.agentdojo.canonical import CanonicalMessage, CanonicalToolSchema
from silenttwin.agentdojo.pipeline import StructuredControlledAttacker, run_tool_loop
from silenttwin.agentdojo.results import (
    ExactTranscriptEvidence,
    conservative_asr,
    make_grades,
    valid_run_asr,
)
from silenttwin.agentdojo.seeds import SeedSchedule
from silenttwin.agentdojo.visibility import public_value
from silenttwin.backends.base import BackendError, BackendErrorStage
from silenttwin.schemas import stable_digest
from tests.fakes.fake_backend import (
    FakeControlledModel,
    FakeResponse,
    make_fake_agentdojo_backend,
)


@dataclass(frozen=True)
class _Response:
    text: str
    metadata: dict


class _TwoTurnModel:
    def __init__(self) -> None:
        self.turn = 0

    def complete(self, prompt: str, *, seed: int, max_tokens: int):
        del prompt, max_tokens
        current_turn = self.turn
        if self.turn == 0:
            value = {
                "content": None,
                "tool_calls": [
                    {"call_id": "c0", "function": "lookup", "arguments": {"key": "x"}}
                ],
            }
        else:
            value = {"content": "done", "tool_calls": []}
        self.turn += 1
        return _Response(
            json.dumps(value),
            {
                "seed": seed,
                "rendered_input": f"tokenizer-rendered-turn-{current_turn}",
            },
        )


def test_native_tool_loop_is_multi_turn_canonical_and_bounded() -> None:
    result = run_tool_loop(
        model_client=_TwoTurnModel(),
        initial_messages=(CanonicalMessage("user", public_value("look up x")),),
        tool_schemas=(
            CanonicalToolSchema(
                "lookup", "lookup", {"type": "object", "properties": {"key": {"type": "string"}}}
            ),
        ),
        execute_call=lambda call: public_value({"value": call.arguments["key"]}),
        seed_for_turn=lambda turn: 100 + turn,
        max_turns=3,
    )
    assert result.terminated
    assert result.output_text == "done"
    assert [call.function for call in result.traces] == ["lookup"]
    assert [record.seed for record in result.model_calls] == [100, 101]
    assert [record.rendered_input_hash for record in result.model_calls] == [
        hashlib.sha256(b"tokenizer-rendered-turn-0").hexdigest(),
        hashlib.sha256(b"tokenizer-rendered-turn-1").hexdigest(),
    ]
    assert result.trace_hash


class _RenderedControlledModel(FakeControlledModel):
    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> FakeResponse:
        response = super().complete(prompt, seed=seed, max_tokens=max_tokens)
        return FakeResponse(
            response.text,
            {
                **dict(response.metadata),
                "rendered_input": "tokenizer-rendered-controlled",
            },
        )


class _FailingRenderedModel:
    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> None:
        del prompt, seed, max_tokens
        raise RuntimeError("fixture failure")

    @staticmethod
    def failure_metadata() -> dict[str, str]:
        return {"rendered_input": "tokenizer-rendered-failure"}


class _PredictionModel:
    def __init__(self, posterior: dict[str, object]) -> None:
        self.posterior = posterior

    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> _Response:
        del prompt, seed, max_tokens
        return _Response(
            json.dumps({"prediction": "theta0", "posterior": self.posterior}),
            {"rendered_input": "tokenizer-rendered-prediction"},
        )


def test_controlled_call_hash_binds_actual_rendered_input_on_success_and_failure() -> None:
    scenario = make_fake_agentdojo_backend().public_scenario
    success = StructuredControlledAttacker(
        _RenderedControlledModel(),
        immutable_model_revision="sha256:" + "1" * 64,
    )
    assert success.choose_probe(scenario, (), probe_index=0, seed=3).valid
    success_call = success.provenance["calls"][0]
    assert success_call["rendered_input_hash"] == hashlib.sha256(
        b"tokenizer-rendered-controlled"
    ).hexdigest()

    failure = StructuredControlledAttacker(
        _FailingRenderedModel(),
        immutable_model_revision="sha256:" + "1" * 64,
    )
    assert not failure.choose_probe(scenario, (), probe_index=0, seed=3).valid
    failure_call = failure.provenance["calls"][0]
    assert failure_call["rendered_input_hash"] == hashlib.sha256(
        b"tokenizer-rendered-failure"
    ).hexdigest()


class _MalformedRenderedModel:
    def complete(self, prompt: str, *, seed: int, max_tokens: int):
        del prompt, seed, max_tokens
        return SimpleNamespace(
            text="not-json",
            metadata={
                "rendered_input": "tokenizer-rendered-malformed",
                "transport": "fixture",
            },
            usage=SimpleNamespace(
                input_tokens=11, output_tokens=2, total_tokens=13
            ),
        )

    @staticmethod
    def failure_metadata() -> dict[str, str]:
        return {"provider_failure": "strict_parse"}


def test_strict_parse_failure_retains_raw_response_metadata_and_usage() -> None:
    scenario = make_fake_agentdojo_backend().public_scenario
    attacker = StructuredControlledAttacker(
        _MalformedRenderedModel(),
        immutable_model_revision="sha256:" + "1" * 64,
    )

    assert not attacker.choose_probe(scenario, (), probe_index=0, seed=3).valid
    call = attacker.provenance["calls"][0]
    assert call["raw_response"] == "not-json"
    assert call["raw_response_hash"] == hashlib.sha256(b"not-json").hexdigest()
    assert call["rendered_chat_template_input"] == "tokenizer-rendered-malformed"
    assert call["metadata"]["usage"]["total_tokens"] == 13
    assert call["failure_metadata"] == {"provider_failure": "strict_parse"}
    assert call["error"].startswith("JSONDecodeError:")


@pytest.mark.parametrize(
    "posterior",
    (
        {"theta0": True, "theta1": False},
        {"theta0": "0.6", "theta1": "0.4"},
        {"theta0": 0.5, "theta1": 0.5, "other": 0.0},
    ),
)
def test_controlled_prediction_requires_exact_json_number_posterior(
    posterior: dict[str, object],
) -> None:
    scenario = make_fake_agentdojo_backend().public_scenario
    attacker = StructuredControlledAttacker(
        _PredictionModel(posterior),
        immutable_model_revision="sha256:" + "1" * 64,
    )

    prediction = attacker.predict_hidden_state(scenario, (), seed=3)

    assert prediction.valid is False
    assert prediction.error is not None


def test_downstream_tool_failure_does_not_duplicate_successful_model_call() -> None:
    result = run_tool_loop(
        model_client=_TwoTurnModel(),
        initial_messages=(CanonicalMessage("user", public_value("look up x")),),
        tool_schemas=(
            CanonicalToolSchema(
                "lookup", "lookup", {"type": "object", "properties": {}}
            ),
        ),
        execute_call=lambda _call: (_ for _ in ()).throw(RuntimeError("tool failed")),
        seed_for_turn=lambda turn: 100 + turn,
        max_turns=3,
    )

    assert result.terminated is False
    assert result.error == "RuntimeError:tool failed"
    assert len(result.model_calls) == 1
    assert result.model_calls[0].error is None
    assert result.model_calls[0].raw_response is not None


def test_seed_schedule_is_q_condition_theta_independent_and_prefix_coupled() -> None:
    left = SeedSchedule("scenario", "sha256:revision", 3)
    right = SeedSchedule("scenario", "sha256:revision", 3)
    assert [left.probe(i) for i in range(4)] == [right.probe(i) for i in range(4)]
    assert left.prediction() == right.prediction()
    assert left.final() == right.final()
    assert "query_budget" in left.to_dict()["forbidden_factors"]
    assert "theta" in left.to_dict()["forbidden_factors"]


def test_error_accounting_is_conservative_and_exact_tv_is_not_applicable() -> None:
    error = BackendError(BackendErrorStage.ATTACK_GRADER, "failed", "fixture")
    invalid = make_grades(utility=True, raw_security_result=None, errors=(error,))
    valid = make_grades(utility=True, raw_security_result=False)
    records = [
        {"run_valid": invalid.run_valid, "conservative_attack_success": invalid.conservative_attack_success},
        {"run_valid": valid.run_valid, "attack_success": valid.attack_success, "conservative_attack_success": valid.conservative_attack_success},
    ]
    assert conservative_asr(records) == 0.5
    assert valid_run_asr(records) == 0.0
    evidence = ExactTranscriptEvidence.learned_not_applicable()
    assert evidence.status == "not_applicable"
    assert evidence.value is None
