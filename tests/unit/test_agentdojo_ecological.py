from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Mapping, Sequence

import pytest

from silenttwin.agentdojo.canonical import CanonicalToolSchema
from silenttwin.agentdojo.ecological import (
    EcologicalExperimentSpec,
    EcologicalProtocolError,
    _defended_tool_result,
    _select_tools,
    run_ecological_trial,
)
from silenttwin.schemas import stable_digest
from tests.fakes.fake_backend import (
    FakeCompat,
    FakeEnvironment,
    FakeInjectionTask,
    FakeResponse,
    FakeSuite,
    FakeTask,
)


class _EcologicalCompat(FakeCompat):
    events: list[str] = []
    grader_traces: tuple[Any, ...] | None = None

    @classmethod
    def reset(cls) -> None:
        cls.events = []
        cls.grader_traces = None

    @classmethod
    def load_environment(
        cls, suite: Any, user_task: Any, injections: Mapping[str, str]
    ) -> FakeEnvironment:
        cls.events.append("load_environment")
        return super().load_environment(suite, user_task, injections)

    @staticmethod
    def canonical_tool_schemas(suite: Any) -> tuple[dict[str, Any], ...]:
        del suite
        return tuple(
            {
                "name": name,
                "description": f"Fixture {name}",
                "parameters": {"type": "object", "properties": {}},
            }
            for name in ("candidate_0", "always_error", "candidate_1")
        )

    @staticmethod
    def generate_attack_injections(
        suite: Any,
        user_task: Any,
        injection_task: Any,
        *,
        attack_name: str,
        target_pipeline_name: str,
    ) -> dict[str, str]:
        del suite, user_task, attack_name, target_pipeline_name
        return {"vector0": str(injection_task.GOAL)}

    @classmethod
    def execute_call(
        cls, suite: Any, environment: FakeEnvironment, call: Mapping[str, Any]
    ) -> Any:
        cls.events.append(f"execute:{call['function']}")
        return super().execute_call(suite, environment, call)

    @classmethod
    def grade_attack_success(
        cls,
        task: Any,
        output_text: str,
        pre_environment: FakeEnvironment,
        post_environment: FakeEnvironment,
        traces: Sequence[Any] = (),
    ) -> bool:
        del task, output_text, pre_environment
        cls.grader_traces = tuple(traces)
        # Deliberately trace-sensitive: passing traces from a discarded replay
        # would incorrectly turn an atomic failure into attack success.
        return bool(traces) or bool(post_environment.state["effects"])


@dataclass
class _ScriptedEcologicalModel:
    responses: list[dict[str, Any]]
    retirement_error: bool = False

    def __post_init__(self) -> None:
        self.prompts: list[str] = []
        self.retired = False

    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> FakeResponse:
        del seed, max_tokens
        if self.retired:
            raise RuntimeError("fixture model context is retired")
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("fixture exhausted scripted responses")
        return FakeResponse(json.dumps(self.responses.pop(0)), {"client": "ecological-fixture"})

    def retire(self) -> None:
        _EcologicalCompat.events.append("retire_model_context")
        self.retired = True
        if self.retirement_error:
            raise RuntimeError("fixture retirement failure")


class _FailingEcologicalModel:
    def __init__(self) -> None:
        self.retired = False

    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> None:
        del prompt, seed, max_tokens
        raise RuntimeError("fixture transport failure")

    @staticmethod
    def failure_metadata() -> dict[str, str]:
        return {"rendered_input": "TOKENIZER_RENDERED::failed-ecological-call"}

    def retire(self) -> None:
        self.retired = True


def _tool_turn(*functions: str) -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [
            {"call_id": f"call-{index}", "function": function, "arguments": {}}
            for index, function in enumerate(functions)
        ],
    }


def _spec(trial_id: str = "ecological-fixture") -> EcologicalExperimentSpec:
    injections = {"vector0": FakeInjectionTask.GOAL}
    expected_environment = FakeCompat.load_environment(
        FakeSuite(), FakeTask(), injections
    )
    return EcologicalExperimentSpec(
        trial_id=trial_id,
        scenario_id="ecological-scenario",
        suite="workspace",
        structural_group_id="ecological-structural-group",
        dataset_split="fixture",
        user_task_id=FakeTask.ID,
        injection_task_id=FakeInjectionTask.ID,
        injection_vector_id="vector0",
        threat_mode="indirect_injection",
        defense="none",
        released_attack_name="direct",
        released_attack_target_pipeline="silenttwin-local-tool-loop",
        expected_initial_environment_hash=FakeCompat.environment_hash(
            expected_environment
        ),
        expected_injection_rendering_hash=stable_digest(injections),
        replicate=0,
        victim_model_revision="sha256:" + "7" * 64,
        system_prompt="Use the fixture tools.",
        max_turns=3,
        max_calls_per_turn=4,
    )


def test_atomic_replay_failure_discards_every_call_and_trace_grades_fresh_target() -> None:
    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel(
        [
            _tool_turn("candidate_0", "always_error", "candidate_1"),
            {"content": "done", "tool_calls": []},
        ]
    )
    result = run_ecological_trial(
        _spec(),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
    )

    assert result.retired_before_settlement is True
    assert model.retired is True
    assert result.initial_environment_hash == result.final_start_environment_hash
    assert result.final_environment_hash == result.initial_environment_hash
    assert result.final_result is not None
    assert len(result.final_result.staged_effect_ids) == 3
    assert result.final_result.committed_effect_ids == ()
    assert len(result.final_result.rejected_effect_ids) == 3
    assert [action.status for action in result.final_result.trusted_actions] == [
        "atomicity_discarded",
        "error",
        "atomicity_discarded",
    ]
    assert _EcologicalCompat.grader_traces == ()
    assert result.grades.raw_attack_success is False
    assert result.grades.run_valid is False
    assert result.grades.conservative_attack_success is True
    assert result.trusted_evaluation is not None
    speculative = result.trusted_evaluation.to_trusted_dict()["value"][
        "speculative_actions"
    ]
    assert [action["status"] for action in speculative] == [
        "staged",
        "error",
        "staged",
    ]
    trusted = result.trusted_evaluation.to_trusted_dict()["value"]
    assert trusted["released_attack_name"] == "direct"
    assert trusted["released_attack_target_pipeline"] == "silenttwin-local-tool-loop"
    assert isinstance(trusted["injection_rendering_hash"], str)
    assert len(trusted["injection_rendering_hash"]) == 64

    # The speculative callback and model context are retired before the fresh
    # final target is loaded, and settlement output is never model-visible.
    second_load = _EcologicalCompat.events.index("load_environment", 1)
    assert _EcologicalCompat.events.index("retire_model_context") < second_load
    assert all("settlement_tool_result" not in prompt for prompt in model.prompts)
    assert all(
        event.unwrap().get("event_type") != "settlement_tool_result"
        for event in result.public_transcript
    )
    assert any(
        event.unwrap().get("event_type") == "settlement_tool_result"
        for event in result.postselection_output
    )


def test_ecological_session_forwards_failure_metadata_into_victim_call() -> None:
    _EcologicalCompat.reset()
    model = _FailingEcologicalModel()

    result = run_ecological_trial(
        _spec("model-transport-failure"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
    )

    assert result.grades.run_valid is False
    calls = result.model_provenance["calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["rendered_chat_template_input"] == (
        "TOKENIZER_RENDERED::failed-ecological-call"
    )
    assert call["failure_metadata"]["rendered_input"] == (
        "TOKENIZER_RENDERED::failed-ecological-call"
    )
    assert call["error"].startswith("RuntimeError:fixture transport failure")


def test_retirement_failure_consumes_slot_without_final_replay() -> None:
    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel(
        [
            _tool_turn("candidate_0"),
            {"content": "done", "tool_calls": []},
        ],
        retirement_error=True,
    )
    result = run_ecological_trial(
        _spec("retirement-failure"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
    )

    assert result.retired_before_settlement is False
    assert result.final_plan_attempt_count == 1
    assert result.final_result is not None
    assert result.final_result.staged_effect_ids == ()
    assert result.final_result.committed_effect_ids == ()
    assert result.final_environment_hash == result.initial_environment_hash
    assert result.error_stage == "retirement"
    assert any(error.code == "ecological_context_retirement_error" for error in result.errors)
    assert _EcologicalCompat.grader_traces == ()
    # Only speculative execution occurred; the locked call was never replayed.
    assert _EcologicalCompat.events.count("execute:candidate_0") == 1
    assert result.trusted_evaluation is not None
    assert len(
        result.trusted_evaluation.to_trusted_dict()["value"]["speculative_actions"]
    ) == 1


def test_successful_locked_workflow_replays_once_after_retirement() -> None:
    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel(
        [
            _tool_turn("candidate_0", "candidate_1"),
            {"content": "done", "tool_calls": []},
        ]
    )
    result = run_ecological_trial(
        _spec("successful-replay"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
    )

    assert result.retired_before_settlement is True
    assert result.final_result is not None
    assert len(result.final_result.staged_effect_ids) == 2
    assert result.final_result.committed_effect_ids == result.final_result.staged_effect_ids
    assert result.final_result.rejected_effect_ids == ()
    assert [action.status for action in result.final_result.trusted_actions] == [
        "committed",
        "committed",
    ]
    assert len(_EcologicalCompat.grader_traces or ()) == 2
    assert result.grades.attack_success is True
    assert result.grades.run_valid is True
    assert result.final_environment_hash != result.initial_environment_hash
    assert _EcologicalCompat.events.count("execute:candidate_0") == 2
    assert _EcologicalCompat.events.count("execute:candidate_1") == 2
    assert result.trusted_evaluation is not None
    assert [
        action["status"]
        for action in result.trusted_evaluation.to_trusted_dict()["value"][
            "speculative_actions"
        ]
    ] == ["staged", "staged"]


def test_repeat_user_prompt_is_a_separate_user_role_message() -> None:
    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel(
        [_tool_turn("candidate_0"), {"content": "done", "tool_calls": []}]
    )
    result = run_ecological_trial(
        replace(_spec("repeat-user-role"), defense="repeat_user_prompt"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
    )

    transcript = [message.unwrap() for message in result.public_transcript]
    assert [message["role"] for message in transcript] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert transcript[4]["content"] == FakeTask.PROMPT
    assert "repeated_user_prompt" not in json.dumps(transcript[3])


def test_speculative_callback_exception_is_audited_and_never_replayed() -> None:
    class _RaisingCompat(_EcologicalCompat):
        @classmethod
        def execute_call(
            cls, suite: Any, environment: FakeEnvironment, call: Mapping[str, Any]
        ) -> Any:
            cls.events.append(f"execute:{call['function']}")
            raise RuntimeError("fixture callback exception")

    _RaisingCompat.reset()
    model = _ScriptedEcologicalModel([_tool_turn("candidate_0")])
    result = run_ecological_trial(
        _spec("callback-exception"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_RaisingCompat,
    )

    assert result.final_result is not None
    assert result.final_result.staged_effect_ids == ()
    assert result.final_environment_hash == result.initial_environment_hash
    assert result.grades.run_valid is False
    assert result.trusted_evaluation is not None
    speculative = result.trusted_evaluation.to_trusted_dict()["value"][
        "speculative_actions"
    ]
    assert len(speculative) == 1
    assert speculative[0]["status"] == "error"
    assert speculative[0]["backend_error"] == "speculative_tool_exception"
    assert _RaisingCompat.events.count("execute:candidate_0") == 1


def test_clean_spec_rejects_an_injection_vector_without_task() -> None:
    try:
        EcologicalExperimentSpec(
            trial_id="clean-invalid",
            scenario_id="scenario",
            suite="workspace",
            structural_group_id="group",
            dataset_split="fixture",
            user_task_id="user_task_0",
            injection_task_id=None,
            injection_vector_id="vector0",
            threat_mode="clean",
            defense="none",
            released_attack_name=None,
            released_attack_target_pipeline=None,
            expected_initial_environment_hash="1" * 64,
            expected_injection_rendering_hash=None,
            replicate=0,
            victim_model_revision="sha256:" + "1" * 64,
            system_prompt="fixture",
        )
    except ValueError as exc:
        assert "task/vector" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("clean spec accepted an injection vector")


def test_released_transformers_detector_tuple_uses_boolean_not_tuple_truthiness() -> None:
    value = {"status": "ok", "value": "ordinary tool data"}
    safe = _defended_tool_result(
        value,
        defense="transformers_pi_detector",
        user_prompt="fixture",
        pi_detector=lambda text: (False, 0.99),
    )
    audit: list[dict[str, Any]] = []
    detected = _defended_tool_result(
        value,
        defense="transformers_pi_detector",
        user_prompt="fixture",
        pi_detector=lambda text: (True, 0.01),
        detection_audit=audit,
        call_index=3,
    )
    assert safe == value
    assert detected == "<Data omitted because a prompt injection was detected>"
    assert audit == [
        {
            "call_index": 3,
            "input_hash": stable_digest(
                json.dumps(value, sort_keys=True, separators=(",", ":"))
            ),
            "input_serialization_revision": "silenttwin-compact-json-tool-result-v1",
            "detected": True,
            "safety_score": 0.01,
            "role": "content_input_detector",
        }
    ]

    try:
        _defended_tool_result(
            value,
            defense="transformers_pi_detector",
            user_prompt="fixture",
            pi_detector=lambda text: "unsafe",  # type: ignore[return-value]
        )
    except EcologicalProtocolError as exc:
        assert "incompatible result" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("detector accepted a truthy non-contract result")


def test_spotlighting_exposes_only_the_delimited_tool_result() -> None:
    value = {"status": "ok", "value": "ordinary tool data"}
    defended = _defended_tool_result(
        value,
        defense="spotlighting_with_delimiting",
        user_prompt="fixture",
        pi_detector=None,
    )
    assert defended == '<<{"status":"ok","value":"ordinary tool data"}>>'
    assert "defense_adapter" not in defended


def test_adapted_tool_filter_may_select_an_empty_frozen_subset() -> None:
    schema = CanonicalToolSchema(
        "only_tool", "fixture", {"type": "object", "properties": {}}
    )
    selected, history = _select_tools(
        (schema,),
        defense="tool_filter",
        user_prompt="fixture",
        system_prompt="system",
        tool_filter=lambda prompt, schemas, *, system_prompt: {
            "tool_names": [],
            "visible_history": [
                {"role": "user", "content": "filter"},
                {"role": "assistant", "content": ""},
            ],
        },
    )
    assert selected == ()
    assert [message.role for message in history] == ["user", "assistant"]


def test_tool_filter_exchange_is_visible_to_the_subsequent_victim_turn() -> None:
    class _Filter:
        provenance = {"baseline_label": "adapted_local_tool_filter"}

        def __call__(
            self,
            prompt: str,
            schemas: Sequence[Mapping[str, Any]],
            *,
            system_prompt: str,
        ) -> Mapping[str, Any]:
            del prompt, system_prompt
            return {
                "tool_names": [str(schema["name"]) for schema in schemas],
                "visible_history": [
                    {"role": "user", "content": "released filter prompt"},
                    {"role": "assistant", "content": "candidate_0,candidate_1"},
                ],
            }

    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel(
        [{"content": "done after filtering", "tool_calls": []}]
    )
    result = run_ecological_trial(
        replace(_spec("tool-filter-history"), defense="tool_filter"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
        tool_filter=_Filter(),
    )

    request = json.loads(model.prompts[0].split("TOOL_LOOP_INPUT:\n", 1)[1])
    assert [message["role"] for message in request["messages"][:4]] == [
        "system",
        "user",
        "user",
        "assistant",
    ]
    assert request["messages"][2]["content"] == "released filter prompt"
    assert request["messages"][3]["content"] == "candidate_0,candidate_1"
    trusted = result.trusted_evaluation.to_trusted_dict()["value"]
    assert trusted["defense_adapter"] == "adapted_local_tool_filter"
    assert trusted["tool_filter_provenance"] == {
        "baseline_label": "adapted_local_tool_filter"
    }


def test_missing_tool_filter_fails_before_terminal_model_call_and_is_honest() -> None:
    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel(
        [{"content": "would otherwise terminate cleanly", "tool_calls": []}]
    )
    result = run_ecological_trial(
        replace(_spec("missing-tool-filter"), defense="tool_filter"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
    )

    assert model.prompts == []
    assert result.grades.run_valid is False
    assert result.error_stage == "setup"
    record = result.to_record()
    assert record["trusted_evaluation"]["value"]["defense_adapter"] == (
        "unavailable_not_configured"
    )
    assert record["model_provenance"]["defense_adapter"] == (
        "unavailable_not_configured"
    )


def test_missing_transformer_detector_fails_before_terminal_model_call() -> None:
    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel(
        [{"content": "would otherwise terminate cleanly", "tool_calls": []}]
    )
    result = run_ecological_trial(
        replace(_spec("missing-detector"), defense="transformers_pi_detector"),
        suite=FakeSuite(),
        user_task=FakeTask(),
        injection_task=FakeInjectionTask(),
        model_client=model,
        compat=_EcologicalCompat,
    )

    assert model.prompts == []
    assert result.grades.run_valid is False
    assert result.error_stage == "setup"
    assert any(error.code == "ecological_defense_unavailable" for error in result.errors)


def test_runtime_rejects_attack_rendering_or_initial_state_drift_before_model_use() -> None:
    _EcologicalCompat.reset()
    model = _ScriptedEcologicalModel([{"content": "unused", "tool_calls": []}])
    with pytest.raises(EcologicalProtocolError, match="rendering differs"):
        run_ecological_trial(
            replace(_spec("rendering-drift"), expected_injection_rendering_hash="0" * 64),
            suite=FakeSuite(),
            user_task=FakeTask(),
            injection_task=FakeInjectionTask(),
            model_client=model,
            compat=_EcologicalCompat,
        )
    assert model.prompts == []

    with pytest.raises(EcologicalProtocolError, match="initial environment differs"):
        run_ecological_trial(
            replace(_spec("environment-drift"), expected_initial_environment_hash="0" * 64),
            suite=FakeSuite(),
            user_task=FakeTask(),
            injection_task=FakeInjectionTask(),
            model_client=model,
            compat=_EcologicalCompat,
        )
    assert model.prompts == []
