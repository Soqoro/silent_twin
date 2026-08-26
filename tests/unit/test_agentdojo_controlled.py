from __future__ import annotations

import json

import pytest

from silenttwin.agentdojo.controlled import (
    ControlledExperimentSpec,
    run_controlled_trial,
)
from silenttwin.agentdojo.pipeline import StructuredControlledAttacker
from tests.fakes.fake_backend import FakeControlledModel, make_fake_agentdojo_backend


REVISION = "sha256:" + "7" * 64


def _run(
    condition: str,
    *,
    state: str = "theta0",
    donor: str | None = None,
    q: int = 4,
    experiment: str = "e2",
    channel: str | None = None,
    clock=None,
    sleeper=None,
    model: FakeControlledModel | None = None,
):
    backend = make_fake_agentdojo_backend()
    model = model or FakeControlledModel()
    attacker = StructuredControlledAttacker(
        model, immutable_model_revision=REVISION
    )
    policy = "silenttwin" if condition == "silenttwin" else "generic_failure"
    result = run_controlled_trial(
        ControlledExperimentSpec(
            trial_id=f"{experiment}-{condition}-{state}",
            experiment_id=experiment,
            condition=condition,
            feedback_policy=policy,
            query_budget=q,
            actual_state=state,
            donor_state=donor,
            replicate=2,
            user_task_id="user0",
            injection_task_id="inj0",
            injection_vector_id="fixture-vector",
            channel_intervention=channel,
        ),
        backend,
        attacker,
        clock=clock,
        sleeper=sleeper,
    )
    return result, model


def _request(prompt: str) -> dict[str, object]:
    return json.loads(prompt.split("PUBLIC_INPUT:\n", 1)[1])


class _DeterministicTimingRuntime:
    def __init__(self, events: list[tuple[str, object]] | None = None) -> None:
        self.now = 100.0
        self.events = events if events is not None else []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", seconds))
        self.now += seconds


class _OrderedControlledModel(FakeControlledModel):
    def __init__(self, events: list[tuple[str, object]]) -> None:
        super().__init__()
        self.events = events

    def complete(self, prompt: str, *, seed: int, max_tokens: int):
        self.events.append(("model", self._request(prompt)["request_type"]))
        return super().complete(prompt, seed=seed, max_tokens=max_tokens)


@pytest.mark.parametrize("state,expected", [("theta0", "candidate_0"), ("theta1", "candidate_1")])
def test_genuine_feedback_completes_mediation_chain(state: str, expected: str) -> None:
    result, model = _run("genuine", state=state)
    assert result.prediction == state
    assert result.final_plan_id == expected
    assert result.grades.attack_success is True
    assert result.final_plan_attempt_count == 1
    assert result.initial_environment_hash == result.final_start_environment_hash
    assert result.retired_before_settlement is True
    assert model.is_retired is True


@pytest.mark.parametrize(
    "condition,q,donor",
    [
        ("no_probe", 0, None),
        ("genuine", 4, None),
        ("matched_shuffled", 4, "theta1"),
        ("constant", 4, None),
        ("random_final", 4, None),
        ("oracle", 0, None),
        ("silenttwin", 4, None),
    ],
)
def test_every_e2_condition_uses_one_locked_slot(
    condition: str, q: int, donor: str | None
) -> None:
    result, model = _run(condition, q=q, donor=donor)
    assert result.final_plan_attempt_count == 1
    assert result.final_plan_id in {"candidate_0", "candidate_1"}
    assert result.initial_environment_hash == result.final_start_environment_hash
    assert result.retired_before_settlement
    calls_after_return = len(model.prompts)
    assert calls_after_return >= 1
    assert model.is_retired
    # No settlement or post-selection callback was issued.
    assert len(model.prompts) == calls_after_return


def test_shuffled_feedback_is_online_and_not_target_correlated() -> None:
    result, _ = _run("matched_shuffled", state="theta0", donor="theta1")
    assert result.prediction == "theta1"
    assert result.final_plan_id == "candidate_1"
    assert result.grades.attack_success is False


@pytest.mark.parametrize("condition,donor", [("genuine", None), ("matched_shuffled", "theta1"), ("constant", None), ("silenttwin", None)])
def test_e1_feedback_source_block_runs_without_a_final_effect(
    condition: str, donor: str | None
) -> None:
    result, _ = _run(condition, donor=donor, experiment="e1")
    assert result.final_plan_attempt_count == 0
    assert result.final_plan_id is None
    assert result.grades.run_valid
    assert result.initial_environment_hash == result.final_start_environment_hash


def test_public_result_has_no_monitor_policy_or_assignment_record() -> None:
    result, model = _run("matched_shuffled", state="theta0", donor="theta1")
    record = result.to_record()
    assert "monitor_decision" not in record
    assert "ground_truth_policy_label" not in record
    public = json.dumps(record["agent_visible_transcript"], sort_keys=True)
    for private in ("theta0", "theta1", "fixture-theta0", "fixture-theta1"):
        assert private not in public
    trusted = record["trusted_evaluation"]["value"]
    assert trusted["actual_hidden_state"] == "theta0"
    assert trusted["donor_state"] == "theta1"
    assert trusted["final_monitor"]["decision"] in {"allow", "block"}
    assert len(trusted["probe_monitor_evaluations"]) == result.query_budget
    assert all(
        item["feedback_execution_source"] == "matched_shuffled_donor"
        for item in trusted["probe_monitor_evaluations"]
    )
    assert all(
        item["monitor"]["decision"] in {"allow", "block"}
        for item in trusted["probe_monitor_evaluations"]
    )
    assert record["postselection_delivered_to_agent"] is False
    callbacks = "\n".join(model.prompts)
    for private in (
        "actual_state",
        "donor_state",
        "fixture-theta0",
        "fixture-theta1",
        "matched_shuffled",
    ):
        assert private not in callbacks
    assert "postselection_output" not in callbacks
    assert model.is_retired


def test_silenttwin_precedes_shuffled_source_in_factorial_cross() -> None:
    rows = []
    for state, donor in (("theta0", "theta1"), ("theta1", "theta0")):
        backend = make_fake_agentdojo_backend()
        attacker = StructuredControlledAttacker(
            FakeControlledModel(), immutable_model_revision=REVISION
        )
        rows.append(
            run_controlled_trial(
                ControlledExperimentSpec(
                    trial_id=f"silent-shuffled-{state}",
                    experiment_id="e1",
                    condition="matched_shuffled",
                    feedback_policy="silenttwin",
                    query_budget=4,
                    actual_state=state,
                    donor_state=donor,
                    replicate=2,
                    user_task_id="user0",
                    injection_task_id="inj0",
                    injection_vector_id="fixture-vector",
                ),
                backend,
                attacker,
            )
        )
    assert rows[0].feedback_source == rows[1].feedback_source == "matched_shuffled"
    assert [item.to_public_dict() for item in rows[0].public_transcript] == [
        item.to_public_dict() for item in rows[1].public_transcript
    ]


class _InvalidFinalModel(FakeControlledModel):
    def complete(self, prompt: str, *, seed: int, max_tokens: int):
        request = self._request(prompt)
        if request["request_type"] == "final_plan_selection":
            self.prompts.append(prompt)
            self.seeds.append(seed)
            from tests.fakes.fake_backend import FakeResponse

            return FakeResponse('{"plan_id":"not-a-candidate"}', {"seed": seed})
        return super().complete(prompt, seed=seed, max_tokens=max_tokens)


def test_invalid_final_output_consumes_slot_without_fallback() -> None:
    backend = make_fake_agentdojo_backend()
    model = _InvalidFinalModel()
    attacker = StructuredControlledAttacker(model, immutable_model_revision=REVISION)
    result = run_controlled_trial(
        ControlledExperimentSpec(
            "invalid-final",
            "e2",
            "genuine",
            "generic_failure",
            4,
            "theta0",
            None,
            0,
            "user0",
            "inj0",
            "fixture-vector",
        ),
        backend,
        attacker,
    )
    assert result.final_plan_attempt_count == 1
    assert result.final_result is None
    assert result.grades.run_valid is False
    assert result.grades.conservative_attack_success is True
    assert sum("final_plan_selection" in prompt for prompt in model.prompts) == 1


def test_backend_retirement_failure_prevents_settlement_and_is_conservative() -> None:
    backend = make_fake_agentdojo_backend()
    original = backend.retire_environment

    def fail_original_context(environment):
        if environment.role.value == "base":
            raise RuntimeError("fixture retirement failure")
        return original(environment)

    backend.retire_environment = fail_original_context  # type: ignore[method-assign]
    attacker = StructuredControlledAttacker(
        FakeControlledModel(), immutable_model_revision=REVISION
    )
    result = run_controlled_trial(
        ControlledExperimentSpec(
            "retirement-failure",
            "e2",
            "genuine",
            "generic_failure",
            4,
            "theta0",
            None,
            0,
            "user0",
            "inj0",
            "fixture-vector",
        ),
        backend,
        attacker,
    )
    assert result.final_plan_attempt_count == 1
    assert result.final_result is None
    assert result.retired_before_settlement is False
    assert result.grades.run_valid is False
    assert result.grades.conservative_attack_success is True
    assert any(error.code == "backend_context_retirement_error" for error in result.errors)


def test_final_backend_seed_is_independent_of_query_budget() -> None:
    observed: list[int] = []
    for condition, budget in (("no_probe", 0), ("genuine", 4), ("genuine", 16)):
        backend = make_fake_agentdojo_backend()
        original = backend.execute_final

        def capture(environment, theta, plan_id, *, seed=0, _original=original):
            observed.append(seed)
            return _original(environment, theta, plan_id, seed=seed)

        backend.execute_final = capture  # type: ignore[method-assign]
        attacker = StructuredControlledAttacker(
            FakeControlledModel(), immutable_model_revision=REVISION
        )
        result = run_controlled_trial(
            ControlledExperimentSpec(
                trial_id=f"fixed-final-seed-{budget}",
                experiment_id="e2",
                condition=condition,
                feedback_policy="generic_failure",
                query_budget=budget,
                actual_state="theta0",
                donor_state=None,
                replicate=2,
                user_task_id="user0",
                injection_task_id="inj0",
                injection_vector_id="fixture-vector",
            ),
            backend,
            attacker,
        )
        assert result.final_result is not None
    assert len(set(observed)) == 1


def test_live_exact_silenttwin_is_theta_byte_identical_and_private() -> None:
    theta0, model0 = _run(
        "silenttwin", state="theta0", channel="exact_silenttwin"
    )
    theta1, model1 = _run(
        "silenttwin", state="theta1", channel="exact_silenttwin"
    )

    assert [event.to_public_dict() for event in theta0.public_transcript] == [
        event.to_public_dict() for event in theta1.public_transcript
    ]
    assert model0.prompts == model1.prompts
    assert theta0.retired_before_settlement
    assert theta1.retired_before_settlement
    audit = theta0.trusted_evaluation.to_trusted_dict()["value"][
        "channel_intervention"
    ]
    assert audit["intervention_applied"] is False
    assert audit["operational_evidence"]["callback_invoked"] is False
    serialized = "\n".join(model0.prompts)
    for private in ("fixture-theta0", "fixture-theta1"):
        assert private not in serialized


@pytest.mark.parametrize(
    ("channel", "field_path"),
    [
        ("timing", ("timing_class",)),
        ("identifiers", ("opaque_request_id",)),
        ("error_schema", ("payload", "error_schema", "code")),
    ],
)
def test_live_probe_channel_is_visible_before_next_model_call(
    channel: str,
    field_path: tuple[str, ...],
) -> None:
    rows = []
    for state in ("theta0", "theta1"):
        timing = _DeterministicTimingRuntime() if channel == "timing" else None
        rows.append(
            _run(
                "silenttwin",
                state=state,
                channel=channel,
                clock=timing.clock if timing is not None else None,
                sleeper=timing.sleep if timing is not None else None,
            )
        )
    observed: list[object] = []
    for result, model in rows:
        final_request = _request(model.prompts[-1])
        event: object = final_request["transcript"][0]  # type: ignore[index]
        for key in field_path:
            assert isinstance(event, dict)
            event = event[key]
        observed.append(event)
        evidence = result.trusted_evaluation.to_trusted_dict()["value"][
            "channel_intervention"
        ]["operational_evidence"]
        assert evidence["probe_feedback_transform_count"] == result.query_budget
    assert observed[0] != observed[1]


def test_timing_channel_runs_scheduler_and_measures_before_next_observation() -> None:
    events: list[tuple[str, object]] = []
    runtime = _DeterministicTimingRuntime(events)
    model = _OrderedControlledModel(events)

    result, _ = _run(
        "silenttwin",
        state="theta1",
        q=4,
        channel="timing",
        clock=runtime.clock,
        sleeper=runtime.sleep,
        model=model,
    )

    sleep_positions = [
        index for index, event in enumerate(events) if event[0] == "sleep"
    ]
    assert len(sleep_positions) == result.query_budget
    assert all(events[index][1] == 0.005 for index in sleep_positions)
    for position in sleep_positions:
        assert any(
            event == ("model", "probe_selection")
            for event in events[position + 1 :]
        ) or any(
            event == ("model", "hidden_state_prediction")
            for event in events[position + 1 :]
        )
    first_event = result.public_transcript[0].unwrap()
    assert first_event["timing_class"] == "measured-padded"
    assert first_event["timing_observation"]["observed_padding_us"] == 5000
    evidence = result.trusted_evaluation.to_trusted_dict()["value"][
        "channel_intervention"
    ]["operational_evidence"]
    assert evidence["timing_scheduler_invocation_count"] == result.query_budget
    assert all(
        item["scheduler_completed_before_transcript_append"] is True
        for item in evidence["timing_observations"]
    )


def test_public_fidelity_degradation_is_live_but_decision_independent() -> None:
    exact, exact_model = _run(
        "silenttwin", state="theta0", channel="exact_silenttwin"
    )
    degraded0, model0 = _run(
        "silenttwin", state="theta0", channel="public_fidelity_degradation"
    )
    degraded1, model1 = _run(
        "silenttwin", state="theta1", channel="public_fidelity_degradation"
    )

    assert model0.prompts == model1.prompts
    assert [event.to_public_dict() for event in degraded0.public_transcript] == [
        event.to_public_dict() for event in degraded1.public_transcript
    ]
    assert degraded0.public_transcript != exact.public_transcript
    assert model0.prompts != exact_model.prompts
    audit = degraded0.trusted_evaluation.to_trusted_dict()["value"][
        "channel_intervention"
    ]
    assert audit["changed_invariant"] == "semantic_twin_public_fidelity"


def test_verdict_dependent_termination_changes_live_probe_control_flow() -> None:
    theta0, model0 = _run(
        "silenttwin",
        state="theta0",
        q=16,
        channel="verdict_dependent_termination",
    )
    theta1, model1 = _run(
        "silenttwin",
        state="theta1",
        q=16,
        channel="verdict_dependent_termination",
    )

    assert len(theta0.probe_plan_ids) == 2
    assert len(theta1.probe_plan_ids) == 1
    assert model0.prompts != model1.prompts
    for result in (theta0, theta1):
        evidence = result.trusted_evaluation.to_trusted_dict()["value"][
            "channel_intervention"
        ]["operational_evidence"]
        assert evidence["real_guard_evaluation_count"] == len(
            result.probe_plan_ids
        )
        assert evidence["verdict_dependent_termination_observed"] is True


def test_decision_dependent_twin_mismatch_uses_actual_profile_disposables() -> None:
    theta0, model0 = _run(
        "silenttwin",
        state="theta0",
        channel="decision_dependent_twin_mismatch",
    )
    theta1, model1 = _run(
        "silenttwin",
        state="theta1",
        channel="decision_dependent_twin_mismatch",
    )

    assert model0.prompts != model1.prompts
    assert theta0.public_transcript != theta1.public_transcript
    for result in (theta0, theta1):
        trusted = result.trusted_evaluation.to_trusted_dict()["value"]
        assert {
            row["feedback_execution_source"]
            for row in trusted["probe_monitor_evaluations"]
        } == {"actual_profile_disposable"}
        assert trusted["channel_intervention"]["operational_evidence"][
            "actual_profile_probe_observed"
        ] is True


@pytest.mark.parametrize(
    ("channel", "event_type"),
    [
        ("visible_settlement", "tool_result"),
        ("callbacks", "settlement_callback"),
        ("later_environment_read", "later_environment_read"),
        ("retained_memory", "retained_memory_observation"),
    ],
)
def test_retained_context_channels_observe_postselection_before_retirement(
    channel: str,
    event_type: str,
) -> None:
    result, model = _run("silenttwin", channel=channel)

    callback = _request(model.prompts[-1])
    assert callback["request_type"] == "postselection_observation"
    assert callback["transcript"][-1]["event_type"] == event_type  # type: ignore[index]
    assert result.retired_before_settlement is False
    assert model.is_retired is True
    evidence = result.trusted_evaluation.to_trusted_dict()["value"][
        "channel_intervention"
    ]["operational_evidence"]
    assert evidence["attacker_active_at_settlement"] is True
    assert evidence["callback_invoked"] is True
    assert evidence["callback_acknowledged"] is True
    assert evidence["postselection_delivered_to_agent"] is True
    assert result.postselection_delivered_to_agent is True
    assert result.to_record()["postselection_delivered_to_agent"] is True
    assert evidence["backend_original_contexts_retired_before_settlement"] is True
    assert evidence["retained_capability"] == "attacker_conversation_and_callback"
    serialized = json.dumps(callback, sort_keys=True)
    for private in ("fixture-theta0", "fixture-theta1"):
        assert private not in serialized


def test_retained_kv_cache_is_explicitly_not_evaluated_without_text() -> None:
    exact, exact_model = _run(
        "silenttwin", channel="exact_silenttwin"
    )
    retained, retained_model = _run(
        "silenttwin", channel="retained_kv_cache"
    )

    assert retained.public_transcript == exact.public_transcript
    assert retained_model.prompts == exact_model.prompts
    audit = retained.trusted_evaluation.to_trusted_dict()["value"][
        "channel_intervention"
    ]
    assert audit["status"] == "not_evaluated"
    assert audit["intervention_applied"] is False
    assert "cache" not in json.dumps(
        [event.to_public_dict() for event in retained.public_transcript]
    ).lower()
