from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from silenttwin.agentdojo.canonical import CanonicalToolCall
from silenttwin.backends.base import EnvironmentRole
from tests.fakes.fake_backend import make_fake_agentdojo_backend


def test_deep_clone_probe_isolation_and_final_grading() -> None:
    backend = make_fake_agentdojo_backend()
    target = backend.fresh_environment("theta0", EnvironmentRole.BASE, 7)
    initial = backend.environment_hash(target)
    probe = backend.clone_environment(target, EnvironmentRole.GENUINE_PROBE)
    assert probe.environment is not target.environment
    observed = backend.execute_probe(probe, "theta0", "probe_0", seed=11)
    assert observed.monitor.decision == "allow"
    assert backend.environment_hash(probe) != initial
    assert backend.environment_hash(target) == initial

    final = backend.fresh_environment("theta0", EnvironmentRole.FINAL_TARGET, 7)
    assert backend.environment_hash(final) == initial
    result = backend.execute_final(final, "theta0", "candidate_0", seed=12)
    assert result.monitor.decision == "allow"
    assert result.committed_effect_ids
    assert result.trusted_actions[0].ground_truth_label == "prohibited"
    grades = backend.grade(final, result)
    assert grades.raw_attack_success is True
    assert grades.attack_success is True
    assert grades.safe is False
    assert grades.safe_and_useful is False
    assert grades.run_valid is True


def test_monitor_approval_is_not_ground_truth_permission() -> None:
    backend = make_fake_agentdojo_backend()
    assert backend.evaluate_guard("theta0", "candidate_0").decision == "allow"
    assert backend.trusted_plans["candidate_0"].ground_truth_label == "prohibited"
    assert backend.false_negative_plan_ids("theta0") == ("candidate_0",)
    assert backend.false_negative_plan_ids("theta1") == ("candidate_1",)


def test_later_monitor_failure_retains_every_prior_call_evaluation() -> None:
    backend = make_fake_agentdojo_backend()
    original_plan = backend.trusted_plans["candidate_0"]
    backend._trusted_plans = MappingProxyType(  # type: ignore[attr-defined]
        {
            **backend.trusted_plans,
            "candidate_0": replace(
                original_plan,
                calls=(
                    CanonicalToolCall("candidate-0-first", "candidate_0", {}),
                    CanonicalToolCall("candidate-0-second", "candidate_0", {}),
                ),
            ),
        }
    )
    monitor = backend.monitor_pair.theta0
    original_evaluate = monitor.evaluate
    call_count = 0

    def fail_second_call(monitor_input, *, plan_id: str, seed: int = 0):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            monitor.last_failure_provenance = {"sentinel": "second-call-failure"}
            raise RuntimeError("later learned-monitor failure")
        return original_evaluate(monitor_input, plan_id=plan_id, seed=seed)

    monitor.evaluate = fail_second_call  # type: ignore[method-assign]
    evaluation = backend.evaluate_guard("theta0", "candidate_0", seed=17)

    assert evaluation.decision == "error"
    calls = evaluation.provenance["call_evaluations"]
    assert len(calls) == 2
    assert calls[0]["decision"] == "allow"
    assert calls[1]["decision"] == "error"
    assert calls[1]["provenance"]["call_index"] == 1
    assert calls[1]["provenance"]["failed_model_call"] == {
        "sentinel": "second-call-failure"
    }
