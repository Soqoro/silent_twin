from __future__ import annotations

from silenttwin.backends.base import EnvironmentRole
from silenttwin.backends.finite_state import FiniteStateBackend
from silenttwin.worlds import build_world_pair


def test_tier1_world_pair_satisfies_shared_backend_contract() -> None:
    pair = build_world_pair("email", instance_index=0, seed=4)
    backend = FiniteStateBackend(pair, runtime_name="generic_failure")
    target = backend.fresh_environment("theta0", EnvironmentRole.BASE, 9)
    initial = backend.environment_hash(target)
    probe = backend.clone_environment(target, EnvironmentRole.GENUINE_PROBE)
    probe_result = backend.execute_probe(
        probe, "theta0", pair.theta0.probes[0].action_id
    )
    assert probe_result.public_events
    assert backend.environment_hash(target) == initial

    final = backend.fresh_environment("theta0", EnvironmentRole.FINAL_TARGET, 9)
    assert backend.environment_hash(final) == initial
    candidate = pair.theta0.exploits[0]
    result = backend.execute_final(final, "theta0", candidate.action_id)
    grades = backend.grade(final, result)
    assert result.trusted_actions[0].ground_truth_label == "prohibited"
    assert grades.attack_success == bool(result.committed_effect_ids)
