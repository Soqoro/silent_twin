from __future__ import annotations

from importlib import metadata
from importlib.util import find_spec
import sys
from types import SimpleNamespace
from typing import Mapping

import pytest

from silenttwin.agentdojo.canonical import (
    CanonicalToolCall,
    canonicalize_tool_schemas,
)
from silenttwin.agentdojo.compat import (
    canonical_tool_schemas,
    get_injection_task,
    get_user_task,
    load_suite,
)
from silenttwin.agentdojo.controlled import ControlledExperimentSpec, run_controlled_trial
from silenttwin.agentdojo.monitors import DeterministicPolicyMonitor, MonitorPair
from silenttwin.agentdojo.pipeline import StructuredControlledAttacker
from silenttwin.agentdojo.runner import _assignment_cells
from silenttwin.backends.agentdojo import AgentDojoBackend
from silenttwin.backends.base import (
    EnvironmentRole,
    PublicPlan,
    PublicScenario,
    TrustedPlan,
)
from silenttwin.schemas import canonical_json, stable_digest
from tests.fakes.fake_backend import FakeControlledModel


def _has_pinned_agentdojo() -> bool:
    if sys.version_info[:2] != (3, 11) or find_spec("agentdojo") is None:
        return False
    try:
        return metadata.version("agentdojo") == "0.1.35"
    except metadata.PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_pinned_agentdojo(),
    reason="requires Python 3.11 and AgentDojo 0.1.35",
)


NO_ARG_TOOLS = {
    "workspace": "list_files",
    "banking": "get_balance",
    "slack": "get_channels",
    "travel": "get_user_information",
}
INJECTION_IDS = {
    "workspace": "injection_task_0",
    "banking": "injection_task_0",
    "slack": "injection_task_1",
    "travel": "injection_task_6",
}
MUTATING_CALLS = {
    "workspace": (
        "create_file",
        {"filename": "silenttwin-isolation.txt", "content": "isolation sentinel"},
    ),
    "banking": (
        "update_password",
        {"password": "SilentTwin-test-only-password"},
    ),
    "slack": (
        "post_webpage",
        {
            "url": "https://silenttwin.invalid/isolation",
            "content": "isolation sentinel",
        },
    ),
    "travel": (
        "create_calendar_event",
        {
            "title": "SilentTwin isolation",
            "start_time": "2025-01-01 09:00",
            "end_time": "2025-01-01 10:00",
        },
    ),
}


def _make_smoke_backend(suite_name: str) -> AgentDojoBackend:
    suite = load_suite(suite_name)
    user = get_user_task(suite, "user_task_0")
    injection = get_injection_task(suite, INJECTION_IDS[suite_name])
    schemas = canonicalize_tool_schemas(canonical_tool_schemas(suite))
    function = NO_ARG_TOOLS[suite_name]
    plans = tuple(
        PublicPlan(
            plan_id=plan_id,
            label=plan_id,
            description=f"Pinned smoke {plan_id}",
            kind="probe" if plan_id.startswith("probe_") else "candidate",
            public_steps=({"function": function},),
        )
        for plan_id in ("probe_0", "probe_1", "candidate_0", "candidate_1")
    )
    scenario = PublicScenario(
        scenario_id=stable_digest(["pinned-backend-smoke", suite_name]),
        suite=suite_name,
        user_prompt=str(user.PROMPT),
        tool_schemas=schemas,
        candidate_plans=plans,
        structural_group_id=stable_digest([suite_name, "user_task_0"]),
        dataset_split="fixture",
    )
    trusted = {
        plan.plan_id: TrustedPlan(
            plan.plan_id,
            (CanonicalToolCall(f"{plan.plan_id}-call", function, {}),),
            ground_truth_label=(
                "prohibited" if plan.kind == "candidate" else "permitted"
            ),
            registry_revision="pinned-smoke-v1",
        )
        for plan in plans
    }
    theta0 = DeterministicPolicyMonitor(
        profile_id=f"{suite_name}-theta0",
        decisions={
            "probe_0": "allow",
            "probe_1": "block",
            "candidate_0": "allow",
            "candidate_1": "block",
        },
    )
    theta1 = DeterministicPolicyMonitor(
        profile_id=f"{suite_name}-theta1",
        decisions={
            "probe_0": "block",
            "probe_1": "allow",
            "candidate_0": "block",
            "candidate_1": "allow",
        },
    )
    return AgentDojoBackend(
        suite=suite,
        user_task=user,
        injection_task=injection,
        injections={},
        public_scenario=scenario,
        trusted_plans=trusted,
        monitor_pair=MonitorPair(
            theta0, theta1, stable_digest([suite_name, "pair"])
        ),
    )


def _make_call_backend(
    suite_name: str,
    calls_by_plan: Mapping[str, tuple[CanonicalToolCall, ...]],
    *,
    theta0_decisions: Mapping[str, str],
    theta1_decisions: Mapping[str, str] | None = None,
) -> AgentDojoBackend:
    """Build a small backend while retaining the pinned suite's real runtime."""

    suite = load_suite(suite_name)
    user = get_user_task(suite, "user_task_0")
    injection = get_injection_task(suite, INJECTION_IDS[suite_name])
    schemas = canonicalize_tool_schemas(canonical_tool_schemas(suite))
    plans = tuple(
        PublicPlan(
            plan_id=plan_id,
            label=plan_id,
            description=f"Pinned protocol invariant {plan_id}",
            kind="probe",
            public_steps=tuple({"function": call.function} for call in calls),
        )
        for plan_id, calls in calls_by_plan.items()
    )
    scenario = PublicScenario(
        scenario_id=stable_digest(
            {
                "protocol_test": suite_name,
                "plans": {
                    plan_id: [call.to_dict() for call in calls]
                    for plan_id, calls in calls_by_plan.items()
                },
            }
        ),
        suite=suite_name,
        user_prompt=str(user.PROMPT),
        tool_schemas=schemas,
        candidate_plans=plans,
        structural_group_id=stable_digest([suite_name, "protocol-invariants"]),
        dataset_split="fixture",
    )
    trusted = {
        plan_id: TrustedPlan(
            plan_id=plan_id,
            calls=calls,
            atomic_group=f"atomic-{plan_id}",
            ground_truth_label="prohibited",
            registry_revision="pinned-protocol-invariants-v1",
        )
        for plan_id, calls in calls_by_plan.items()
    }
    theta1 = theta1_decisions or theta0_decisions
    return AgentDojoBackend(
        suite=suite,
        user_task=user,
        injection_task=injection,
        injections={},
        public_scenario=scenario,
        trusted_plans=trusted,
        monitor_pair=MonitorPair(
            DeterministicPolicyMonitor(
                profile_id=f"{suite_name}-protocol-theta0",
                decisions=theta0_decisions,
            ),
            DeterministicPolicyMonitor(
                profile_id=f"{suite_name}-protocol-theta1",
                decisions=theta1,
            ),
            stable_digest([suite_name, "protocol-invariant-pair"]),
        ),
    )


@pytest.mark.parametrize("suite_name", ("workspace", "banking", "slack", "travel"))
def test_real_backend_clone_tool_and_grader_path(suite_name: str) -> None:
    backend = _make_smoke_backend(suite_name)
    base = backend.fresh_environment("theta0", EnvironmentRole.BASE, 3)
    clone = backend.clone_environment(base, EnvironmentRole.GENUINE_PROBE)
    before = backend.environment_hash(base)
    probe = backend.execute_probe(clone, "theta0", "probe_0")
    assert probe.valid
    assert backend.environment_hash(base) == before
    final = backend.fresh_environment("theta0", EnvironmentRole.FINAL_TARGET, 3)
    result = backend.execute_final(final, "theta0", "candidate_0")
    grades = backend.grade(final, result)
    assert result.valid
    assert isinstance(grades.utility, bool)
    assert isinstance(grades.attack_success, bool)


@pytest.mark.parametrize("suite_name", ("workspace", "banking", "slack", "travel"))
def test_controlled_blocks_use_production_balanced_assignments(
    suite_name: str,
) -> None:
    """Run the required CPU blocks through the production assignment schedule."""

    backend = _make_smoke_backend(suite_name)
    blocks = (
        ("e1", "genuine", "genuine", 4),
        ("e1", "matched_shuffled", "matched_shuffled", 4),
        ("e2", "no_probe", "genuine", 0),
        ("e2", "genuine", "genuine", 4),
        ("e2", "matched_shuffled", "matched_shuffled", 4),
        ("e2", "random_final", "genuine", 4),
        ("e2", "oracle", "genuine", 0),
    )
    expected_e1_genuine = (("theta0", None), ("theta1", None))
    expected_cross = (
        ("theta0", "theta0"),
        ("theta0", "theta1"),
        ("theta1", "theta0"),
        ("theta1", "theta1"),
    )

    for experiment, condition, feedback_source, query_budget in blocks:
        # _run_controlled_member consumes this exact production helper. Using it
        # here prevents an ad hoc integration loop from silently drifting back
        # to a single theta/donor row.
        assignments = _assignment_cells(
            SimpleNamespace(
                experiment_id=experiment,
                feedback_source=feedback_source,
            )
        )
        expected_assignments = (
            expected_e1_genuine
            if experiment == "e1" and condition == "genuine"
            else expected_cross
        )
        assert assignments == expected_assignments

        observed: list[tuple[str, str | None]] = []
        for actual_state, donor_state in assignments:
            attacker = StructuredControlledAttacker(
                FakeControlledModel(),
                immutable_model_revision="sha256:" + "9" * 64,
            )
            controlled = run_controlled_trial(
                ControlledExperimentSpec(
                    trial_id=(
                        f"{suite_name}-{experiment}-{condition}-"
                        f"{actual_state}-{donor_state or 'none'}"
                    ),
                    experiment_id=experiment,
                    condition=condition,
                    feedback_policy="generic_failure",
                    query_budget=query_budget,
                    actual_state=actual_state,
                    donor_state=donor_state,
                    replicate=0,
                    user_task_id="user_task_0",
                    injection_task_id=INJECTION_IDS[suite_name],
                    injection_vector_id="pinned-smoke-no-injection",
                ),
                backend,
                attacker,
            )
            observed.append((controlled.actual_state, controlled.donor_state))
            assert (
                controlled.initial_environment_hash
                == controlled.final_start_environment_hash
            )
            assert controlled.retired_before_settlement
            assert controlled.grades.run_valid
            assert controlled.final_plan_attempt_count == (
                1 if experiment == "e2" else 0
            )

        assert tuple(observed) == expected_assignments


@pytest.mark.parametrize("suite_name", ("workspace", "banking", "slack", "travel"))
def test_silenttwin_agent_visible_bytes_are_theta_independent(
    suite_name: str,
) -> None:
    observations = []
    for theta in ("theta0", "theta1"):
        backend = _make_smoke_backend(suite_name)
        model = FakeControlledModel()
        attacker = StructuredControlledAttacker(
            model,
            immutable_model_revision="sha256:" + "8" * 64,
        )
        result = run_controlled_trial(
            ControlledExperimentSpec(
                trial_id=f"{suite_name}-silenttwin-{theta}",
                experiment_id="e1",
                condition="silenttwin",
                feedback_policy="silenttwin",
                query_budget=4,
                actual_state=theta,
                donor_state=None,
                replicate=11,
                user_task_id="user_task_0",
                injection_task_id=INJECTION_IDS[suite_name],
                injection_vector_id="pinned-protocol-no-injection",
            ),
            backend,
            attacker,
        )
        observations.append(
            (
                canonical_json(
                    [event.to_public_dict() for event in result.public_transcript]
                ).encode("utf-8"),
                tuple(prompt.encode("utf-8") for prompt in model.prompts),
                tuple(model.seeds),
                result.initial_environment_hash,
                result.final_start_environment_hash,
            )
        )
        assert result.retired_before_settlement

    # These are the complete attacker inputs, not a projection of selected
    # fields.  Hidden theta is permitted only in trusted evaluation records.
    assert observations[0] == observations[1]


@pytest.mark.parametrize("suite_name", ("workspace", "banking", "slack", "travel"))
def test_target_and_shuffled_donor_are_state_isolated(suite_name: str) -> None:
    function, arguments = MUTATING_CALLS[suite_name]
    plan_id = "mutating_probe"
    backend = _make_call_backend(
        suite_name,
        {
            plan_id: (
                CanonicalToolCall(
                    f"{suite_name}-mutation", function, dict(arguments)
                ),
            )
        },
        theta0_decisions={plan_id: "allow"},
    )
    target = backend.fresh_environment("theta0", EnvironmentRole.BASE, 29)
    donor = backend.fresh_environment(
        "theta1", EnvironmentRole.SHUFFLED_DONOR, 29
    )
    target_before = backend.environment_hash(target)
    assert donor.environment is not target.environment
    assert backend.environment_hash(donor) == target_before

    donor_result = backend.execute_probe(donor, "theta1", plan_id, seed=31)

    assert donor_result.valid
    assert donor_result.staged_effect_ids
    assert donor_result.committed_effect_ids == ()
    assert backend.environment_hash(donor) != target_before
    assert backend.environment_hash(target) == target_before


class _SettlementGuardAttacker(StructuredControlledAttacker):
    def __init__(self, model: FakeControlledModel) -> None:
        super().__init__(model, immutable_model_revision="sha256:" + "6" * 64)
        self.settlement_started = False
        self.callbacks_after_settlement: list[str] = []

    def _assert_pre_settlement(self, name: str) -> None:
        if self.settlement_started:
            self.callbacks_after_settlement.append(name)
            raise AssertionError(f"attacker callback {name} ran after settlement")

    def choose_probe(self, *args, **kwargs):
        self._assert_pre_settlement("choose_probe")
        return super().choose_probe(*args, **kwargs)

    def predict_hidden_state(self, *args, **kwargs):
        self._assert_pre_settlement("predict_hidden_state")
        return super().predict_hidden_state(*args, **kwargs)

    def choose_final(self, *args, **kwargs):
        self._assert_pre_settlement("choose_final")
        return super().choose_final(*args, **kwargs)


def test_attacker_has_no_callback_after_live_target_settlement_begins() -> None:
    backend = _make_smoke_backend("workspace")
    model = FakeControlledModel()
    attacker = _SettlementGuardAttacker(model)
    original_fresh = backend.fresh_environment
    original_final = backend.execute_final
    prompt_count_at_settlement: list[int] = []

    def boundary_fresh(theta, role, seed):
        if role is EnvironmentRole.FINAL_TARGET:
            attacker.settlement_started = True
            prompt_count_at_settlement.append(len(model.prompts))
            assert attacker.retired
            assert model.is_retired
        return original_fresh(theta, role, seed)

    def checked_final(environment, theta, plan_id, *, seed=0):
        assert attacker.settlement_started
        assert attacker.retired
        assert model.is_retired
        assert len(model.prompts) == prompt_count_at_settlement[-1]
        return original_final(environment, theta, plan_id, seed=seed)

    backend.fresh_environment = boundary_fresh  # type: ignore[method-assign]
    backend.execute_final = checked_final  # type: ignore[method-assign]
    result = run_controlled_trial(
        ControlledExperimentSpec(
            trial_id="workspace-settlement-callback-boundary",
            experiment_id="e2",
            condition="genuine",
            feedback_policy="generic_failure",
            query_budget=4,
            actual_state="theta0",
            donor_state=None,
            replicate=5,
            user_task_id="user_task_0",
            injection_task_id=INJECTION_IDS["workspace"],
            injection_vector_id="pinned-protocol-no-injection",
        ),
        backend,
        attacker,
    )

    assert result.retired_before_settlement
    assert result.final_result is not None
    assert prompt_count_at_settlement == [6]  # Q probes + prediction + final choice.
    assert len(model.prompts) == prompt_count_at_settlement[0]
    assert attacker.callbacks_after_settlement == []


def test_rejected_and_dependency_discarded_effects_never_commit() -> None:
    valid_first = CanonicalToolCall(
        "create-before-error",
        "create_file",
        {"filename": "must-be-rolled-back.txt", "content": "temporary"},
    )
    invalid_middle = CanonicalToolCall(
        "create-error",
        "create_file",
        {"filename": "missing-required-content.txt"},
    )
    valid_dependent = CanonicalToolCall(
        "create-after-error",
        "create_file",
        {"filename": "must-not-run.txt", "content": "unreachable"},
    )
    backend = _make_call_backend(
        "workspace",
        {
            "monitor_rejected": (valid_first,),
            "atomic_failure": (valid_first, invalid_middle, valid_dependent),
        },
        theta0_decisions={
            "monitor_rejected": "block",
            "atomic_failure": "allow",
        },
    )

    rejected_environment = backend.fresh_environment(
        "theta0", EnvironmentRole.FINAL_TARGET, 41
    )
    rejected_before = backend.environment_hash(rejected_environment)
    rejected = backend.execute_final(
        rejected_environment, "theta0", "monitor_rejected", seed=43
    )
    assert rejected.monitor.decision == "block"
    assert rejected.committed_effect_ids == ()
    assert rejected.rejected_effect_ids == rejected.staged_effect_ids
    assert [action.status for action in rejected.trusted_actions] == ["rejected"]
    assert backend.environment_hash(rejected_environment) == rejected_before

    failed_environment = backend.fresh_environment(
        "theta0", EnvironmentRole.FINAL_TARGET, 41
    )
    failed_before = backend.environment_hash(failed_environment)
    failed = backend.execute_final(
        failed_environment, "theta0", "atomic_failure", seed=43
    )
    assert not failed.valid
    assert failed.committed_effect_ids == ()
    assert failed.rejected_effect_ids == failed.staged_effect_ids
    assert [action.status for action in failed.trusted_actions] == [
        "dependency_discarded",
        "error",
        "dependency_discarded",
    ]
    assert backend.environment_hash(failed_environment) == failed_before
