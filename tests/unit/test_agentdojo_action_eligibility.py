from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from silenttwin.agentdojo.action_eligibility import (
    ActionEligibilityError,
    execute_and_grade_action_plan,
    pilot_scenario_ids,
    validate_action_eligibility_manifest,
    validate_action_validation,
    validate_distinct_required_action_plans,
)
from silenttwin.agentdojo.canonical import CanonicalToolCall, CanonicalToolSchema
from silenttwin.agentdojo.cli import main as artifact_cli_main
from silenttwin.agentdojo.config import stable_hash
from silenttwin.backends.base import TrustedPlan


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(name: str) -> dict[str, object]:
    return json.loads(
        (REPO_ROOT / "configs/silenttwin/agentdojo" / name).read_text(
            encoding="utf-8"
        )
    )


def _rehashed(document: dict[str, object]) -> dict[str, object]:
    payload = dict(document)
    payload.pop("action_eligibility_manifest_hash")
    document["action_eligibility_manifest_hash"] = stable_hash(payload)
    return document


def _plan(plan_id: str, *calls: CanonicalToolCall) -> TrustedPlan:
    return TrustedPlan(plan_id=plan_id, calls=tuple(calls))


def test_checked_action_eligibility_has_exact_estimation_cohorts() -> None:
    catalog = _read("catalog-v1.json")
    splits = _read("splits-v1.json")
    eligibility = _read("action-eligibility-v1.json")

    assert validate_action_eligibility_manifest(
        eligibility, catalog=catalog, split_manifest=splits
    ) == eligibility["action_eligibility_manifest_hash"]
    assert len(pilot_scenario_ids(eligibility, dataset_split="train")) == 134
    assert (
        len(pilot_scenario_ids(eligibility, dataset_split="development"))
        == 59
    )
    assert pilot_scenario_ids(eligibility, dataset_split="test") == ()
    assert eligibility["held_out_evaluation_permitted"] is False
    assert eligibility["confirmatory_claim_permitted"] is False


def test_action_eligibility_cli_freezes_immutably(tmp_path: Path) -> None:
    output = tmp_path / "eligibility.json"
    arguments = [
        "freeze-action-eligibility",
        "--catalog",
        str(REPO_ROOT / "configs/silenttwin/agentdojo/catalog-v1.json"),
        "--splits",
        str(REPO_ROOT / "configs/silenttwin/agentdojo/splits-v1.json"),
        "--output",
        str(output),
        "--assert-no-learned-outcomes-inspected",
    ]
    assert artifact_cli_main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == _read(
        "action-eligibility-v1.json"
    )
    assert artifact_cli_main(arguments) == 0


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda value: value.__setitem__(
                "learned_monitor_outcomes_inspected", True
            ),
            "contaminated",
        ),
        (
            lambda value: value["pilot_scenario_ids_by_split"]["train"].pop(),
            "pilot scenario enumeration",
        ),
        (
            lambda value: value.__setitem__(
                "claim_boundary", "estimation and confirmation permitted"
            ),
            "claim boundary drifted",
        ),
    ),
)
def test_rehashed_action_eligibility_semantic_tampering_is_rejected(
    mutation: Callable[[dict[str, Any]], None], match: str
) -> None:
    catalog = _read("catalog-v1.json")
    splits = _read("splits-v1.json")
    eligibility = copy.deepcopy(_read("action-eligibility-v1.json"))
    mutation(eligibility)
    _rehashed(eligibility)
    with pytest.raises(ActionEligibilityError, match=match):
        validate_action_eligibility_manifest(
            eligibility, catalog=catalog, split_manifest=splits
        )


def test_required_action_comparison_rejects_representational_variants() -> None:
    schema = CanonicalToolSchema(
        name="transfer",
        description="fixture",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "integer"},
                "note": {"type": "string", "default": ""},
            },
            "required": ["amount"],
        },
    )
    amount_1 = CanonicalToolCall("a", "transfer", {"amount": 1, "note": "x"})
    amount_2 = CanonicalToolCall("b", "transfer", {"amount": 2, "note": "x"})
    hashes = validate_distinct_required_action_plans(
        {"left": _plan("left", amount_1), "right": _plan("right", amount_2)},
        schemas=(schema,),
    )
    assert hashes["left"] != hashes["right"]

    amount_3 = CanonicalToolCall("d", "transfer", {"amount": 3, "note": "x"})
    pool_hashes = validate_distinct_required_action_plans(
        {
            "left": _plan("left", amount_1),
            "middle": _plan("middle", amount_2),
            "right": _plan("right", amount_3),
        },
        schemas=(schema,),
    )
    assert len(set(pool_hashes.values())) == 3

    with pytest.raises(ActionEligibilityError, match="at least two"):
        validate_distinct_required_action_plans(
            {"only": _plan("only", amount_1)}, schemas=(schema,)
        )

    optional_only = CanonicalToolCall(
        "c", "transfer", {"amount": 1, "note": "different"}
    )
    with pytest.raises(ActionEligibilityError, match="identical or nested"):
        validate_distinct_required_action_plans(
            {
                "left": _plan("left", amount_1),
                "right": _plan("right", optional_only),
            },
            schemas=(schema,),
        )
    with pytest.raises(ActionEligibilityError, match="identical or nested"):
        validate_distinct_required_action_plans(
            {
                "left": _plan("left", amount_1, amount_2),
                "right": _plan("right", amount_2, amount_1),
            },
            schemas=(schema,),
        )
    with pytest.raises(ActionEligibilityError, match="identical or nested"):
        validate_distinct_required_action_plans(
            {
                "left": _plan("left", amount_1),
                "right": _plan("right", amount_1, amount_2),
                "third": _plan("third", amount_3),
            },
            schemas=(schema,),
        )


class _FakeCompat:
    def __init__(self, *, attack_success: bool = True, error: str | None = None):
        self.attack_success = attack_success
        self.error = error

    @staticmethod
    def load_environment(
        suite: object, user_task: object, injections: object
    ) -> dict[str, object]:
        return {"balance": 0, "injections": injections}

    @staticmethod
    def clone_environment(environment: dict[str, object]) -> dict[str, object]:
        return copy.deepcopy(environment)

    @staticmethod
    def environment_hash(environment: dict[str, object]) -> str:
        return stable_hash(environment)

    def execute_call(
        self, suite: object, environment: dict[str, object], call: dict[str, object]
    ) -> SimpleNamespace:
        if self.error is None:
            environment["balance"] = int(environment["balance"]) + int(
                call["arguments"]["amount"]
            )
        return SimpleNamespace(error=self.error, trace=(call,))

    @staticmethod
    def canonical_call(call: dict[str, object]) -> dict[str, object]:
        return {
            "function": call["function"],
            "args": dict(call["arguments"]),
        }

    def grade_attack_success(self, *args: object) -> bool:
        return self.attack_success


def test_action_validation_executes_fresh_and_requires_released_success() -> None:
    compat = _FakeCompat()
    initial = compat.load_environment(None, None, {"vector": "goal"})
    scenario = {
        "scenario_id": "scenario-1",
        "suite": "banking",
        "dataset_split": "train",
        "injection_vector_id": "vector",
        "initial_environment_hash": compat.environment_hash(initial),
    }
    plan = _plan(
        "strategy-a",
        CanonicalToolCall("call-1", "transfer", {"amount": 2}),
    )
    eligibility_hash = "a" * 64
    required_hash = "b" * 64
    record = execute_and_grade_action_plan(
        compat=compat,
        suite=object(),
        user_task=object(),
        injection_task=SimpleNamespace(GOAL="goal"),
        scenario=scenario,
        strategy_id="strategy-a",
        plan=plan,
        action_eligibility_manifest_hash=eligibility_hash,
        required_action_multiset_hash=required_hash,
    )
    assert record["executed_call_count"] == 1
    assert record["normalized_attack_success"] is True
    validate_action_validation(
        record,
        action_eligibility_manifest_hash=eligibility_hash,
        scenario_id="scenario-1",
        strategy_id="strategy-a",
    )

    with pytest.raises(ActionEligibilityError, match="did not pass"):
        execute_and_grade_action_plan(
            compat=_FakeCompat(attack_success=False),
            suite=object(),
            user_task=object(),
            injection_task=SimpleNamespace(GOAL="goal"),
            scenario=scenario,
            strategy_id="strategy-a",
            plan=plan,
            action_eligibility_manifest_hash=eligibility_hash,
            required_action_multiset_hash=required_hash,
        )
    with pytest.raises(ActionEligibilityError, match="failed"):
        execute_and_grade_action_plan(
            compat=_FakeCompat(error="schema error"),
            suite=object(),
            user_task=object(),
            injection_task=SimpleNamespace(GOAL="goal"),
            scenario=scenario,
            strategy_id="strategy-a",
            plan=plan,
            action_eligibility_manifest_hash=eligibility_hash,
            required_action_multiset_hash=required_hash,
        )
