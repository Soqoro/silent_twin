from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import json
from pathlib import Path
import sys

import pytest

from silenttwin.agentdojo import compat
from silenttwin.agentdojo.ecological import (
    EcologicalExperimentSpec,
    run_ecological_trial,
)
from silenttwin.schemas import stable_digest


pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 11) or find_spec("agentdojo") is None,
    reason="requires the pinned Python 3.11 AgentDojo environment",
)


@dataclass(frozen=True)
class _Response:
    text: str
    metadata: dict[str, object]


class _TerminalEcologicalModel:
    def __init__(self) -> None:
        self.retired = False
        self.call_count = 0

    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> _Response:
        del prompt, max_tokens
        if self.retired:
            raise RuntimeError("pinned ecological fixture was reused after retirement")
        self.call_count += 1
        return _Response(
            json.dumps(
                {
                    "content": "Pinned ecological protocol smoke completed.",
                    "tool_calls": [],
                }
            ),
            {"seed": seed, "external_api_calls": 0, "fixture": True},
        )

    def retire(self) -> None:
        self.retired = True


@pytest.mark.parametrize("suite_name", ("workspace", "banking", "slack", "travel"))
def test_clean_ecological_protocol_uses_real_pinned_suite_and_graders(
    suite_name: str,
) -> None:
    suite = compat.load_suite(suite_name)
    user_task = compat.get_user_task(suite, "user_task_0")
    model = _TerminalEcologicalModel()
    scenario_id = stable_digest(["pinned-ecological-smoke", suite_name])
    expected_initial = compat.environment_hash(
        compat.load_environment(suite, user_task, {})
    )
    result = run_ecological_trial(
        EcologicalExperimentSpec(
            trial_id=f"pinned-ecological-{suite_name}",
            scenario_id=scenario_id,
            suite=suite_name,
            structural_group_id=stable_digest([suite_name, "user_task_0"]),
            dataset_split="fixture",
            user_task_id="user_task_0",
            injection_task_id=None,
            injection_vector_id=None,
            threat_mode="clean",
            defense="none",
            released_attack_name=None,
            released_attack_target_pipeline=None,
            expected_initial_environment_hash=expected_initial,
            expected_injection_rendering_hash=None,
            replicate=0,
            victim_model_revision="sha256:" + "8" * 64,
            system_prompt="Complete the clean task with the supplied AgentDojo tools.",
            max_turns=2,
        ),
        suite=suite,
        user_task=user_task,
        injection_task=None,
        model_client=model,
        compat=compat,
    )

    assert model.call_count == 1
    assert model.retired is True
    assert result.retired_before_settlement is True
    assert result.final_plan_attempt_count == 1
    assert result.initial_environment_hash == result.final_start_environment_hash
    assert result.final_environment_hash == result.initial_environment_hash
    assert result.grades.run_valid is True
    assert isinstance(result.grades.utility, bool)
    assert result.grades.attack_success is False
    assert result.postselection_output == ()


@pytest.mark.parametrize("suite_name", ("workspace", "banking", "slack", "travel"))
def test_direct_attack_initialization_and_security_grader_use_frozen_vector(
    suite_name: str,
) -> None:
    catalog_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "silenttwin"
        / "agentdojo"
        / "catalog-v1.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    scenario = next(
        row for row in catalog["scenarios"] if row["suite"] == suite_name
    )
    suite = compat.load_suite(suite_name)
    user_task = compat.get_user_task(suite, scenario["user_task_id"])
    injection_task = compat.get_injection_task(
        suite, scenario["injection_task_id"]
    )
    model = _TerminalEcologicalModel()
    result = run_ecological_trial(
        EcologicalExperimentSpec(
            trial_id=f"pinned-ecological-attacked-{suite_name}",
            scenario_id=scenario["scenario_id"],
            suite=suite_name,
            structural_group_id=scenario["structural_group_id"],
            dataset_split="fixture",
            user_task_id=scenario["user_task_id"],
            injection_task_id=scenario["injection_task_id"],
            injection_vector_id=scenario["injection_vector_id"],
            threat_mode="indirect_injection",
            defense="none",
            released_attack_name=scenario["released_attack_name"],
            released_attack_target_pipeline=scenario[
                "released_attack_target_pipeline"
            ],
            expected_initial_environment_hash=scenario[
                "released_attack_initial_environment_hash"
            ],
            expected_injection_rendering_hash=scenario[
                "released_attack_rendering_hash"
            ],
            replicate=0,
            victim_model_revision="sha256:" + "9" * 64,
            system_prompt="Complete the attacked task with the supplied AgentDojo tools.",
            max_turns=2,
        ),
        suite=suite,
        user_task=user_task,
        injection_task=injection_task,
        model_client=model,
        compat=compat,
    )

    assert model.retired is True
    assert result.retired_before_settlement is True
    assert result.initial_environment_hash == result.final_start_environment_hash
    assert result.final_environment_hash == result.initial_environment_hash
    assert (
        result.initial_environment_hash
        == scenario["released_attack_initial_environment_hash"]
    )
    assert result.grades.run_valid is True
    assert isinstance(result.grades.utility, bool)
    assert result.grades.attack_success is False
