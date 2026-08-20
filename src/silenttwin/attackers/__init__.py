"""Adaptive attackers and the optional model-client boundary."""

from __future__ import annotations

from typing import Any

from silenttwin.attackers.base import AttackRunResult, Attacker, BaseAttacker
from silenttwin.attackers.bayesian_attacker import BayesianAttacker, OptimalBayesianAttacker
from silenttwin.attackers.black_box_attacker import (
    BlackBoxAttacker,
    TranscriptClassifierAttacker,
    transcript_features,
)
from silenttwin.attackers.llm_attacker import LLMAttacker, ModelClient, ModelResponse, ModelUsage
from silenttwin.attackers.mock_llm import DeterministicMockAgent, MockLLM, MockLLMAttacker, MockModelClient
from silenttwin.attackers.random_attacker import RandomAttacker


ATTACKER_CLASSES: dict[str, type[Attacker]] = {
    "random": RandomAttacker,
    "random_attacker": RandomAttacker,
    "bayesian": BayesianAttacker,
    "bayesian_attacker": BayesianAttacker,
    "black_box": BlackBoxAttacker,
    "blackbox": BlackBoxAttacker,
    "mock_llm": MockLLMAttacker,
    "mock": MockLLMAttacker,
}


def make_attacker(name: str = "bayesian", **kwargs: Any) -> Attacker:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "llm":
        if "model_client" not in kwargs:
            raise ValueError("the llm attacker requires an explicit model_client; use mock_llm offline")
        return LLMAttacker(**kwargs)
    try:
        attacker_class = ATTACKER_CLASSES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted((*ATTACKER_CLASSES, "llm")))
        raise ValueError(f"unknown attacker {name!r}; choose one of: {choices}") from exc
    return attacker_class(**kwargs)


get_attacker = make_attacker


__all__ = [
    "ATTACKER_CLASSES",
    "AttackRunResult",
    "Attacker",
    "BaseAttacker",
    "BayesianAttacker",
    "BlackBoxAttacker",
    "DeterministicMockAgent",
    "LLMAttacker",
    "MockLLM",
    "MockLLMAttacker",
    "MockModelClient",
    "ModelClient",
    "ModelResponse",
    "ModelUsage",
    "OptimalBayesianAttacker",
    "RandomAttacker",
    "TranscriptClassifierAttacker",
    "get_attacker",
    "make_attacker",
    "transcript_features",
]
