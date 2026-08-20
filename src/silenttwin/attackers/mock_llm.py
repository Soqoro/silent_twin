"""Deterministic offline model client and attacker."""

from __future__ import annotations

import re
from typing import Any

from silenttwin.attackers.llm_attacker import LLMAttacker, ModelResponse, ModelUsage


class MockModelClient:
    """A zero-network deterministic stand-in for Tier-2 model clients."""

    def complete(self, prompt: str, *, seed: int = 0, max_tokens: int = 64) -> ModelResponse:
        lowered = prompt.lower()
        # Look at the cloud probe's public result.  The canonical JSON snippet
        # is intentionally parsed heuristically to exercise the same interface
        # an external model would use without accessing trusted metadata.
        cloud_position = lowered.rfind("probe_cloud_link")
        suffix = lowered[cloud_position:] if cloud_position >= 0 else lowered
        failure_markers = ("generic_failure", "policy_blocked", '"status":"denied"', '"status":"error"')
        prediction = "theta1" if cloud_position >= 0 and any(marker in suffix for marker in failure_markers) else "theta0"
        input_tokens = max(1, len(prompt.split()))
        return ModelResponse(
            text=prediction,
            usage=ModelUsage(input_tokens=input_tokens, output_tokens=1),
            metadata={"client": "deterministic-mock", "seed": seed},
        )


class MockLLMAttacker(LLMAttacker):
    name = "mock_llm"

    def __init__(self, *, seed: int = 0, model_client: MockModelClient | None = None, **kwargs: Any) -> None:
        super().__init__(model_client or MockModelClient(), seed=seed, **kwargs)


DeterministicMockAgent = MockLLMAttacker
MockLLM = MockModelClient

__all__ = ["DeterministicMockAgent", "MockLLM", "MockLLMAttacker", "MockModelClient"]
