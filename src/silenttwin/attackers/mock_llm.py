"""Deterministic structured-output model used by offline tests."""

from __future__ import annotations

import json
from typing import Any, Mapping

from silenttwin.attackers.llm_attacker import LLMAttacker, ModelResponse, ModelUsage


class MockModelClient:
    """Zero-network client that reasons only over the serialized public input."""

    model_id = "deterministic-structured-mock"
    model_revision = "v2"

    def complete(self, prompt: str, *, seed: int = 0, max_tokens: int = 128) -> ModelResponse:
        del max_tokens
        try:
            public_input = json.loads(prompt.split("PUBLIC_INPUT:\n", 1)[1])
            events = public_input.get("transcript", {}).get("events", [])
            probes = public_input.get("probe_actions", [])
            first_probe_id = probes[0].get("action_id") if probes else None
            runtime_name = str(public_input.get("runtime_name", ""))
            feedback_protocol = str(public_input.get("feedback_protocol", ""))
        except (IndexError, json.JSONDecodeError, AttributeError):
            events = []
            first_probe_id = None
            runtime_name = ""
            feedback_protocol = ""

        if "REQUEST_TYPE:probe_selection" in prompt:
            # A deterministic public-only adaptive policy: start with the
            # first channel, then switch channels after a negative observation.
            probe_ids = [
                str(item.get("action_id"))
                for item in probes
                if isinstance(item, Mapping) and item.get("action_id")
            ]
            selected = probe_ids[0] if probe_ids else ""
            if len(probe_ids) > 1 and events:
                last = events[-1] if isinstance(events[-1], Mapping) else {}
                if str(last.get("status", "")).lower() in {
                    "denied",
                    "error",
                    "refused",
                    "terminated",
                    "quarantined",
                    "temporary_rejection",
                } or bool(last.get("error")):
                    selected = probe_ids[1]
            text = json.dumps(
                {"probe_action_id": selected},
                sort_keys=True,
                separators=(",", ":"),
            )
            return ModelResponse(
                text=text,
                usage=ModelUsage(
                    input_tokens=max(1, len(prompt.split())), output_tokens=6
                ),
                metadata={
                    "client": "deterministic-structured-mock",
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "tokenizer_revision": "none",
                    "seed": seed,
                    "retries": 0,
                    "external_api_calls": 0,
                },
            )

        diagnostic_events = [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("action_id") == first_probe_id
        ]
        state_independent = feedback_protocol == "constant" or runtime_name in {
            "silenttwin",
            "silenttwin:exact",
            "silenttwin_exact",
            "exact_silenttwin",
        }
        if not diagnostic_events or state_independent:
            prediction = "theta0"
            posterior = {"theta0": 0.5, "theta1": 0.5}
        else:
            event = diagnostic_events[-1]
            status = str(event.get("status", "")).lower()
            error = event.get("error")
            timing = str(event.get("timing_bucket", "")).lower()
            identifier = str(event.get("identifier", "")).lower()
            value = event.get("value")
            negative = status in {
                "denied",
                "error",
                "refused",
                "terminated",
                "quarantined",
                "temporary_rejection",
            } or bool(error) or timing == "slow" or identifier.endswith("-rejected") or (
                isinstance(value, Mapping) and value.get("accepted") is False
            )
            prediction = "theta1" if negative else "theta0"
            posterior = (
                {"theta0": 0.0, "theta1": 1.0}
                if negative
                else {"theta0": 1.0, "theta1": 0.0}
            )
        text = json.dumps(
            {"prediction": prediction, "posterior": posterior},
            sort_keys=True,
            separators=(",", ":"),
        )
        return ModelResponse(
            text=text,
            usage=ModelUsage(input_tokens=max(1, len(prompt.split())), output_tokens=12),
            metadata={
                "client": "deterministic-structured-mock",
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_revision": "none",
                "seed": seed,
                "retries": 0,
                "external_api_calls": 0,
            },
        )


class MockLLMAttacker(LLMAttacker):
    name = "mock_llm"

    def __init__(
        self,
        *,
        model_client: MockModelClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_client or MockModelClient(), **kwargs)

    # The mock is local and deterministic, so confidence traces do not spend
    # model/API budget.  Real LLMAttacker intentionally leaves this unavailable.
    online_prediction = LLMAttacker.predict_hidden_state


DeterministicMockAgent = MockLLMAttacker
MockLLM = MockModelClient


__all__ = ["DeterministicMockAgent", "MockLLM", "MockLLMAttacker", "MockModelClient"]
