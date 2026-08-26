from __future__ import annotations

from dataclasses import dataclass

import pytest

from silenttwin.agentdojo.canonical import (
    canonicalize_tool_call,
    canonicalize_tool_schema,
    strict_json_loads,
)
from silenttwin.agentdojo.visibility import (
    TrustedOnly,
    VisibilityBoundaryError,
    assert_private_canaries_absent,
    public_value,
    trusted_value,
)
from silenttwin.io.jsonl import ResultValidationError, _check_visible_value
from silenttwin.schemas import assert_agent_visible


class _Parameters:
    def model_json_schema(self):
        return {"type": "object", "required": ["recipient"]}


class _PydanticTool:
    name = "send_email"
    description = "Send one email"
    parameters = _Parameters()


class _PydanticCall:
    def model_dump(self, *, mode: str):
        assert mode == "json"
        return {"function": "send_email", "args": {"recipient": "a@example.test"}, "id": "c1"}


@dataclass(frozen=True)
class _Secret(TrustedOnly):
    theta: str


def test_pydantic_agentdojo_objects_are_canonicalized_exactly() -> None:
    schema = canonicalize_tool_schema(_PydanticTool())
    call = canonicalize_tool_call(_PydanticCall())
    assert schema.to_dict() == {
        "name": "send_email",
        "description": "Send one email",
        "parameters": {"type": "object", "required": ["recipient"]},
    }
    assert call.to_dict() == {
        "call_id": "c1",
        "function": "send_email",
        "arguments": {"recipient": "a@example.test"},
    }


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(ValueError):
        strict_json_loads('{"plan_id":"a","plan_id":"b"}')
    with pytest.raises(ValueError):
        strict_json_loads('{"score":NaN}')


def test_typed_agentdojo_payload_accepts_application_keys_but_legacy_does_not() -> None:
    value = public_value(
        {
            "world": "workspace",
            "session": "public-session-column",
            "monitor_decision": "application-table-heading",
            "profile_id": "public-customer-profile",
            "verdict": "application-record-value",
            "attack_success": "application-field-name",
            "committed_effect_ids": ["public-workflow-effect"],
        }
    )
    assert_agent_visible(value)
    _check_visible_value(value.to_public_dict())
    with pytest.raises((ValueError, ResultValidationError)):
        _check_visible_value(value.unwrap())


def test_trusted_dto_and_private_canary_cannot_cross_public_boundary() -> None:
    with pytest.raises(VisibilityBoundaryError):
        public_value({"nested": _Secret("theta0")})
    with pytest.raises(VisibilityBoundaryError):
        public_value(
            {
                "nested": {
                    "schema": "silenttwin.agentdojo.trusted.v1",
                    "value": {"actual_hidden_state": "theta0"},
                }
            }
        )
    with pytest.raises(ValueError):
        assert_agent_visible(trusted_value({"actual_hidden_state": "theta0"}))
    with pytest.raises(ValueError):
        assert_agent_visible(
            trusted_value({"otherwise_harmless": "secret"}).to_trusted_dict()
        )
    visible = public_value({"message": "assignment-canary-91"})
    with pytest.raises(VisibilityBoundaryError):
        assert_private_canaries_absent(visible, ("assignment-canary-91",))
