"""Typed AgentDojo public/trusted serialization boundaries.

AgentDojo tool results are application data.  They may legitimately contain
keys such as ``state``, ``session``, or ``world``; consequently the Tier-1
generic-key denylist is not an appropriate security boundary.  This module
uses explicit namespace marker types instead.  Trusted objects cannot be
placed in a :class:`PublicEnvelope`, irrespective of the keys used by their
payload.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from silenttwin.schemas import canonical_json


class VisibilityBoundaryError(ValueError):
    """A trusted value was about to cross an agent-visible boundary."""


class TrustedOnly:
    """Marker base for values that must never implement public serialization."""

    __silenttwin_namespace__ = "trusted"


def _contains_trusted(value: Any, path: str = "$") -> None:
    if isinstance(value, TrustedOnly) or getattr(
        value, "__silenttwin_namespace__", None
    ) == "trusted":
        raise VisibilityBoundaryError(f"trusted value cannot enter public data at {path}")
    if isinstance(value, Mapping):
        schema = value.get("schema")
        if isinstance(schema, str) and (
            ".trusted" in schema or schema.endswith(".private.v1")
        ):
            raise VisibilityBoundaryError(
                f"serialized trusted envelope cannot enter public data at {path}"
            )
        for key, item in value.items():
            _contains_trusted(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _contains_trusted(item, f"{path}[{index}]")
        return
    if is_dataclass(value):
        for item in fields(value):
            _contains_trusted(getattr(value, item.name), f"{path}.{item.name}")


def _json_copy(value: Any) -> Any:
    """Return a detached canonical JSON value.

    Marker inspection happens before serialization so a trusted dataclass
    cannot become innocuous-looking JSON through generic dataclass traversal.
    """

    _contains_trusted(value)
    return json.loads(canonical_json(value))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True, slots=True)
class PublicEnvelope:
    """An immutable, explicitly agent-visible JSON value."""

    value: Any
    schema: str = "silenttwin.agentdojo.public.v1"
    __silenttwin_namespace__ = "public"

    def __post_init__(self) -> None:
        if not self.schema.startswith("silenttwin.agentdojo.") or ".trusted" in self.schema:
            raise VisibilityBoundaryError(
                "public envelope schema must be an AgentDojo public namespace"
            )
        object.__setattr__(self, "value", _freeze(_json_copy(self.value)))

    def to_public_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "value": _thaw(self.value)}

    def unwrap(self) -> Any:
        return _thaw(self.value)


@dataclass(frozen=True, slots=True)
class TrustedEnvelope(TrustedOnly):
    """Trusted JSON data with deliberately no ``to_public_dict`` method."""

    value: Any
    schema: str = "silenttwin.agentdojo.trusted.v1"

    def __post_init__(self) -> None:
        # Trusted data still has to be durable JSON.  It may contain other
        # trusted DTOs, so use their explicit trusted representation first.
        value = self.value
        if isinstance(value, TrustedOnly) and hasattr(value, "to_trusted_dict"):
            value = value.to_trusted_dict()
        object.__setattr__(
            self,
            "value",
            _freeze(json.loads(canonical_json(value))),
        )

    def to_trusted_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "value": _thaw(self.value)}


@dataclass(frozen=True, slots=True)
class AgentDojoVisibleMessage:
    """One model-visible message whose body is a public envelope."""

    role: str
    content: PublicEnvelope
    name: str | None = None
    tool_call_id: str | None = None
    __silenttwin_namespace__ = "public"

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if not isinstance(self.content, PublicEnvelope):
            raise TypeError("visible message content must be a PublicEnvelope")

    def to_public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content.unwrap(),
        }
        if self.name is not None:
            result["name"] = self.name
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


def public_value(value: Any, *, schema: str = "silenttwin.agentdojo.public.v1") -> PublicEnvelope:
    return PublicEnvelope(value=value, schema=schema)


def trusted_value(
    value: Any, *, schema: str = "silenttwin.agentdojo.trusted.v1"
) -> TrustedEnvelope:
    return TrustedEnvelope(value=value, schema=schema)


def assert_agent_visible_serialization(value: Any, path: str = "$") -> None:
    """Validate a durable agent-visible JSON value after deserialization.

    Marker classes protect live objects.  This complementary validator protects
    checkpointed JSON from a forged trusted envelope or a private control-plane
    DTO that has been flattened into an otherwise public-looking mapping.
    """

    if isinstance(value, TrustedOnly) or getattr(
        value, "__silenttwin_namespace__", None
    ) == "trusted":
        raise VisibilityBoundaryError(f"trusted value entered public data at {path}")
    if isinstance(value, PublicEnvelope):
        value = value.to_public_dict()
    elif isinstance(value, AgentDojoVisibleMessage):
        value = value.to_public_dict()
    if isinstance(value, Mapping):
        schema = value.get("schema")
        if isinstance(schema, str) and (
            ".trusted" in schema or schema.endswith(".private.v1")
        ):
            raise VisibilityBoundaryError(
                f"serialized trusted envelope entered public data at {path}"
            )
        namespace = value.get("__silenttwin_namespace__")
        if namespace in {"trusted", "private"}:
            raise VisibilityBoundaryError(
                f"private namespace marker entered public data at {path}"
            )
        for key, item in value.items():
            field = str(key)
            assert_agent_visible_serialization(item, f"{path}.{field}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            assert_agent_visible_serialization(item, f"{path}[{index}]")
        return
    if is_dataclass(value):
        for item in fields(value):
            assert_agent_visible_serialization(
                getattr(value, item.name), f"{path}.{item.name}"
            )


def assert_private_canaries_absent(value: Any, canaries: Sequence[str]) -> None:
    """Test/validation helper for assignment-specific secret sentinels.

    Unlike a key denylist, sentinel checking is content-sensitive and does not
    reserve ordinary application vocabulary.
    """

    materialized = value.to_public_dict() if hasattr(value, "to_public_dict") else value
    serialized = canonical_json(materialized)
    leaked = [str(canary) for canary in canaries if str(canary) in serialized]
    if leaked:
        raise VisibilityBoundaryError(
            f"private canary value(s) entered public serialization: {leaked}"
        )


__all__ = [
    "AgentDojoVisibleMessage",
    "PublicEnvelope",
    "TrustedEnvelope",
    "TrustedOnly",
    "VisibilityBoundaryError",
    "assert_agent_visible_serialization",
    "assert_private_canaries_absent",
    "public_value",
    "trusted_value",
]
