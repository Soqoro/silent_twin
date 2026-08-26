"""Canonical AgentDojo messages, tool schemas, calls, and hashes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from silenttwin.schemas import canonical_json, stable_digest

from .visibility import PublicEnvelope, public_value


def strict_json_loads(text: str) -> Any:
    """Parse RFC-style JSON while rejecting duplicates and non-finite values."""

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        text,
        object_pairs_hook=object_hook,
        parse_constant=reject_constant,
    )


def _copy_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    copied = json.loads(canonical_json(value))
    if not isinstance(copied, dict):
        raise TypeError("canonical mapping did not serialize as an object")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class CanonicalToolSchema:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must be non-empty")
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "parameters", _copy_mapping(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class CanonicalToolCall:
    call_id: str
    function: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("tool call ID must be non-empty")
        if not self.function:
            raise ValueError("tool function must be non-empty")
        object.__setattr__(self, "arguments", _copy_mapping(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "function": self.function,
            "arguments": dict(self.arguments),
        }

    def public_dict(self, *, protect_arguments: bool = False) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "function": self.function,
            **(
                {"arguments_digest": stable_digest(self.arguments)}
                if protect_arguments
                else {"arguments": dict(self.arguments)}
            ),
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class CanonicalMessage:
    role: str
    content: PublicEnvelope
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[CanonicalToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported canonical message role: {self.role!r}")
        if not isinstance(self.content, PublicEnvelope):
            object.__setattr__(self, "content", public_value(self.content))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool calls")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content.unwrap(),
        }
        if self.name is not None:
            result["name"] = self.name
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return result


def canonicalize_tool_schema(value: Any) -> CanonicalToolSchema:
    if isinstance(value, CanonicalToolSchema):
        return value
    # AgentDojo 0.1.35 exposes Tool/Function objects rather than OpenAI-style
    # dictionaries.  Keep that upstream shape at this boundary only; the rest
    # of SilentTwin consumes the dependency-free DTO above.
    if not isinstance(value, Mapping) and hasattr(value, "name"):
        parameters = getattr(value, "parameters", {})
        if hasattr(parameters, "model_json_schema"):
            parameters = parameters.model_json_schema()
        elif hasattr(parameters, "model_dump"):
            parameters = parameters.model_dump(mode="json")
        return CanonicalToolSchema(
            name=str(getattr(value, "name", "")),
            description=str(getattr(value, "description", "") or ""),
            parameters=parameters,
        )
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("tool schema must be a mapping")
    function = value.get("function")
    if isinstance(function, Mapping):
        value = function
    parameters = value.get("parameters", value.get("input_schema", {}))
    if not isinstance(parameters, Mapping):
        raise TypeError("tool parameters must be an object")
    return CanonicalToolSchema(
        name=str(value.get("name", "")),
        description=str(value.get("description", "")),
        parameters=parameters,
    )


def canonicalize_tool_schemas(values: Iterable[Any]) -> tuple[CanonicalToolSchema, ...]:
    # Preserve the model-visible suite order.  Hashing remains deterministic
    # because the order itself is part of the protocol.
    schemas = tuple(canonicalize_tool_schema(value) for value in values)
    names = [schema.name for schema in schemas]
    if len(names) != len(set(names)):
        raise ValueError("canonical tool schemas contain duplicate names")
    return schemas


def canonicalize_tool_call(value: Any, *, default_id: str | None = None) -> CanonicalToolCall:
    if isinstance(value, CanonicalToolCall):
        return value
    # Pydantic FunctionCall has ``model_dump`` and deliberately no ``to_dict``.
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        function = getattr(value, "function", None)
        arguments = getattr(value, "args", getattr(value, "arguments", None))
        call_id = getattr(value, "id", None)
        if function is None or arguments is None:
            raise TypeError("tool call must be a mapping or expose function/args")
        value = {"function": function, "arguments": arguments, "call_id": call_id}
    nested = value.get("function")
    if isinstance(nested, Mapping):
        function_name = nested.get("name", "")
        arguments = nested.get("arguments", {})
    else:
        function_name = value.get("name", nested or "")
        arguments = value.get("arguments", value.get("args", {}))
    if isinstance(arguments, str):
        try:
            arguments = strict_json_loads(arguments)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("tool-call arguments string is not valid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise TypeError("tool-call arguments must be an object")
    call_id = value.get("call_id", value.get("id", default_id))
    if call_id is None:
        call_id = f"call-{stable_digest([function_name, arguments])[:16]}"
    return CanonicalToolCall(str(call_id), str(function_name), arguments)


def canonicalize_message(value: Any) -> CanonicalMessage:
    if isinstance(value, CanonicalMessage):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("message must be a mapping")
    calls = tuple(
        canonicalize_tool_call(call, default_id=f"call-{index}")
        for index, call in enumerate(value.get("tool_calls", ()))
    )
    return CanonicalMessage(
        role=str(value.get("role", "")),
        content=public_value(value.get("content", "")),
        name=(str(value["name"]) if value.get("name") is not None else None),
        tool_call_id=(
            str(value["tool_call_id"])
            if value.get("tool_call_id") is not None
            else None
        ),
        tool_calls=calls,
    )


def canonicalize_messages(values: Sequence[Any]) -> tuple[CanonicalMessage, ...]:
    return tuple(canonicalize_message(value) for value in values)


def schema_hash(values: Sequence[CanonicalToolSchema]) -> str:
    return stable_digest([value.to_dict() for value in values])


def messages_hash(values: Sequence[CanonicalMessage]) -> str:
    return stable_digest([value.to_dict() for value in values])


def calls_hash(values: Sequence[CanonicalToolCall]) -> str:
    return stable_digest([value.to_dict() for value in values])


__all__ = [
    "CanonicalMessage",
    "CanonicalToolCall",
    "CanonicalToolSchema",
    "calls_hash",
    "canonicalize_message",
    "canonicalize_messages",
    "canonicalize_tool_call",
    "canonicalize_tool_schema",
    "canonicalize_tool_schemas",
    "messages_hash",
    "schema_hash",
    "strict_json_loads",
]
