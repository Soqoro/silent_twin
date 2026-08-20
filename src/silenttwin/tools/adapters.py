"""Small adapters for deterministic mocks and future real tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from silenttwin.schemas import StagedEffect, ToolResult
from silenttwin.tools.base import Tool, ToolSchema


ToolFunction = Callable[[Mapping[str, Any]], ToolResult | Any]


@dataclass
class FunctionToolAdapter(Tool):
    schema: ToolSchema
    function: ToolFunction

    def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        result = self.function(dict(arguments))
        if isinstance(result, ToolResult):
            return result
        return ToolResult(ok=True, value=result, effectful=self.schema.effectful)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.schema.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.schema.name}")
        self._tools[tool.schema.name] = tool

    def invoke(self, name: str, arguments: Mapping[str, Any], *, session: Any = None) -> ToolResult:
        if session is not None:
            session.ensure_active("invoke a tool")
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc
        return tool.invoke(arguments)

    call = invoke

    @property
    def schemas(self) -> dict[str, dict[str, Any]]:
        return {name: tool.schema.to_dict() for name, tool in sorted(self._tools.items())}


@dataclass
class EffectExecutorAdapter:
    """Trusted adapter called only by the final controller."""

    function: Callable[[StagedEffect], Any]

    def __call__(self, effect: StagedEffect) -> Any:
        return self.function(effect)


ToolAdapter = FunctionToolAdapter


__all__ = [
    "EffectExecutorAdapter",
    "FunctionToolAdapter",
    "ToolAdapter",
    "ToolFunction",
    "ToolRegistry",
]
