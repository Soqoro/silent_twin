"""Tool interfaces shared by Tier-1 mocks and Tier-2 adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from silenttwin.schemas import ToolResult


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    effectful: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "effectful": self.effectful,
        }


class Tool(ABC):
    schema: ToolSchema

    @abstractmethod
    def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        raise NotImplementedError

    def __call__(self, **arguments: Any) -> ToolResult:
        return self.invoke(arguments)


BaseTool = Tool


__all__ = ["BaseTool", "Tool", "ToolResult", "ToolSchema"]
