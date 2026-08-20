"""World interfaces and public-projection helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from silenttwin.schemas import ActionProposal


class World(ABC):
    @abstractmethod
    def public_projection(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def action(self, action_id: str) -> ActionProposal:
        raise NotImplementedError


BaseWorld = World


__all__ = ["BaseWorld", "World"]
