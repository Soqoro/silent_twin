"""Fallible enforcement monitor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from silenttwin.schemas import ActionProposal, MonitorDecision


class Monitor(ABC):
    """Predicts allow/block; it is not the source of truth for safety."""

    @abstractmethod
    def decide(self, action: ActionProposal) -> MonitorDecision:
        raise NotImplementedError

    def evaluate(self, action: ActionProposal) -> MonitorDecision:
        return self.decide(action)

    def allows(self, action: ActionProposal) -> bool:
        return self.decide(action) is MonitorDecision.ALLOW

    def predicts_allow(self, action: ActionProposal) -> bool:
        return self.allows(action)


__all__ = ["Monitor"]
