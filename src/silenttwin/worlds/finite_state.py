"""Deterministic finite-state world used by Tier-1 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from silenttwin.policy.monitor import Monitor
from silenttwin.policy.true_policy import TruePolicy
from silenttwin.schemas import ActionProposal, canonical_json
from silenttwin.worlds.base import World


@dataclass
class FiniteStateWorld(World):
    world_id: str
    suite: str
    public_task: str
    public_environment: Mapping[str, Any]
    tool_schemas: Mapping[str, Any]
    public_runtime_config: Mapping[str, Any]
    private_state: str
    true_policy: TruePolicy
    monitor: Monitor
    actions: tuple[ActionProposal, ...]
    policy_epoch: int = 0
    _action_index: dict[str, ActionProposal] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.world_id:
            raise ValueError("world_id must be non-empty")
        if not self.suite:
            raise ValueError("suite must be non-empty")
        # Canonical round-trip provides defensive copies and JSON validation.
        import json

        self.public_environment = MappingProxyType(json.loads(canonical_json(self.public_environment)))
        self.tool_schemas = MappingProxyType(json.loads(canonical_json(self.tool_schemas)))
        self.public_runtime_config = MappingProxyType(json.loads(canonical_json(self.public_runtime_config)))
        self.actions = tuple(self.actions)
        self._action_index = {action.action_id: action for action in self.actions}
        if len(self._action_index) != len(self.actions):
            raise ValueError("action IDs must be unique within a world")

    @property
    def theta(self) -> str:
        """Trusted alias.  Never include this property in public projection."""

        return self.private_state

    @property
    def state(self) -> str:
        return self.private_state

    @property
    def action_catalog(self) -> Mapping[str, ActionProposal]:
        return MappingProxyType(self._action_index)

    def action(self, action_id: str) -> ActionProposal:
        try:
            return self._action_index[action_id]
        except KeyError as exc:
            raise KeyError(f"unknown action {action_id!r} in {self.suite} world") from exc

    get_action = action

    def actions_with_kind(self, kind: str) -> tuple[ActionProposal, ...]:
        return tuple(action for action in self.actions if action.public_metadata.get("kind") == kind)

    @property
    def probes(self) -> tuple[ActionProposal, ...]:
        return self.actions_with_kind("probe")

    @property
    def exploits(self) -> tuple[ActionProposal, ...]:
        return self.actions_with_kind("exploit")

    @property
    def workflow_actions(self) -> tuple[ActionProposal, ...]:
        return self.actions_with_kind("workflow")

    def public_projection(self) -> Mapping[str, Any]:
        return {
            "suite": self.suite,
            "public_task": self.public_task,
            "public_environment": dict(self.public_environment),
            "tool_schemas": dict(self.tool_schemas),
            "public_runtime_config": dict(self.public_runtime_config),
            "actions": [action.to_dict() for action in self.actions],
        }

    def trusted_projection(self) -> Mapping[str, Any]:
        return {
            **self.public_projection(),
            "world_id": self.world_id,
            "private_state": self.private_state,
            "policy_epoch": self.policy_epoch,
        }


__all__ = ["FiniteStateWorld"]
