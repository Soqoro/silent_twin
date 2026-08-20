"""Validated adjacent-world pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from silenttwin.schemas import canonical_json, stable_digest
from silenttwin.worlds.finite_state import FiniteStateWorld


def normalize_state(state: str | int | bool) -> str:
    if isinstance(state, bool):
        return "theta1" if state else "theta0"
    normalized = str(state).strip().lower().replace("θ", "theta").replace("-", "_")
    aliases = {
        "0": "theta0",
        "theta_0": "theta0",
        "theta0": "theta0",
        "world0": "theta0",
        "1": "theta1",
        "theta_1": "theta1",
        "theta1": "theta1",
        "world1": "theta1",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown private state: {state!r}") from exc


@dataclass(frozen=True)
class WorldPair:
    theta0: FiniteStateWorld
    theta1: FiniteStateWorld
    pair_id: str | None = None
    public_instance_hash: str = field(init=False)

    def __post_init__(self) -> None:
        left = canonical_json(self.theta0.public_projection())
        right = canonical_json(self.theta1.public_projection())
        if left != right:
            raise ValueError("paired worlds must have identical initial public projections")
        if normalize_state(self.theta0.private_state) != "theta0":
            raise ValueError("theta0 world must carry private state theta0")
        if normalize_state(self.theta1.private_state) != "theta1":
            raise ValueError("theta1 world must carry private state theta1")
        if self.pair_id is None:
            object.__setattr__(self, "pair_id", self.theta0.world_id.rsplit("-theta", 1)[0])
        object.__setattr__(self, "public_instance_hash", stable_digest(self.theta0.public_projection()))

    @property
    def world0(self) -> FiniteStateWorld:
        return self.theta0

    @property
    def world1(self) -> FiniteStateWorld:
        return self.theta1

    @property
    def paired_world_id(self) -> str:
        return str(self.pair_id)

    @property
    def worlds(self) -> tuple[FiniteStateWorld, FiniteStateWorld]:
        return (self.theta0, self.theta1)

    @property
    def public_projection(self):
        return self.theta0.public_projection()

    def world_for_state(self, state: str | int | bool) -> FiniteStateWorld:
        return self.theta0 if normalize_state(state) == "theta0" else self.theta1

    select = world_for_state

    def __iter__(self) -> Iterator[FiniteStateWorld]:
        return iter(self.worlds)


PairedWorld = WorldPair


__all__ = ["PairedWorld", "WorldPair", "normalize_state"]
