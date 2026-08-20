"""Paired finite-state worlds and factories."""

from __future__ import annotations

from typing import Callable

from silenttwin.worlds.base import BaseWorld, World
from silenttwin.worlds.finite_state import FiniteStateWorld
from silenttwin.worlds.paired_world import PairedWorld, WorldPair, normalize_state
from silenttwin.worlds.templates import (
    make_calendar_world_pair,
    make_email_world_pair,
    make_files_world_pair,
    make_payments_world_pair,
)


WORLD_FACTORIES: dict[str, Callable[..., WorldPair]] = {
    "email": make_email_world_pair,
    "files": make_files_world_pair,
    "calendar": make_calendar_world_pair,
    "payments": make_payments_world_pair,
}


def build_world_pair(
    world_suite: str = "email",
    instance_id: str | int | None = None,
    seed: int = 0,
    **_: object,
) -> WorldPair:
    normalized = world_suite.strip().lower().replace("-", "_")
    try:
        factory = WORLD_FACTORIES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(WORLD_FACTORIES))
        raise ValueError(f"unknown world suite {world_suite!r}; choose one of: {choices}") from exc
    return factory(instance_id=instance_id, seed=seed)


make_world_pair = build_world_pair
world_pair_for_suite = build_world_pair


__all__ = [
    "BaseWorld",
    "FiniteStateWorld",
    "PairedWorld",
    "WORLD_FACTORIES",
    "World",
    "WorldPair",
    "build_world_pair",
    "make_world_pair",
    "normalize_state",
    "world_pair_for_suite",
]
