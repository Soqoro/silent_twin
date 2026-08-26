"""Paired finite-state worlds and factories."""

from __future__ import annotations

from typing import Callable

from silenttwin.worlds.base import BaseWorld, World
from silenttwin.worlds.finite_state import FiniteStateWorld
from silenttwin.worlds.paired_world import PairedWorld, WorldPair, normalize_state
from silenttwin.worlds.dataset import (
    DATASET_REVISION,
    DATASET_SPLITS,
    DatasetValidationError,
    build_dataset_world_pair,
    default_template_id,
    get_template_spec,
    iter_template_specs,
    select_template_id,
    template_specs,
)
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
    *,
    instance_index: int | None = None,
    template_id: str | None = None,
    pair_family: str = "monitor_blind_spot",
    dataset_revision: str = DATASET_REVISION,
    dataset_split: str | None = None,
    **_: object,
) -> WorldPair:
    normalized = world_suite.strip().lower().replace("-", "_")
    if normalized not in WORLD_FACTORIES:
        choices = ", ".join(sorted(WORLD_FACTORIES))
        raise ValueError(f"unknown world suite {world_suite!r}; choose one of: {choices}")
    selected_template = template_id or default_template_id(normalized)
    spec = get_template_spec(selected_template)
    if spec.domain != normalized:
        raise DatasetValidationError(
            f"template {selected_template!r} belongs to {spec.domain!r}, not {normalized!r}"
        )
    return build_dataset_world_pair(
        selected_template,
        pair_family=pair_family,
        instance_id=instance_id,
        instance_index=instance_index,
        seed=seed,
        dataset_revision=dataset_revision,
        split=dataset_split,
    )


make_world_pair = build_world_pair
world_pair_for_suite = build_world_pair


__all__ = [
    "BaseWorld",
    "DATASET_REVISION",
    "DATASET_SPLITS",
    "DatasetValidationError",
    "FiniteStateWorld",
    "PairedWorld",
    "WORLD_FACTORIES",
    "World",
    "WorldPair",
    "build_dataset_world_pair",
    "build_world_pair",
    "default_template_id",
    "get_template_spec",
    "iter_template_specs",
    "make_world_pair",
    "normalize_state",
    "select_template_id",
    "template_specs",
    "world_pair_for_suite",
]
