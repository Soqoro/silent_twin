"""Versioned Tier-1 workflow registry and strict split validation.

The split is a property of a structural template, never of a generated seed,
state, feedback condition, or runtime.  This module is dependency-free and
keeps the checked-in JSON manifests auditable against the executable registry.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from silenttwin.schemas import stable_digest
from silenttwin.worlds.paired_world import WorldPair
from silenttwin.worlds.templates._common import (
    DATASET_REVISION,
    DATASET_SPLITS,
    MONITOR_BLIND_SPOT,
    PAIR_FAMILIES,
    TemplateSpec,
    build_registered_template_pair,
    instance_variation_manifest_record,
    normalize_pair_family,
)


DATASET_MANIFEST_SCHEMA = "silenttwin.dataset.v1"
SPLIT_MANIFEST_SCHEMA = "silenttwin.dataset_split.v1"
EXPECTED_TEMPLATE_COUNT = 16
EXPECTED_DOMAINS = ("calendar", "email", "files", "payments")


class DatasetValidationError(ValueError):
    """The executable registry and declared dataset contract disagree."""


@lru_cache(maxsize=1)
def template_specs() -> tuple[TemplateSpec, ...]:
    # Imports stay lazy so domain modules can import the shared dataclasses
    # without a registry initialization cycle.
    from silenttwin.worlds.templates.calendar import TEMPLATE_SPECS as calendar
    from silenttwin.worlds.templates.email import TEMPLATE_SPECS as email
    from silenttwin.worlds.templates.files import TEMPLATE_SPECS as files
    from silenttwin.worlds.templates.payments import TEMPLATE_SPECS as payments

    specs = tuple(sorted((*calendar, *email, *files, *payments), key=lambda item: item.template_id))
    validate_template_registry(specs)
    return specs


def validate_template_registry(specs: Sequence[TemplateSpec]) -> None:
    if len(specs) != EXPECTED_TEMPLATE_COUNT:
        raise DatasetValidationError(
            f"expected {EXPECTED_TEMPLATE_COUNT} structural templates, found {len(specs)}"
        )
    template_ids = [item.template_id for item in specs]
    if len(set(template_ids)) != len(template_ids):
        raise DatasetValidationError("template IDs must be unique")
    monitor_ids = [item.monitor_configuration_id for item in specs]
    if len(set(monitor_ids)) != len(monitor_ids):
        raise DatasetValidationError("monitor configuration IDs must be template-unique")
    signatures = [item.structural_signature for item in specs]
    if len(set(signatures)) != len(signatures):
        raise DatasetValidationError(
            "nominal template duplication detected: structural signatures must be unique"
        )
    domains = {item.domain for item in specs}
    if domains != set(EXPECTED_DOMAINS):
        raise DatasetValidationError(
            f"dataset domains {sorted(domains)} do not match {list(EXPECTED_DOMAINS)}"
        )
    for domain in EXPECTED_DOMAINS:
        members = [item for item in specs if item.domain == domain]
        if len(members) != 4:
            raise DatasetValidationError(
                f"domain {domain!r} must contain four structural templates"
            )
        missing_splits = set(DATASET_SPLITS) - {item.split for item in members}
        if missing_splits:
            raise DatasetValidationError(
                f"domain {domain!r} is absent from splits {sorted(missing_splits)}"
            )
    split_sizes = {
        split: sum(item.split == split for item in specs) for split in DATASET_SPLITS
    }
    if split_sizes != {"train": 8, "development": 4, "test": 4}:
        raise DatasetValidationError(f"unexpected template split sizes: {split_sizes}")
    for item in specs:
        if set(item.pair_families) != set(PAIR_FAMILIES):
            raise DatasetValidationError(
                f"template {item.template_id!r} does not keep both pair variants together"
            )


def get_template_spec(template_id: str) -> TemplateSpec:
    matches = [item for item in template_specs() if item.template_id == template_id]
    if not matches:
        choices = ", ".join(item.template_id for item in template_specs())
        raise DatasetValidationError(
            f"unknown template {template_id!r}; choose one of: {choices}"
        )
    return matches[0]


def iter_template_specs(
    *, domain: str | None = None, split: str | None = None
) -> tuple[TemplateSpec, ...]:
    if domain is not None and domain not in EXPECTED_DOMAINS:
        raise DatasetValidationError(f"unknown dataset domain {domain!r}")
    if split is not None and split not in DATASET_SPLITS:
        raise DatasetValidationError(f"unknown dataset split {split!r}")
    return tuple(
        item
        for item in template_specs()
        if (domain is None or item.domain == domain)
        and (split is None or item.split == split)
    )


def default_template_id(domain: str) -> str:
    members = iter_template_specs(domain=domain, split="train")
    if not members:
        raise DatasetValidationError(f"domain {domain!r} has no training template")
    return members[0].template_id


def select_template_id(
    domain: str,
    split: str,
    sample_index: int,
    requested: str | None = None,
    dataset_revision: str = DATASET_REVISION,
) -> str:
    """Select a template deterministically without permitting split leakage."""

    if dataset_revision != DATASET_REVISION:
        raise DatasetValidationError(
            f"dataset revision {dataset_revision!r} != supported {DATASET_REVISION!r}"
        )
    if not isinstance(sample_index, int) or isinstance(sample_index, bool) or sample_index < 0:
        raise DatasetValidationError("sample_index must be a non-negative integer")
    candidates = iter_template_specs(domain=domain, split=split)
    if requested is not None:
        spec = get_template_spec(requested)
        if spec.domain != domain:
            raise DatasetValidationError(
                f"requested template {requested!r} belongs to domain {spec.domain!r}, not {domain!r}"
            )
        if spec.split != split:
            raise DatasetValidationError(
                f"requested template {requested!r} belongs to split {spec.split!r}, not {split!r}"
            )
        return spec.template_id
    if not candidates:
        raise DatasetValidationError(
            f"no templates are registered for domain={domain!r}, split={split!r}"
        )
    return candidates[sample_index % len(candidates)].template_id


def build_dataset_world_pair(
    template_id: str,
    pair_family: str = MONITOR_BLIND_SPOT,
    instance_id: str | int | None = None,
    seed: int = 0,
    *,
    instance_index: int | None = None,
    dataset_revision: str = DATASET_REVISION,
    split: str | None = None,
) -> WorldPair:
    if dataset_revision != DATASET_REVISION:
        raise DatasetValidationError(
            f"dataset revision {dataset_revision!r} != supported {DATASET_REVISION!r}"
        )
    spec = get_template_spec(template_id)
    if split is not None and spec.split != split:
        raise DatasetValidationError(
            f"template {template_id!r} belongs to split {spec.split!r}, not {split!r}"
        )
    return build_registered_template_pair(
        spec,
        pair_family=normalize_pair_family(pair_family),
        instance_id=instance_id,
        instance_index=instance_index,
        seed=seed,
    )


def template_manifest_record(spec: TemplateSpec) -> dict[str, Any]:
    return {
        "template_id": spec.template_id,
        "domain": spec.domain,
        "split": spec.split,
        "structural_family": spec.structural_family,
        "structural_signature": spec.structural_signature,
        "monitor_configuration_id": spec.monitor_configuration_id,
        "pair_families": list(spec.pair_families),
    }


def expected_dataset_manifest() -> dict[str, Any]:
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "dataset_revision": DATASET_REVISION,
        "template_count": EXPECTED_TEMPLATE_COUNT,
        "domains": list(EXPECTED_DOMAINS),
        "instance_variation": instance_variation_manifest_record(),
        "split_strategy": {
            "unit": "structural_template",
            "counts": {split: len(iter_template_specs(split=split)) for split in DATASET_SPLITS},
            "domain_stratified": True,
            "all_pair_variants_remain_together": True,
            "monitor_configurations_disjoint": True,
        },
        "templates": [template_manifest_record(item) for item in template_specs()],
    }


def expected_split_manifest(split: str) -> dict[str, Any]:
    members = iter_template_specs(split=split)
    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "template_ids": [item.template_id for item in members],
        "monitor_configuration_ids": [
            item.monitor_configuration_id for item in members
        ],
    }


def _require_exact_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    if dict(actual) != dict(expected):
        actual_hash = stable_digest(actual)
        expected_hash = stable_digest(expected)
        raise DatasetValidationError(
            f"{label} does not exactly match executable registry "
            f"(actual={actual_hash}, expected={expected_hash})"
        )


def validate_dataset_manifest(manifest: Mapping[str, Any]) -> str:
    _require_exact_mapping(manifest, expected_dataset_manifest(), label="dataset manifest")
    return stable_digest(manifest)


def validate_split_manifests(manifests: Mapping[str, Mapping[str, Any]]) -> str:
    if set(manifests) != set(DATASET_SPLITS):
        raise DatasetValidationError(
            f"split manifests must contain exactly {list(DATASET_SPLITS)}"
        )
    seen_templates: set[str] = set()
    monitor_sets: dict[str, set[str]] = {}
    for split in DATASET_SPLITS:
        manifest = manifests[split]
        _require_exact_mapping(
            manifest, expected_split_manifest(split), label=f"{split} split manifest"
        )
        template_ids = set(str(item) for item in manifest["template_ids"])
        overlap = seen_templates & template_ids
        if overlap:
            raise DatasetValidationError(
                f"templates leak across split manifests: {sorted(overlap)}"
            )
        seen_templates.update(template_ids)
        monitor_sets[split] = set(
            str(item) for item in manifest["monitor_configuration_ids"]
        )
    expected_templates = {item.template_id for item in template_specs()}
    if seen_templates != expected_templates:
        raise DatasetValidationError("split manifests are not an exhaustive partition")
    for index, left in enumerate(DATASET_SPLITS):
        for right in DATASET_SPLITS[index + 1 :]:
            overlap = monitor_sets[left] & monitor_sets[right]
            if overlap:
                raise DatasetValidationError(
                    f"monitor configurations leak between {left} and {right}: {sorted(overlap)}"
                )
    return stable_digest({split: manifests[split] for split in DATASET_SPLITS})


def validate_train_evaluation_separation(
    training_template_ids: Iterable[str], evaluation_template_ids: Iterable[str]
) -> None:
    training = {get_template_spec(item).template_id for item in training_template_ids}
    evaluation = {get_template_spec(item).template_id for item in evaluation_template_ids}
    overlap = training & evaluation
    if overlap:
        raise DatasetValidationError(
            f"training and evaluation reuse templates: {sorted(overlap)}"
        )
    training_monitors = {
        get_template_spec(item).monitor_configuration_id for item in training
    }
    evaluation_monitors = {
        get_template_spec(item).monitor_configuration_id for item in evaluation
    }
    monitor_overlap = training_monitors & evaluation_monitors
    if monitor_overlap:
        raise DatasetValidationError(
            f"training and evaluation reuse monitor configurations: {sorted(monitor_overlap)}"
        )


def validate_records_respect_split(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_split: str,
    dataset_revision: str = DATASET_REVISION,
) -> None:
    if expected_split not in DATASET_SPLITS:
        raise DatasetValidationError(f"unknown dataset split {expected_split!r}")
    if dataset_revision != DATASET_REVISION:
        raise DatasetValidationError("unsupported dataset revision")
    for index, record in enumerate(records):
        template_id = record.get("template_id")
        if not isinstance(template_id, str):
            raise DatasetValidationError(f"record {index} lacks template_id")
        spec = get_template_spec(template_id)
        if spec.split != expected_split:
            raise DatasetValidationError(
                f"record {index} template {template_id!r} leaks from split {spec.split!r}"
            )
        if record.get("dataset_split") != expected_split:
            raise DatasetValidationError(f"record {index} has incorrect dataset_split")
        if record.get("dataset_revision") != DATASET_REVISION:
            raise DatasetValidationError(f"record {index} has incorrect dataset_revision")


def _default_config_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "silenttwin"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"missing dataset manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"invalid JSON in dataset manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(f"dataset manifest is not an object: {path}")
    return value


def load_and_validate_dataset_manifests(
    config_root: Path | str | None = None,
) -> dict[str, str]:
    root = Path(config_root) if config_root is not None else _default_config_root()
    dataset_manifest = _load_json_object(root / "dataset-v1.json")
    split_manifests = {
        split: _load_json_object(root / "splits" / f"{split}.json")
        for split in DATASET_SPLITS
    }
    return {
        "dataset_manifest_hash": validate_dataset_manifest(dataset_manifest),
        "split_manifest_hash": validate_split_manifests(split_manifests),
    }


__all__ = [
    "DATASET_MANIFEST_SCHEMA",
    "DATASET_REVISION",
    "DATASET_SPLITS",
    "EXPECTED_DOMAINS",
    "EXPECTED_TEMPLATE_COUNT",
    "PAIR_FAMILIES",
    "SPLIT_MANIFEST_SCHEMA",
    "DatasetValidationError",
    "build_dataset_world_pair",
    "default_template_id",
    "expected_dataset_manifest",
    "expected_split_manifest",
    "get_template_spec",
    "iter_template_specs",
    "load_and_validate_dataset_manifests",
    "select_template_id",
    "template_manifest_record",
    "template_specs",
    "validate_dataset_manifest",
    "validate_records_respect_split",
    "validate_split_manifests",
    "validate_template_registry",
    "validate_train_evaluation_separation",
]
