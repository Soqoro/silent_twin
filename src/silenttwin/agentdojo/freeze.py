"""Hash-bound AgentDojo development-evidence and held-out freeze records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from silenttwin.metrics.power import find_required_sample_size

from .config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
    AgentDojoConfigError,
    require_hash,
    stable_hash,
)


AGENTDOJO_FREEZE_SCHEMA = "silenttwin.agentdojo.sample_size_freeze.v1"
AGENTDOJO_UPSTREAM_SCHEMA = "silenttwin.agentdojo.upstream_bindings.v1"
AGENTDOJO_POWER_SPEC_SCHEMA = "silenttwin.agentdojo.power_spec.v1"
AGENTDOJO_POWER_EVIDENCE_SCHEMA = "silenttwin.agentdojo.development_power.v1"
AGENTDOJO_POWER_TARGET_MINIMUM = 0.80
AGENTDOJO_TEST_SELECTION_ALGORITHM = (
    "canonical-suite-round-robin-then-canonical-structural-group-v1"
)


@dataclass(frozen=True, slots=True)
class UpstreamBindings:
    catalog_hash: str
    scenario_registry_revision: str
    scenario_registry_hash: str
    split_manifest_hash: str
    candidate_strategy_catalog_hash: str
    pair_registry_hash: str
    analysis_plan_hash: str
    dependency_lock_hash: str
    package_version: str = AGENTDOJO_PACKAGE_VERSION
    source_revision: str = AGENTDOJO_SOURCE_REVISION
    benchmark_version: str = AGENTDOJO_BENCHMARK_VERSION
    schema_version: str = AGENTDOJO_UPSTREAM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != AGENTDOJO_UPSTREAM_SCHEMA:
            raise AgentDojoConfigError("unsupported upstream-binding schema")
        if self.package_version != AGENTDOJO_PACKAGE_VERSION:
            raise AgentDojoConfigError("freeze package version is not agentdojo==0.1.35")
        if self.source_revision != AGENTDOJO_SOURCE_REVISION:
            raise AgentDojoConfigError("freeze source revision is not the pinned release")
        if self.benchmark_version != AGENTDOJO_BENCHMARK_VERSION:
            raise AgentDojoConfigError("freeze benchmark version is not v1.2.2")
        if not self.scenario_registry_revision:
            raise AgentDojoConfigError("scenario registry revision must be non-empty")
        for name in (
            "catalog_hash",
            "scenario_registry_hash",
            "split_manifest_hash",
            "candidate_strategy_catalog_hash",
            "pair_registry_hash",
            "analysis_plan_hash",
            "dependency_lock_hash",
        ):
            require_hash(name, str(getattr(self, name)))

    @property
    def binding_hash(self) -> str:
        return stable_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "binding_hash": self.binding_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UpstreamBindings":
        data = dict(value)
        recorded = data.pop("binding_hash", None)
        result = cls(**data)
        if recorded is not None and recorded != result.binding_hash:
            raise AgentDojoConfigError("upstream binding hash is invalid")
        return result


def _validate_suite_mapping(
    name: str,
    value: Mapping[str, Any],
    *,
    positive_integers: bool = False,
    hashes: bool = False,
) -> dict[str, Any]:
    if set(value) != set(AGENTDOJO_SUITES):
        raise AgentDojoConfigError(
            f"{name} must contain exactly the four AgentDojo suites"
        )
    result: dict[str, Any] = {}
    for suite in AGENTDOJO_SUITES:
        item = value[suite]
        if positive_integers:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise AgentDojoConfigError(f"{name}.{suite} must be a positive integer")
        if hashes:
            require_hash(f"{name}.{suite}", str(item))
        result[suite] = item
    return result


def validate_power_analysis_spec(
    value: Mapping[str, Any], *, experiment_id: str
) -> dict[str, Any]:
    """Validate the preregistered paired-binary power protocol."""

    spec = dict(value)
    if spec.get("schema_version") != AGENTDOJO_POWER_SPEC_SCHEMA:
        raise AgentDojoConfigError("unsupported AgentDojo power-analysis spec")
    target_power = spec.get("target_power")
    alpha = spec.get("alpha")
    simulations = spec.get("simulations")
    seed = spec.get("seed")
    if (
        isinstance(target_power, bool)
        or not isinstance(target_power, (int, float))
        or not AGENTDOJO_POWER_TARGET_MINIMUM <= float(target_power) < 1.0
    ):
        raise AgentDojoConfigError("power target must be at least 0.80 and below one")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise AgentDojoConfigError("power alpha must lie strictly inside (0,1)")
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
        raise AgentDojoConfigError("power simulations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AgentDojoConfigError("power seed must be a non-negative integer")
    candidates = spec.get("candidate_total_independent_unit_counts")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in candidates)
        or candidates != sorted(set(candidates))
    ):
        raise AgentDojoConfigError(
            "power candidate independent-unit counts must be sorted unique positive integers"
        )
    if candidates[0] < len(AGENTDOJO_SUITES):
        raise AgentDojoConfigError("power candidates must permit every suite stratum")
    for name in (
        "minimum_structural_groups_per_suite",
        "preferred_structural_groups_per_suite",
    ):
        item = spec.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise AgentDojoConfigError(f"{name} must be a positive integer")
    if int(spec["preferred_structural_groups_per_suite"]) < int(
        spec["minimum_structural_groups_per_suite"]
    ):
        raise AgentDojoConfigError("preferred suite size cannot be below the minimum")
    if candidates[0] < int(spec["minimum_structural_groups_per_suite"]) * len(
        AGENTDOJO_SUITES
    ):
        raise AgentDojoConfigError(
            "power candidates cannot undercut the structural minimum across suites"
        )
    if spec.get("binary_reduction") != (
        "mean_at_least_half_after_structural_group_nested_row_averaging"
    ):
        raise AgentDojoConfigError("unsupported power binary reduction")
    experiments = spec.get("experiments")
    experiment = experiments.get(experiment_id) if isinstance(experiments, Mapping) else None
    if not isinstance(experiment, Mapping):
        raise AgentDojoConfigError(
            f"power-analysis spec lacks experiment {experiment_id!r}"
        )
    metric = experiment.get("metric")
    fields = experiment.get("condition_fields")
    target = experiment.get("target")
    reference = experiment.get("reference")
    effect = experiment.get("minimum_detectable_absolute_effect")
    if not isinstance(metric, str) or not metric:
        raise AgentDojoConfigError("power estimand metric must be non-empty")
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(item, str) or not item for item in fields)
        or len(set(fields)) != len(fields)
    ):
        raise AgentDojoConfigError("power estimand condition fields are invalid")
    if not isinstance(target, Mapping) or set(target) != set(fields):
        raise AgentDojoConfigError("power target selector does not match condition fields")
    if not isinstance(reference, Mapping) or set(reference) != set(fields):
        raise AgentDojoConfigError("power reference selector does not match condition fields")
    if target == reference:
        raise AgentDojoConfigError("power target and reference selectors must differ")
    if (
        isinstance(effect, bool)
        or not isinstance(effect, (int, float))
        or not 0.0 < float(effect) <= 1.0
    ):
        raise AgentDojoConfigError(
            "minimum detectable absolute effect must lie in (0,1]"
        )
    return spec


def make_development_power_evidence(
    *,
    experiment_id: str,
    primary_contrast_id: str,
    development_evidence_hash: str,
    power_analysis_spec: Mapping[str, Any],
    paired_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create recomputable prospective power evidence from development pairs."""

    if experiment_id not in {"e1", "e2", "e3", "e4"}:
        raise AgentDojoConfigError("paired held-out power is defined only for E1-E4")
    if not primary_contrast_id.startswith(f"{experiment_id}_"):
        raise AgentDojoConfigError("power contrast belongs to another experiment")
    require_hash("development_evidence_hash", development_evidence_hash)
    spec = validate_power_analysis_spec(power_analysis_spec, experiment_id=experiment_id)
    normalized_pairs: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in paired_outcomes:
        suite = row.get("agentdojo_suite")
        group = row.get("structural_group_id")
        target_value = row.get("target")
        reference_value = row.get("reference")
        if (
            suite not in AGENTDOJO_SUITES
            or not isinstance(group, str)
            or not group
            or target_value not in {0, 1, False, True}
            or reference_value not in {0, 1, False, True}
        ):
            raise AgentDojoConfigError("development power has an invalid paired unit")
        identity = (str(suite), group)
        if identity in identities:
            raise AgentDojoConfigError("development power repeats an independent unit")
        identities.add(identity)
        normalized_pairs.append(
            {
                "agentdojo_suite": str(suite),
                "structural_group_id": group,
                "target": int(target_value),
                "reference": int(reference_value),
            }
        )
    normalized_pairs.sort(
        key=lambda row: (
            AGENTDOJO_SUITES.index(str(row["agentdojo_suite"])),
            str(row["structural_group_id"]),
        )
    )
    target = tuple(int(row["target"]) for row in normalized_pairs)
    reference = tuple(int(row["reference"]) for row in normalized_pairs)
    if not target:
        raise AgentDojoConfigError("development power requires non-empty paired outcomes")
    target_only = sum(left == 1 and right == 0 for left, right in zip(target, reference))
    reference_only = sum(left == 0 and right == 1 for left, right in zip(target, reference))
    both = sum(left == 1 and right == 1 for left, right in zip(target, reference))
    neither = len(target) - target_only - reference_only - both
    observed_effect = (target_only - reference_only) / len(target)
    observed_discordance = (target_only + reference_only) / len(target)
    experiment_spec = dict(spec["experiments"][experiment_id])
    design_effect = float(experiment_spec["minimum_detectable_absolute_effect"])
    # A paired-binary model cannot have |effect| > discordance.  A zero or
    # unusually low pilot discordance therefore uses the smallest coherent
    # prospective discordance, stated explicitly rather than silently clipping.
    design_discordance = max(observed_discordance, design_effect)
    required = find_required_sample_size(
        spec["candidate_total_independent_unit_counts"],
        effect=design_effect,
        discordance=design_discordance,
        target_power=float(spec["target_power"]),
        alpha=float(spec["alpha"]),
        simulations=int(spec["simulations"]),
        seed=int(spec["seed"]),
    ).to_dict()
    payload = {
        "schema_version": AGENTDOJO_POWER_EVIDENCE_SCHEMA,
        "status": "estimated_not_frozen",
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "experiment_id": experiment_id,
        "primary_contrast_id": primary_contrast_id,
        "independent_unit": "structural_group_id",
        "development_evidence_hash": development_evidence_hash,
        "power_analysis_spec": spec,
        "power_analysis_spec_hash": stable_hash(spec),
        "estimand": experiment_spec,
        "binary_reduction": spec["binary_reduction"],
        "paired_outcomes": normalized_pairs,
        "paired_outcomes_hash": stable_hash(normalized_pairs),
        "observed_pair_count": len(target),
        "discordance_counts": {
            "target_only": target_only,
            "reference_only": reference_only,
            "both": both,
            "neither": neither,
        },
        "observed_effect": observed_effect,
        "observed_discordance": observed_discordance,
        "design_effect": design_effect,
        "design_discordance": design_discordance,
        "discordance_floor_policy": "max(observed_discordance,design_effect)",
        "required_sample_size": required,
        "recommendation_status": (
            "target_power_reached_within_candidates"
            if required["achieved"]
            else "target_power_not_reached_within_candidates"
        ),
    }
    return {**payload, "power_evidence_hash": stable_hash(payload)}


def validate_development_power_evidence(
    evidence: Mapping[str, Any],
    *,
    experiment_id: str,
    primary_contrast_id: str,
    development_evidence_hash: str,
) -> dict[str, Any]:
    """Reject unbound or internally invented power mappings."""

    payload = dict(evidence)
    recorded_hash = payload.pop("power_evidence_hash", None)
    if recorded_hash != stable_hash(payload):
        raise AgentDojoConfigError("development power-evidence hash is invalid")
    expected = {
        "schema_version": AGENTDOJO_POWER_EVIDENCE_SCHEMA,
        "status": "estimated_not_frozen",
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "experiment_id": experiment_id,
        "primary_contrast_id": primary_contrast_id,
        "independent_unit": "structural_group_id",
        "development_evidence_hash": development_evidence_hash,
        "discordance_floor_policy": "max(observed_discordance,design_effect)",
    }
    for name, expected_value in expected.items():
        if payload.get(name) != expected_value:
            raise AgentDojoConfigError(f"development power evidence has incompatible {name}")
    spec_raw = payload.get("power_analysis_spec")
    if not isinstance(spec_raw, Mapping):
        raise AgentDojoConfigError("development power evidence lacks its preregistered spec")
    spec = validate_power_analysis_spec(spec_raw, experiment_id=experiment_id)
    if payload.get("power_analysis_spec_hash") != stable_hash(spec):
        raise AgentDojoConfigError("development power spec hash is invalid")
    experiment_spec = dict(spec["experiments"][experiment_id])
    if payload.get("estimand") != experiment_spec or payload.get(
        "binary_reduction"
    ) != spec["binary_reduction"]:
        raise AgentDojoConfigError("development power estimand is not preregistered")
    paired_raw = payload.get("paired_outcomes")
    if not isinstance(paired_raw, list):
        raise AgentDojoConfigError("development power lacks paired independent units")
    normalized_pairs: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in paired_raw:
        if not isinstance(row, Mapping) or set(row) != {
            "agentdojo_suite",
            "structural_group_id",
            "target",
            "reference",
        }:
            raise AgentDojoConfigError("development power has a malformed paired unit")
        suite = row["agentdojo_suite"]
        group = row["structural_group_id"]
        target_value = row["target"]
        reference_value = row["reference"]
        if (
            suite not in AGENTDOJO_SUITES
            or not isinstance(group, str)
            or not group
            or isinstance(target_value, bool)
            or not isinstance(target_value, int)
            or target_value not in {0, 1}
            or isinstance(reference_value, bool)
            or not isinstance(reference_value, int)
            or reference_value not in {0, 1}
        ):
            raise AgentDojoConfigError("development power has an invalid paired unit")
        identity = (str(suite), group)
        if identity in identities:
            raise AgentDojoConfigError("development power repeats an independent unit")
        identities.add(identity)
        normalized_pairs.append(dict(row))
    expected_order = sorted(
        normalized_pairs,
        key=lambda row: (
            AGENTDOJO_SUITES.index(str(row["agentdojo_suite"])),
            str(row["structural_group_id"]),
        ),
    )
    if normalized_pairs != expected_order or payload.get(
        "paired_outcomes_hash"
    ) != stable_hash(normalized_pairs):
        raise AgentDojoConfigError("development power paired-outcome binding is invalid")
    if {str(row["agentdojo_suite"]) for row in normalized_pairs} != set(
        AGENTDOJO_SUITES
    ):
        raise AgentDojoConfigError(
            "development power requires paired outcomes from all four suites"
        )
    count = payload.get("observed_pair_count")
    counts = payload.get("discordance_counts")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise AgentDojoConfigError("development power has no positive paired count")
    if count != len(normalized_pairs):
        raise AgentDojoConfigError("development power paired count is inconsistent")
    if not isinstance(counts, Mapping) or set(counts) != {
        "target_only",
        "reference_only",
        "both",
        "neither",
    }:
        raise AgentDojoConfigError("development power discordance table is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ) or sum(int(value) for value in counts.values()) != count:
        raise AgentDojoConfigError("development power discordance counts are inconsistent")
    recomputed_counts = {
        "target_only": sum(
            row["target"] == 1 and row["reference"] == 0 for row in normalized_pairs
        ),
        "reference_only": sum(
            row["target"] == 0 and row["reference"] == 1 for row in normalized_pairs
        ),
        "both": sum(
            row["target"] == 1 and row["reference"] == 1 for row in normalized_pairs
        ),
        "neither": sum(
            row["target"] == 0 and row["reference"] == 0 for row in normalized_pairs
        ),
    }
    if dict(counts) != recomputed_counts:
        raise AgentDojoConfigError("development power discordance table is not data-derived")
    target_only = recomputed_counts["target_only"]
    reference_only = recomputed_counts["reference_only"]
    observed_effect = (target_only - reference_only) / count
    observed_discordance = (target_only + reference_only) / count
    design_effect = float(experiment_spec["minimum_detectable_absolute_effect"])
    design_discordance = max(observed_discordance, design_effect)
    for name, expected_value in (
        ("observed_effect", observed_effect),
        ("observed_discordance", observed_discordance),
        ("design_effect", design_effect),
        ("design_discordance", design_discordance),
    ):
        raw = payload.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isclose(float(raw), expected_value, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise AgentDojoConfigError(f"development power {name} is inconsistent")
    recomputed = find_required_sample_size(
        spec["candidate_total_independent_unit_counts"],
        effect=design_effect,
        discordance=design_discordance,
        target_power=float(spec["target_power"]),
        alpha=float(spec["alpha"]),
        simulations=int(spec["simulations"]),
        seed=int(spec["seed"]),
    ).to_dict()
    if payload.get("required_sample_size") != recomputed:
        raise AgentDojoConfigError("development power recommendation is not reproducible")
    expected_status = (
        "target_power_reached_within_candidates"
        if recomputed["achieved"]
        else "target_power_not_reached_within_candidates"
    )
    if payload.get("recommendation_status") != expected_status:
        raise AgentDojoConfigError("development power disposition is inconsistent")
    return {**payload, "power_evidence_hash": recorded_hash}


def validate_development_analysis_manifest(
    manifest: Mapping[str, Any],
    *,
    experiment_id: str,
    primary_contrast_id: str,
    upstream: UpstreamBindings,
) -> dict[str, Any]:
    """Validate the exact self-hashed development artifact consumed by a freeze."""

    document = dict(manifest)
    recorded_hash = document.pop("analysis_manifest_hash", None)
    require_hash("development_analysis_manifest_hash", str(recorded_hash))
    if recorded_hash != stable_hash(document):
        raise AgentDojoConfigError("development analysis-manifest hash is invalid")
    expected = {
        "schema_version": "silenttwin.agentdojo.analysis_manifest.v1",
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "experiment_id": experiment_id,
        "dataset_split": "development",
        "analysis_plan_hash": upstream.analysis_plan_hash,
        "upstream_chain_hash": upstream.binding_hash,
        "grid_validation_mode": "exact_expected_grid",
        "suite_coverage_status": "full_four_suite",
        "confirmatory_suite_coverage_eligible": True,
        "sample_size_freeze_eligible": True,
        "suite_strata": list(AGENTDOJO_SUITES),
        "fixture_mode": False,
        "scientific_evidence_eligible": True,
    }
    for name, expected_value in expected.items():
        if document.get(name) != expected_value:
            raise AgentDojoConfigError(
                f"development analysis manifest has incompatible {name}"
            )
    evidence_hash = str(document.get("development_evidence_hash"))
    require_hash("development_evidence_hash", evidence_hash)
    digest_payload = document.get("current_evidence_digest_payload")
    if not isinstance(digest_payload, Mapping):
        raise AgentDojoConfigError(
            "development analysis manifest lacks its evidence digest payload"
        )
    if (
        document.get("current_evidence_hash") != stable_hash(digest_payload)
        or document.get("current_evidence_hash") != evidence_hash
    ):
        raise AgentDojoConfigError(
            "development analysis current/development evidence hashes disagree"
        )
    power_raw = document.get("development_power_analysis")
    if not isinstance(power_raw, Mapping):
        raise AgentDojoConfigError(
            "development analysis manifest lacks power evidence"
        )
    power = validate_development_power_evidence(
        power_raw,
        experiment_id=experiment_id,
        primary_contrast_id=primary_contrast_id,
        development_evidence_hash=evidence_hash,
    )
    if digest_payload.get("paired_power_outcomes") != power["paired_outcomes"]:
        raise AgentDojoConfigError(
            "development power evidence is not the exact analysis digest cohort"
        )
    suite_counts = document.get("suite_independent_unit_counts")
    expected_counts = {
        suite: sum(
            row["agentdojo_suite"] == suite for row in power["paired_outcomes"]
        )
        for suite in AGENTDOJO_SUITES
    }
    if suite_counts != expected_counts or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in expected_counts.values()
    ):
        raise AgentDojoConfigError(
            "development analysis suite counts do not match its power cohort"
        )
    return {
        **document,
        "analysis_manifest_hash": recorded_hash,
        "development_power_analysis": power,
    }


def deterministic_test_allocation(
    available_by_suite: Mapping[str, int], *, requested_total: int
) -> dict[str, int]:
    """Allocate a total N in pinned suite order without duplicating groups."""

    available = _validate_suite_mapping(
        "available_test_independent_unit_count_by_suite",
        available_by_suite,
        positive_integers=True,
    )
    if (
        isinstance(requested_total, bool)
        or not isinstance(requested_total, int)
        or requested_total < len(AGENTDOJO_SUITES)
        or requested_total > sum(int(value) for value in available.values())
    ):
        raise AgentDojoConfigError("requested held-out total is infeasible")
    selected = {suite: 0 for suite in AGENTDOJO_SUITES}
    remaining = requested_total
    while remaining:
        progressed = False
        for suite in AGENTDOJO_SUITES:
            if remaining == 0:
                break
            if selected[suite] < int(available[suite]):
                selected[suite] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # pragma: no cover - guarded by total validation
            raise AgentDojoConfigError("held-out allocation exhausted available groups")
    if any(value <= 0 for value in selected.values()):
        raise AgentDojoConfigError("held-out allocation omitted a suite stratum")
    return selected


def make_agentdojo_sample_size_freeze(
    *,
    experiment_id: str,
    primary_contrast_id: str,
    upstream: UpstreamBindings,
    development_analysis_manifest_hash: str,
    development_evidence_hash: str,
    independent_unit_count_by_suite: Mapping[str, int],
    available_test_independent_unit_count_by_suite: Mapping[str, int],
    selected_test_bundle_hash_by_suite: Mapping[str, str],
    power_evidence: Mapping[str, Any],
    selected_structural_group_ids_by_suite: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if experiment_id not in {"e1", "e2", "e3", "e4"}:
        raise AgentDojoConfigError("AgentDojo freeze experiment must be E1-E4")
    if not primary_contrast_id.startswith(f"{experiment_id}_"):
        raise AgentDojoConfigError("primary contrast belongs to another experiment")
    require_hash("development_analysis_manifest_hash", development_analysis_manifest_hash)
    require_hash("development_evidence_hash", development_evidence_hash)
    validated_power = validate_development_power_evidence(
        power_evidence,
        experiment_id=experiment_id,
        primary_contrast_id=primary_contrast_id,
        development_evidence_hash=development_evidence_hash,
    )
    available = _validate_suite_mapping(
        "available_test_independent_unit_count_by_suite",
        available_test_independent_unit_count_by_suite,
        positive_integers=True,
    )
    counts = _validate_suite_mapping(
        "independent_unit_count_by_suite",
        independent_unit_count_by_suite,
        positive_integers=True,
    )
    bundles = _validate_suite_mapping(
        "selected_test_bundle_hash_by_suite",
        selected_test_bundle_hash_by_suite,
        hashes=True,
    )
    if set(selected_structural_group_ids_by_suite) != set(AGENTDOJO_SUITES):
        raise AgentDojoConfigError(
            "selected structural groups must contain exactly four suites"
        )
    spec = validated_power["power_analysis_spec"]
    minimum_per_suite = int(spec["minimum_structural_groups_per_suite"])
    recommended = validated_power["required_sample_size"]["selected_sample_size"]
    available_total = sum(int(value) for value in available.values())
    confirmatory_feasible = (
        isinstance(recommended, int)
        and not isinstance(recommended, bool)
        and recommended >= minimum_per_suite * len(AGENTDOJO_SUITES)
        and recommended <= available_total
        and all(int(available[suite]) >= minimum_per_suite for suite in AGENTDOJO_SUITES)
    )
    selected_total = int(recommended) if confirmatory_feasible else available_total
    expected_counts = deterministic_test_allocation(
        available, requested_total=selected_total
    )
    if counts != expected_counts:
        raise AgentDojoConfigError(
            "suite sample sizes do not match the deterministic power allocation"
        )
    selected: dict[str, list[str]] = {}
    for suite in AGENTDOJO_SUITES:
        identifiers = list(selected_structural_group_ids_by_suite[suite])
        if (
            len(identifiers) != counts[suite]
            or len(set(identifiers)) != len(identifiers)
            or any(not isinstance(item, str) or not item for item in identifiers)
        ):
            raise AgentDojoConfigError(
                f"selected structural groups for {suite} do not match frozen N"
            )
        selected[suite] = identifiers
    claim_disposition = (
        "confirmatory_power_target_met"
        if confirmatory_feasible
        else "underpowered_estimation_only"
    )
    structural_shortfalls = {
        suite: {
            "available": int(available[suite]),
            "minimum": minimum_per_suite,
        }
        for suite in AGENTDOJO_SUITES
        if int(available[suite]) < minimum_per_suite
    }
    payload = {
        "schema_version": AGENTDOJO_FREEZE_SCHEMA,
        "status": "frozen",
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "experiment_id": experiment_id,
        "primary_contrast_id": primary_contrast_id,
        "independent_unit": "structural_group_id",
        "independent_unit_count_by_suite": counts,
        "available_test_independent_unit_count_by_suite": available,
        "selected_test_bundle_hash_by_suite": bundles,
        "selected_structural_group_ids_by_suite": selected,
        "upstream_bindings": upstream.to_dict(),
        "development_analysis_manifest_hash": development_analysis_manifest_hash,
        "development_evidence_hash": development_evidence_hash,
        "power_evidence": validated_power,
        "power_evidence_hash": validated_power["power_evidence_hash"],
        "selection_algorithm": AGENTDOJO_TEST_SELECTION_ALGORITHM,
        "selected_total_independent_unit_count": selected_total,
        "claim_disposition": claim_disposition,
        "structural_minimum_shortfalls": structural_shortfalls,
        "frozen_before_test": True,
        "test_results_inspected": False,
    }
    return {**payload, "freeze_hash": stable_hash(payload)}


def validate_agentdojo_sample_size_freeze(
    freeze: Mapping[str, Any],
    *,
    experiment_id: str,
    primary_contrast_id: str,
    upstream: UpstreamBindings,
    suite: str | None = None,
    development_analysis_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(freeze)
    recorded_hash = payload.pop("freeze_hash", None)
    if recorded_hash != stable_hash(payload):
        raise AgentDojoConfigError("AgentDojo sample-size freeze hash is invalid")
    if payload.get("schema_version") != AGENTDOJO_FREEZE_SCHEMA:
        raise AgentDojoConfigError("unsupported AgentDojo sample-size freeze schema")
    if payload.get("status") != "frozen" or payload.get("environment_backend") != "agentdojo":
        raise AgentDojoConfigError("AgentDojo held-out sample size is not frozen")
    if payload.get("tier2_track") != "controlled":
        raise AgentDojoConfigError("held-out causal freeze must use the controlled track")
    if payload.get("experiment_id") != experiment_id:
        raise AgentDojoConfigError("sample-size freeze belongs to another experiment")
    if payload.get("primary_contrast_id") != primary_contrast_id:
        raise AgentDojoConfigError("sample-size freeze uses another primary contrast")
    if payload.get("independent_unit") != "structural_group_id":
        raise AgentDojoConfigError("sample-size freeze uses the wrong independent unit")
    if payload.get("frozen_before_test") is not True or payload.get(
        "test_results_inspected"
    ) is not False:
        raise AgentDojoConfigError("sample-size freeze was not made before held-out inspection")
    recorded_upstream = payload.get("upstream_bindings")
    if not isinstance(recorded_upstream, Mapping):
        raise AgentDojoConfigError("sample-size freeze lacks upstream bindings")
    if UpstreamBindings.from_mapping(recorded_upstream) != upstream:
        raise AgentDojoConfigError("sample-size freeze uses a different upstream chain")
    counts_raw = payload.get("independent_unit_count_by_suite")
    available_raw = payload.get("available_test_independent_unit_count_by_suite")
    bundles_raw = payload.get("selected_test_bundle_hash_by_suite")
    if (
        not isinstance(counts_raw, Mapping)
        or not isinstance(available_raw, Mapping)
        or not isinstance(bundles_raw, Mapping)
    ):
        raise AgentDojoConfigError("sample-size freeze lacks suite-stratified N/bundles")
    counts = _validate_suite_mapping(
        "independent_unit_count_by_suite", counts_raw, positive_integers=True
    )
    bundles = _validate_suite_mapping(
        "selected_test_bundle_hash_by_suite", bundles_raw, hashes=True
    )
    available = _validate_suite_mapping(
        "available_test_independent_unit_count_by_suite",
        available_raw,
        positive_integers=True,
    )
    for name in ("development_analysis_manifest_hash", "development_evidence_hash"):
        require_hash(name, str(payload.get(name)))
    selected_raw = payload.get("selected_structural_group_ids_by_suite")
    selected: dict[str, list[str]] = {}
    if not isinstance(selected_raw, Mapping) or set(selected_raw) != set(
        AGENTDOJO_SUITES
    ):
        raise AgentDojoConfigError(
            "selected structural groups must contain exactly four suites"
        )
    for selected_suite in AGENTDOJO_SUITES:
        raw_ids = selected_raw[selected_suite]
        if not isinstance(raw_ids, list) or any(
            not isinstance(item, str) or not item for item in raw_ids
        ):
            raise AgentDojoConfigError(
                f"selected structural groups for {selected_suite} are invalid"
            )
        if len(raw_ids) != counts[selected_suite] or len(set(raw_ids)) != len(
            raw_ids
        ):
            raise AgentDojoConfigError(
                f"selected structural groups for {selected_suite} do not match frozen N"
            )
        selected[selected_suite] = list(raw_ids)
    power = payload.get("power_evidence")
    if not isinstance(power, Mapping):
        raise AgentDojoConfigError("sample-size freeze lacks development power evidence")
    validated_power = validate_development_power_evidence(
        power,
        experiment_id=experiment_id,
        primary_contrast_id=primary_contrast_id,
        development_evidence_hash=str(payload.get("development_evidence_hash")),
    )
    if payload.get("power_evidence_hash") != validated_power["power_evidence_hash"]:
        raise AgentDojoConfigError("sample-size freeze power-evidence binding is invalid")
    if payload.get("selection_algorithm") != AGENTDOJO_TEST_SELECTION_ALGORITHM:
        raise AgentDojoConfigError("sample-size freeze uses another selection algorithm")
    minimum_per_suite = int(
        validated_power["power_analysis_spec"]["minimum_structural_groups_per_suite"]
    )
    recommended = validated_power["required_sample_size"]["selected_sample_size"]
    available_total = sum(int(value) for value in available.values())
    confirmatory_feasible = (
        isinstance(recommended, int)
        and not isinstance(recommended, bool)
        and recommended >= minimum_per_suite * len(AGENTDOJO_SUITES)
        and recommended <= available_total
        and all(int(available[item]) >= minimum_per_suite for item in AGENTDOJO_SUITES)
    )
    selected_total = int(recommended) if confirmatory_feasible else available_total
    expected_counts = deterministic_test_allocation(
        available, requested_total=selected_total
    )
    if counts != expected_counts:
        raise AgentDojoConfigError(
            "sample-size freeze suite allocation is not power-derived"
        )
    expected_disposition = (
        "confirmatory_power_target_met"
        if confirmatory_feasible
        else "underpowered_estimation_only"
    )
    if payload.get("claim_disposition") != expected_disposition or payload.get(
        "selected_total_independent_unit_count"
    ) != selected_total:
        raise AgentDojoConfigError("sample-size freeze power disposition is inconsistent")
    expected_shortfalls = {
        item: {"available": int(available[item]), "minimum": minimum_per_suite}
        for item in AGENTDOJO_SUITES
        if int(available[item]) < minimum_per_suite
    }
    if payload.get("structural_minimum_shortfalls") != expected_shortfalls:
        raise AgentDojoConfigError("sample-size freeze structural shortfalls are inconsistent")
    if development_analysis_manifest is not None:
        development = validate_development_analysis_manifest(
            development_analysis_manifest,
            experiment_id=experiment_id,
            primary_contrast_id=primary_contrast_id,
            upstream=upstream,
        )
        if (
            development["analysis_manifest_hash"]
            != payload["development_analysis_manifest_hash"]
            or development["development_evidence_hash"]
            != payload["development_evidence_hash"]
            or development["development_power_analysis"]["power_evidence_hash"]
            != payload["power_evidence_hash"]
        ):
            raise AgentDojoConfigError(
                "sample-size freeze differs from its development analysis manifest"
            )
    result = {
        **payload,
        "freeze_hash": recorded_hash,
        "independent_unit_count_by_suite": counts,
        "available_test_independent_unit_count_by_suite": available,
        "selected_test_bundle_hash_by_suite": bundles,
        "selected_structural_group_ids_by_suite": selected,
    }
    if suite is not None:
        if suite not in AGENTDOJO_SUITES:
            raise AgentDojoConfigError(f"unknown AgentDojo suite {suite!r}")
        result["selected_suite"] = suite
        result["selected_independent_unit_count"] = counts[suite]
        result["selected_test_bundle_hash"] = bundles[suite]
    return result


__all__ = [
    "AGENTDOJO_FREEZE_SCHEMA",
    "AGENTDOJO_POWER_EVIDENCE_SCHEMA",
    "AGENTDOJO_POWER_SPEC_SCHEMA",
    "AGENTDOJO_POWER_TARGET_MINIMUM",
    "AGENTDOJO_TEST_SELECTION_ALGORITHM",
    "AGENTDOJO_UPSTREAM_SCHEMA",
    "UpstreamBindings",
    "deterministic_test_allocation",
    "make_agentdojo_sample_size_freeze",
    "make_development_power_evidence",
    "validate_agentdojo_sample_size_freeze",
    "validate_development_analysis_manifest",
    "validate_development_power_evidence",
    "validate_power_analysis_spec",
]
