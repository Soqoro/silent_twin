from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from silenttwin.agentdojo.action_eligibility import (
    ACTION_VALIDATION_SCHEMA_VERSION,
    pilot_scenario_ids,
)
from silenttwin.agentdojo.compat import (
    EXPECTED_ATTACKS,
    EXPECTED_DEFENSES,
    EXPECTED_INTERNAL_BENCHMARK_VERSIONS,
    EXPECTED_RELEASE_COUNTS,
    EXPECTED_WHEEL_SHA256,
)
from silenttwin.agentdojo.config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
    stable_hash,
)
from silenttwin.agentdojo.pair_mining import (
    OBSERVATION_SCHEMA_VERSION,
    PairMiningError,
    _best_construction,
    make_monitor_observation,
    make_observation_set_manifest,
    make_train_pair_design_audit,
    make_train_pair_feasibility_report,
    make_candidate_strategy_catalog,
    mine_pair_registry,
    monitor_pair_binding,
    validate_candidate_strategy_catalog,
    validate_estimation_strategy_coverage,
    validate_observation_set_manifest,
    validate_pair_registry,
)
from silenttwin.agentdojo.monitors import (
    GRANITE_GUARDIAN_ACTION_PROMPT_TEMPLATE,
    monitor_text_hash,
)
from silenttwin.agentdojo.runtime_integrity import (
    EXPECTED_INSTALLED_PAYLOAD_SHA256,
    LearnedRuntimeReport,
    learned_runtime_manifest_fingerprint,
    make_learned_runtime_provenance,
    not_applicable_learned_runtime_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _action_eligibility() -> dict[str, object]:
    return json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/action-eligibility-v1.json"
        ).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def frozen_inputs() -> tuple[dict[str, object], dict[str, object]]:
    catalog = json.loads(
        (REPO_ROOT / "configs/silenttwin/agentdojo/catalog-v1.json").read_text(
            encoding="utf-8"
        )
    )
    splits = json.loads(
        (REPO_ROOT / "configs/silenttwin/agentdojo/splits-v1.json").read_text(
            encoding="utf-8"
        )
    )
    return catalog, splits


@pytest.fixture(scope="module")
def strategy_catalog(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
) -> dict[str, object]:
    catalog, splits = frozen_inputs
    pilot_ids = sorted(
        {
            *pilot_scenario_ids(_action_eligibility(), dataset_split="train"),
            *pilot_scenario_ids(
                _action_eligibility(), dataset_split="development"
            ),
        }
    )
    return make_candidate_strategy_catalog(
        catalog_hash=str(catalog["catalog_hash"]),
        split_manifest_hash=str(splits["split_manifest_hash"]),
        transformation_family_revision="fixture-transformations-v1",
        train_evidence_hash=stable_hash({"fixture": "train-construction"}),
        strategies=(
            {
                "strategy_id": "strategy_a",
                "transformation_family": "family_a",
                "ground_truth_label": "prohibited",
                "frozen_on_split": "train",
                "default_plan_policy": "forbidden",
                "scenario_plans": {
                    scenario_id: {
                        "calls": [{"function": "candidate_a", "args": {}}]
                    }
                    for scenario_id in pilot_ids
                },
                "plan_template": {
                    "calls": [{"function": "candidate_a", "args": {}}]
                },
            },
            {
                "strategy_id": "strategy_b",
                "transformation_family": "family_b",
                "ground_truth_label": "prohibited",
                "frozen_on_split": "train",
                "default_plan_policy": "forbidden",
                "scenario_plans": {
                    scenario_id: {
                        "calls": [{"function": "candidate_b", "args": {}}]
                    }
                    for scenario_id in pilot_ids
                },
                "plan_template": {
                    "calls": [{"function": "candidate_b", "args": {}}]
                },
            },
        ),
        monitor_profiles=(
            {
                "profile_id": "profile_a",
                "profile_hash": stable_hash({"profile": "a"}),
                "frozen_on_split": "train",
            },
            {
                "profile_id": "profile_b",
                "profile_hash": stable_hash({"profile": "b"}),
                "frozen_on_split": "train",
            },
        ),
    )


def _eligible_scenarios(
    catalog: dict[str, object], split: str
) -> list[dict[str, object]]:
    allowed = set(
        pilot_scenario_ids(_action_eligibility(), dataset_split=split)
    )
    return [
        row
        for row in catalog["scenarios"]
        if row["scenario_id"] in allowed
    ]


def _action_validation(
    scenario: dict[str, object], strategy_id: str, call_hash: str
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ACTION_VALIDATION_SCHEMA_VERSION,
        "action_eligibility_manifest_hash": _action_eligibility()[
            "action_eligibility_manifest_hash"
        ],
        "scenario_id": scenario["scenario_id"],
        "suite": scenario["suite"],
        "dataset_split": scenario["dataset_split"],
        "strategy_id": strategy_id,
        "required_action_multiset_hash": stable_hash(
            [scenario["scenario_id"], strategy_id, "required"]
        ),
        "call_sequence_hash": call_hash,
        "initial_environment_hash": scenario["initial_environment_hash"],
        "post_environment_hash": stable_hash(
            [scenario["scenario_id"], strategy_id, "post"]
        ),
        "trace_hash": stable_hash([scenario["scenario_id"], strategy_id, "trace"]),
        "executed_call_count": 1,
        "execution_errors": [],
        "released_raw_security_result": True,
        "normalized_attack_success": True,
        "external_api_calls": 0,
    }
    return {**payload, "action_validation_hash": stable_hash(payload)}


def _observations(
    catalog: dict[str, object],
    splits: dict[str, object],
    strategy_catalog: dict[str, object],
    *,
    split: str,
    reverse: bool = False,
    all_block: bool = False,
) -> list[dict[str, object]]:
    scenarios = _eligible_scenarios(catalog, split)
    strategies = {
        str(row["strategy_id"]): row for row in strategy_catalog["strategies"]
    }
    forward = {
        ("profile_a", "strategy_a"): "allow",
        ("profile_b", "strategy_a"): "block",
        ("profile_a", "strategy_b"): "block",
        ("profile_b", "strategy_b"): "allow",
    }
    for strategy_id in sorted(set(strategies) - {"strategy_a", "strategy_b"}):
        forward[("profile_a", strategy_id)] = "block"
        forward[("profile_b", strategy_id)] = "block"
    if reverse:
        forward = {
            key: "block" if decision == "allow" else "allow"
            for key, decision in forward.items()
        }
    if all_block:
        forward = {key: "block" for key in forward}
    rows: list[dict[str, object]] = []
    profiles = {
        str(row["profile_id"]): row for row in strategy_catalog["monitor_profiles"]
    }
    for scenario in scenarios:
        suite = str(scenario["suite"])
        for (profile_id, strategy_id), decision in forward.items():
            call_hash = stable_hash([scenario["scenario_id"], strategy_id])
            action_validation = _action_validation(
                scenario, strategy_id, call_hash
            )
            rows.append(
                make_monitor_observation(
                    scenario=scenario,
                    strategy=strategies[strategy_id],
                    profile=profiles[profile_id],
                    catalog_hash=str(catalog["catalog_hash"]),
                    split_manifest_hash=str(splits["split_manifest_hash"]),
                    candidate_strategy_catalog_hash=str(
                        strategy_catalog["candidate_strategy_catalog_hash"]
                    ),
                    monitor_decision=decision,
                    monitor_score=0.9 if decision == "block" else 0.1,
                    call_sequence_hash=call_hash,
                    monitor_input_hashes=(stable_hash([call_hash, profile_id]),),
                    seed_schedule=(0,),
                    call_provenance=({"fixture": True},),
                    generator_source_tree_hash=stable_hash("fixture-source"),
                    action_eligibility_manifest_hash=str(
                        _action_eligibility()["action_eligibility_manifest_hash"]
                    ),
                    action_validation_hash=str(
                        action_validation["action_validation_hash"]
                    ),
                )
            )
    return rows


def _observation_manifest(
    rows: list[dict[str, object]],
    *,
    split: str,
    catalog: dict[str, object],
    splits: dict[str, object],
    strategy_catalog: dict[str, object],
    learned_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    validations: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        scenario = next(
            scenario
            for scenario in catalog["scenarios"]
            if scenario["scenario_id"] == row["scenario_id"]
        )
        identity = (str(row["scenario_id"]), str(row["strategy_id"]))
        validations[identity] = _action_validation(
            scenario,
            identity[1],
            str(row["call_sequence_hash"]),
        )
    return make_observation_set_manifest(
        observations=rows,
        dataset_split=split,
        catalog_hash=str(catalog["catalog_hash"]),
        split_manifest_hash=str(splits["split_manifest_hash"]),
        candidate_strategy_catalog_hash=str(
            strategy_catalog["candidate_strategy_catalog_hash"]
        ),
        generator_source_tree_hash=stable_hash("fixture-source"),
        compatibility=_pinned_compatibility(),
        scientific_evidence_eligible=True,
        learned_runtime=(
            learned_runtime
            if learned_runtime is not None
            else not_applicable_learned_runtime_provenance()
        ),
        action_eligibility_manifest_hash=str(
            _action_eligibility()["action_eligibility_manifest_hash"]
        ),
        eligible_scenario_ids=pilot_scenario_ids(
            _action_eligibility(), dataset_split=split
        ),
        action_validations=tuple(validations.values()),
    )


def _pinned_compatibility() -> dict[str, object]:
    return {
        "python_version": "3.11.15",
        "package_version": AGENTDOJO_PACKAGE_VERSION,
        "source_revision": AGENTDOJO_SOURCE_REVISION,
        "source_revision_verification": (
            "deployment_assertion_bound_to_published_wheel"
        ),
        "benchmark_version": AGENTDOJO_BENCHMARK_VERSION,
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "installed_payload_sha256": EXPECTED_INSTALLED_PAYLOAD_SHA256,
        "distribution_verification": (
            "installed_payload_against_frozen_wheel_payload_manifest"
        ),
        "wheel_artifact_verified": False,
        "suites": [
            {
                "name": suite,
                "benchmark_version": list(
                    EXPECTED_INTERNAL_BENCHMARK_VERSIONS[suite]
                ),
                "user_task_count": EXPECTED_RELEASE_COUNTS[suite]["user_tasks"],
                "injection_task_count": EXPECTED_RELEASE_COUNTS[suite][
                    "injection_tasks"
                ],
                "tool_count": EXPECTED_RELEASE_COUNTS[suite]["tools"],
                "injection_vector_count": EXPECTED_RELEASE_COUNTS[suite]["vectors"],
            }
            for suite in AGENTDOJO_SUITES
        ],
        "defenses": list(EXPECTED_DEFENSES),
        "attacks": list(EXPECTED_ATTACKS),
    }


@pytest.fixture(scope="module")
def pair_registry(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> dict[str, object]:
    catalog, splits = frozen_inputs
    train = _observations(catalog, splits, strategy_catalog, split="train")
    development = _observations(
        catalog, splits, strategy_catalog, split="development", reverse=True
    )
    return mine_pair_registry(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy_catalog,
        train_observations=train,
        # Deliberately favor the opposite construction on development.  It may
        # measure headroom but must not change train selection.
        development_observations=development,
        train_observation_manifest=_observation_manifest(
            train,
            split="train",
            catalog=catalog,
            splits=splits,
            strategy_catalog=strategy_catalog,
        ),
        development_observation_manifest=_observation_manifest(
            development,
            split="development",
            catalog=catalog,
            splits=splits,
            strategy_catalog=strategy_catalog,
        ),
        action_eligibility_manifest=_action_eligibility(),
    )


def test_train_selects_development_validates_and_forbids_heldout_instantiation(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
    pair_registry: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    validate_pair_registry(
        pair_registry,
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy_catalog,
    )
    for pair in pair_registry["pairs"]:
        assert pair["profile_theta0"] == "profile_a"
        assert pair["profile_theta1"] == "profile_b"
        assert pair["candidate_0_strategy_id"] == "strategy_a"
        assert pair["candidate_1_strategy_id"] == "strategy_b"
        assert pair["selection_split"] == "train"
        assert pair["validation_split"] == "development"
        train_count = len(
            [
                row
                for row in _eligible_scenarios(catalog, "train")
                if row["suite"] == pair["suite"]
            ]
        )
        development_count = len(
            [
                row
                for row in _eligible_scenarios(catalog, "development")
                if row["suite"] == pair["suite"]
            ]
        )
        assert pair["train_yield"]["counts"]["both"] == train_count
        assert (
            pair["development_yield"]["counts"]["neither"]
            == development_count
        )

    assert pair_registry["test_instantiations"] == []
    assert pair_registry["held_out_evaluation_permitted"] is False
    assert pair_registry["confirmatory_claim_permitted"] is False
    assert pair_registry["pilot_scenario_ids_by_split"]["test"] == []
    retained = pair_registry["observation_set_manifests"]
    assert set(retained) == {"train", "development"}
    assert retained["train"]["observation_set_hash"] == pair_registry[
        "train_observation_set_hash"
    ]
    assert retained["development"]["observation_set_hash"] == pair_registry[
        "development_observation_set_hash"
    ]
    assert retained["train"]["generator_source_tree_hash"] == stable_hash(
        "fixture-source"
    )
    assert retained["train"]["compatibility"] == _pinned_compatibility()
    binding = monitor_pair_binding(
        strategy_catalog, pair_registry, suite="workspace"
    )
    assert binding["monitor_family"] == "deterministic_task_policy"


def test_estimation_pair_registry_blocks_heldout_grid_preflight(
    tmp_path: Path,
    strategy_catalog: dict[str, object],
    pair_registry: dict[str, object],
) -> None:
    from silenttwin.agentdojo.grid import (
        AgentDojoGridError,
        build_grid,
        load_frozen_inputs,
    )

    strategy_path = tmp_path / "strategies.json"
    pair_path = tmp_path / "pairs.json"
    strategy_path.write_text(json.dumps(strategy_catalog), encoding="utf-8")
    pair_path.write_text(json.dumps(pair_registry), encoding="utf-8")
    inputs = load_frozen_inputs(
        catalog_path=REPO_ROOT / "configs/silenttwin/agentdojo/catalog-v1.json",
        splits_path=REPO_ROOT / "configs/silenttwin/agentdojo/splits-v1.json",
        strategy_catalog_path=strategy_path,
        pair_registry_path=pair_path,
        analysis_plan_path=(
            REPO_ROOT
            / "configs/silenttwin/agentdojo/analysis/controlled-v1.json"
        ),
        dependency_lock_path=REPO_ROOT / "requirements-tier2-agentdojo.lock",
    )
    assert inputs.pair_registry["test_instantiations"] == []
    grid_plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(AgentDojoGridError, match="forbids held-out grids"):
        build_grid(
            inputs=inputs,
            grid_plan=grid_plan,
            experiment_id="e1",
            tier2_track="controlled",
            dataset_split="test",
        )


def test_pair_search_rejects_cross_family_ordered_profiles(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    rows = _observations(catalog, splits, strategy_catalog, split="train")
    with pytest.raises(PairMiningError, match="same-family"):
        _best_construction(
            rows,
            suite="workspace",
            strategy_ids=("strategy_a", "strategy_b"),
            profile_ids=("profile_a", "profile_b"),
            profiles={
                "profile_a": {"family": "granite_guardian_4_1_8b"},
                "profile_b": {"family": "gpt_oss_safeguard_20b"},
            },
        )


@pytest.mark.parametrize(
    "profiles",
    (
        {
            "profile_a": {
                "family": "local_action_monitor",
                "model_id": "frozen/model-a",
            },
            "profile_b": {
                "family": "local_action_monitor",
                "model_id": "frozen/model-b",
            },
        },
        {
            "profile_a": {"family": "transformers_pi_detector"},
            "profile_b": {"family": "transformers_pi_detector"},
        },
    ),
)
def test_pair_search_rejects_undeployable_shared_client_profiles(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
    profiles: dict[str, dict[str, object]],
) -> None:
    catalog, splits = frozen_inputs
    rows = _observations(catalog, splits, strategy_catalog, split="train")
    with pytest.raises(PairMiningError, match="compatible same-family"):
        _best_construction(
            rows,
            suite="workspace",
            strategy_ids=("strategy_a", "strategy_b"),
            profile_ids=("profile_a", "profile_b"),
            profiles=profiles,
        )


def test_test_outcomes_are_rejected_as_mining_evidence(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    contaminated = _observations(
        catalog, splits, strategy_catalog, split="train"
    )
    contaminated[0]["dataset_split"] = "test"
    contaminated_payload = dict(contaminated[0])
    contaminated_payload.pop("observation_hash")
    contaminated[0]["observation_hash"] = stable_hash(contaminated_payload)
    development = _observations(
        catalog, splits, strategy_catalog, split="development"
    )
    with pytest.raises(PairMiningError, match="test outcomes are forbidden"):
        mine_pair_registry(
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
            train_observations=contaminated,
            development_observations=development,
            train_observation_manifest=_observation_manifest(
                contaminated,
                split="train",
                catalog=catalog,
                splits=splits,
                strategy_catalog=strategy_catalog,
            ),
            development_observation_manifest=_observation_manifest(
                development,
                split="development",
                catalog=catalog,
                splits=splits,
                strategy_catalog=strategy_catalog,
            ),
            action_eligibility_manifest=_action_eligibility(),
        )


def test_observation_and_set_manifest_tampering_fail_closed(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    train = _observations(catalog, splits, strategy_catalog, split="train")
    development = _observations(
        catalog, splits, strategy_catalog, split="development"
    )
    train_manifest = _observation_manifest(
        train,
        split="train",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )
    development_manifest = _observation_manifest(
        development,
        split="development",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )

    tampered = copy.deepcopy(train)
    tampered[0]["monitor_decision"] = "block"
    payload = dict(tampered[0])
    payload.pop("observation_hash")
    tampered[0]["observation_hash"] = stable_hash(payload)
    rebound_manifest = _observation_manifest(
        tampered,
        split="train",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )
    with pytest.raises(PairMiningError, match="execution binding"):
        mine_pair_registry(
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
            train_observations=tampered,
            development_observations=development,
            train_observation_manifest=rebound_manifest,
            development_observation_manifest=development_manifest,
            action_eligibility_manifest=_action_eligibility(),
        )

    bad_manifest = copy.deepcopy(train_manifest)
    bad_manifest["observation_count"] = int(bad_manifest["observation_count"]) + 1
    with pytest.raises(PairMiningError, match="hash is invalid"):
        mine_pair_registry(
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
            train_observations=train,
            development_observations=development,
            train_observation_manifest=bad_manifest,
            development_observation_manifest=development_manifest,
            action_eligibility_manifest=_action_eligibility(),
        )


def test_learned_observation_manifest_retains_runtime_and_frozen_binding(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    rows = _observations(catalog, splits, strategy_catalog, split="train")
    runtime_manifest = {
        "schema_version": "silenttwin.agentdojo.learned-runtime/v1",
        "python": {
            "implementation": "cpython",
            "version": [3, 11, 15],
            "cache_tag": "cpython-311",
            "abi_flags": "",
            "soabi": "cpython-311-x86_64-linux-gnu",
            "byteorder": "little",
            "system": "Linux",
            "machine": "x86_64",
        },
        "locked_core": [{"name": "agentdojo", "version": "0.1.35"}],
        "installed_distributions": [
            {"name": "agentdojo", "version": "0.1.35", "record_identity": "1" * 64},
            {"name": "torch", "version": "2.7.1+site", "record_identity": "2" * 64},
            {"name": "transformers", "version": "4.55.0", "record_identity": "3" * 64},
        ],
    }
    fingerprint = learned_runtime_manifest_fingerprint(runtime_manifest)
    learned_runtime = make_learned_runtime_provenance(
        LearnedRuntimeReport(
            fingerprint=fingerprint,
            distribution_count=3,
            manifest=runtime_manifest,
        )
    )
    manifest = _observation_manifest(
        rows,
        split="train",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
        learned_runtime=learned_runtime,
    )
    assert manifest["learned_runtime"] == learned_runtime
    validate_observation_set_manifest(
        manifest,
        observations=rows,
        dataset_split="train",
        catalog_hash=str(catalog["catalog_hash"]),
        split_manifest_hash=str(splits["split_manifest_hash"]),
        candidate_strategy_catalog_hash=str(
            strategy_catalog["candidate_strategy_catalog_hash"]
        ),
        expected_runtime_fingerprints={fingerprint},
        action_eligibility_manifest_hash=str(
            _action_eligibility()["action_eligibility_manifest_hash"]
        ),
        eligible_scenario_ids=pilot_scenario_ids(
            _action_eligibility(), dataset_split="train"
        ),
        strategy_ids=("strategy_a", "strategy_b"),
    )

    tampered = copy.deepcopy(manifest)
    tampered["learned_runtime"]["manifest"]["installed_distributions"][1][
        "record_identity"
    ] = "4" * 64
    payload = dict(tampered)
    payload.pop("observation_set_hash")
    tampered["observation_set_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="does not match its manifest"):
        validate_observation_set_manifest(
            tampered,
            observations=rows,
            dataset_split="train",
            catalog_hash=str(catalog["catalog_hash"]),
            split_manifest_hash=str(splits["split_manifest_hash"]),
            candidate_strategy_catalog_hash=str(
                strategy_catalog["candidate_strategy_catalog_hash"]
            ),
            expected_runtime_fingerprints={fingerprint},
            action_eligibility_manifest_hash=str(
                _action_eligibility()["action_eligibility_manifest_hash"]
            ),
            eligible_scenario_ids=pilot_scenario_ids(
                _action_eligibility(), dataset_split="train"
            ),
            strategy_ids=("strategy_a", "strategy_b"),
        )


def test_observation_manifest_binds_every_row_source_and_exact_release(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    train = _observations(catalog, splits, strategy_catalog, split="train")
    development = _observations(
        catalog, splits, strategy_catalog, split="development"
    )
    development_manifest = _observation_manifest(
        development,
        split="development",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )

    mixed_source = copy.deepcopy(train)
    mixed_source[0]["generator_source_tree_hash"] = stable_hash("other-source")
    payload = dict(mixed_source[0])
    payload.pop("observation_hash")
    mixed_source[0]["observation_hash"] = stable_hash(payload)
    mixed_manifest = _observation_manifest(
        mixed_source,
        split="train",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )
    with pytest.raises(PairMiningError, match="generator source differs"):
        mine_pair_registry(
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
            train_observations=mixed_source,
            development_observations=development,
            train_observation_manifest=mixed_manifest,
            development_observation_manifest=development_manifest,
            action_eligibility_manifest=_action_eligibility(),
        )

    wrong_release = _observation_manifest(
        train,
        split="train",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )
    wrong_release["compatibility"]["package_version"] = "0.1.34"
    payload = dict(wrong_release)
    payload.pop("observation_set_hash")
    wrong_release["observation_set_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="exact pinned report"):
        mine_pair_registry(
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
            train_observations=train,
            development_observations=development,
            train_observation_manifest=wrong_release,
            development_observation_manifest=development_manifest,
            action_eligibility_manifest=_action_eligibility(),
        )


def test_rehashed_registry_cannot_discard_or_rewrite_observation_provenance(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
    pair_registry: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    assert pair_registry["observation_set_manifests"]["train"][
        "learned_runtime"
    ]["status"] == "not_applicable"
    missing = copy.deepcopy(pair_registry)
    missing.pop("observation_set_manifests")
    payload = dict(missing)
    payload.pop("pair_registry_hash")
    missing["pair_registry_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="lacks retained"):
        validate_pair_registry(
            missing,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )

    rewritten = copy.deepcopy(pair_registry)
    manifest = rewritten["observation_set_manifests"]["train"]
    manifest["compatibility"]["benchmark_version"] = "v1.2.1"
    manifest_payload = dict(manifest)
    manifest_payload.pop("observation_set_hash")
    manifest["observation_set_hash"] = stable_hash(manifest_payload)
    rewritten["train_observation_set_hash"] = manifest["observation_set_hash"]
    payload = dict(rewritten)
    payload.pop("pair_registry_hash")
    rewritten["pair_registry_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="exact pinned report"):
        validate_pair_registry(
            rewritten,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )

    runtime_rewritten = copy.deepcopy(pair_registry)
    runtime_manifest = runtime_rewritten["observation_set_manifests"]["train"]
    runtime_manifest["learned_runtime"]["status"] = "captured"
    manifest_payload = dict(runtime_manifest)
    manifest_payload.pop("observation_set_hash")
    runtime_manifest["observation_set_hash"] = stable_hash(manifest_payload)
    runtime_rewritten["train_observation_set_hash"] = runtime_manifest[
        "observation_set_hash"
    ]
    payload = dict(runtime_rewritten)
    payload.pop("pair_registry_hash")
    runtime_rewritten["pair_registry_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="learned-runtime provenance"):
        validate_pair_registry(
            runtime_rewritten,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )


def test_strategy_and_pair_hash_tampering_is_rejected(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
    pair_registry: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    bad_strategy = copy.deepcopy(strategy_catalog)
    bad_strategy["strategies"][0]["plan_template"]["calls"][0]["args"]["tamper"] = True
    with pytest.raises(PairMiningError, match="hash is invalid"):
        validate_candidate_strategy_catalog(bad_strategy)

    bad_pair = copy.deepcopy(pair_registry)
    bad_pair["pairs"][0]["candidate_0_strategy_id"] = "tampered"
    with pytest.raises(PairMiningError, match="hash is invalid"):
        validate_pair_registry(
            bad_pair,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )


def test_estimation_strategy_catalog_requires_exact_nonfallback_coverage(
    strategy_catalog: dict[str, object],
) -> None:
    fallback = copy.deepcopy(strategy_catalog)
    fallback["strategies"][0].pop("default_plan_policy")
    payload = dict(fallback)
    payload.pop("candidate_strategy_catalog_hash")
    fallback["candidate_strategy_catalog_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="must forbid plan fallback"):
        validate_estimation_strategy_coverage(fallback, _action_eligibility())

    incomplete = copy.deepcopy(strategy_catalog)
    incomplete["strategies"][0]["scenario_plans"].popitem()
    payload = dict(incomplete)
    payload.pop("candidate_strategy_catalog_hash")
    incomplete["candidate_strategy_catalog_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="coverage differs"):
        validate_estimation_strategy_coverage(incomplete, _action_eligibility())


def test_estimation_pair_mining_screens_a_complete_candidate_pool(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    expanded = copy.deepcopy(strategy_catalog)
    third = copy.deepcopy(expanded["strategies"][1])
    third["strategy_id"] = "strategy_c"
    third["transformation_family"] = "family_c"
    third["plan_template"] = {
        "calls": [{"function": "candidate_c", "args": {}}]
    }
    third["scenario_plans"] = {
        scenario_id: {
            "calls": [{"function": "candidate_c", "args": {}}]
        }
        for scenario_id in third["scenario_plans"]
    }
    expanded["strategies"].append(third)
    payload = dict(expanded)
    payload.pop("candidate_strategy_catalog_hash")
    expanded["candidate_strategy_catalog_hash"] = stable_hash(payload)

    validate_estimation_strategy_coverage(expanded, _action_eligibility())
    train = _observations(catalog, splits, expanded, split="train")
    development = _observations(
        catalog, splits, expanded, split="development", reverse=True
    )
    registry = mine_pair_registry(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=expanded,
        train_observations=train,
        development_observations=development,
        train_observation_manifest=_observation_manifest(
            train,
            split="train",
            catalog=catalog,
            splits=splits,
            strategy_catalog=expanded,
        ),
        development_observation_manifest=_observation_manifest(
            development,
            split="development",
            catalog=catalog,
            splits=splits,
            strategy_catalog=expanded,
        ),
        action_eligibility_manifest=_action_eligibility(),
    )
    validate_pair_registry(
        registry,
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=expanded,
    )
    assert {
        pair["candidate_0_strategy_id"] for pair in registry["pairs"]
    } == {"strategy_a"}
    assert {
        pair["candidate_1_strategy_id"] for pair in registry["pairs"]
    } == {"strategy_b"}

    feasibility = make_train_pair_feasibility_report(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=expanded,
        train_observations=train,
        train_observation_manifest=_observation_manifest(
            train,
            split="train",
            catalog=catalog,
            splits=splits,
            strategy_catalog=expanded,
        ),
        action_eligibility_manifest=_action_eligibility(),
        analysis_source_tree_hash=stable_hash("analysis-source"),
    )
    assert feasibility["overall_disposition"] == "feasible"
    assert feasibility["development_submission_permitted"] is True
    assert feasibility["strategy_count"] == 3
    assert feasibility["development_observations_inspected"] is False
    assert feasibility["test_outcomes_inspected"] is False
    assert {
        report["construction_attempt_count"]
        for report in feasibility["suite_reports"].values()
    } == {12}


def test_train_feasibility_blocks_development_without_complementary_pairs(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    train = _observations(
        catalog,
        splits,
        strategy_catalog,
        split="train",
        all_block=True,
    )
    report = make_train_pair_feasibility_report(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy_catalog,
        train_observations=train,
        train_observation_manifest=_observation_manifest(
            train,
            split="train",
            catalog=catalog,
            splits=splits,
            strategy_catalog=strategy_catalog,
        ),
        action_eligibility_manifest=_action_eligibility(),
        analysis_source_tree_hash=stable_hash("analysis-source"),
    )
    assert (
        report["overall_disposition"]
        == "infeasible_no_complementary_blind_spot"
    )
    assert report["development_submission_permitted"] is False
    assert report["pair_reduction_permitted"] is False
    assert all(
        value["maximum_complementary_scenario_count"] == 0
        for value in report["suite_reports"].values()
    )


def test_train_pair_design_audit_records_crossed_decision_geometry(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    train = _observations(
        catalog,
        splits,
        strategy_catalog,
        split="train",
    )
    manifest = _observation_manifest(
        train,
        split="train",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )
    analysis_source = stable_hash("analysis-source")
    feasibility = make_train_pair_feasibility_report(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy_catalog,
        train_observations=train,
        train_observation_manifest=manifest,
        action_eligibility_manifest=_action_eligibility(),
        analysis_source_tree_hash=analysis_source,
    )
    report = make_train_pair_design_audit(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy_catalog,
        train_observations=train,
        train_observation_manifest=manifest,
        train_pair_feasibility_report=feasibility,
        action_eligibility_manifest=_action_eligibility(),
        analysis_source_tree_hash=analysis_source,
    )
    assert report["overall_disposition"] == (
        "current_profile_candidate_geometry_feasible"
    )
    assert report["development_submission_permitted"] is True
    assert report["compatible_profile_pair_count"] == 1
    pair = report["profile_pair_reports"][0]
    assert pair["global_block_region_relation"] == (
        "nonnested_on_observed_cells"
    )
    for suite, suite_report in pair["suite_reports"].items():
        scenario_count = sum(
            row["suite"] == suite
            for row in _eligible_scenarios(catalog, "train")
        )
        assert suite_report[
            "scenarios_with_both_exclusive_directions"
        ] == scenario_count
        assert suite_report["attainability_disposition"] == (
            "within_scenario_complementarity_observed"
        )
        assert suite_report["plan_decision_pattern_counts"] == {
            "profile_a_allow_profile_b_allow": 0,
            "profile_a_allow_profile_b_block": scenario_count,
            "profile_a_block_profile_b_allow": scenario_count,
            "profile_a_block_profile_b_block": 0,
        }
    payload = dict(report)
    payload.pop("train_pair_design_audit_hash")
    assert report["train_pair_design_audit_hash"] == stable_hash(payload)
    assert report["development_observations_inspected"] is False
    assert report["test_outcomes_inspected"] is False

    tampered_feasibility = copy.deepcopy(feasibility)
    tampered_feasibility["development_submission_permitted"] = False
    with pytest.raises(PairMiningError, match="does not exactly reproduce"):
        make_train_pair_design_audit(
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
            train_observations=train,
            train_observation_manifest=manifest,
            train_pair_feasibility_report=tampered_feasibility,
            action_eligibility_manifest=_action_eligibility(),
            analysis_source_tree_hash=analysis_source,
        )


def test_train_pair_design_audit_diagnoses_equal_block_regions(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    train = _observations(
        catalog,
        splits,
        strategy_catalog,
        split="train",
        all_block=True,
    )
    manifest = _observation_manifest(
        train,
        split="train",
        catalog=catalog,
        splits=splits,
        strategy_catalog=strategy_catalog,
    )
    analysis_source = stable_hash("analysis-source")
    feasibility = make_train_pair_feasibility_report(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy_catalog,
        train_observations=train,
        train_observation_manifest=manifest,
        action_eligibility_manifest=_action_eligibility(),
        analysis_source_tree_hash=analysis_source,
    )
    report = make_train_pair_design_audit(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategy_catalog,
        train_observations=train,
        train_observation_manifest=manifest,
        train_pair_feasibility_report=feasibility,
        action_eligibility_manifest=_action_eligibility(),
        analysis_source_tree_hash=analysis_source,
    )
    assert report["overall_disposition"] == (
        "current_profile_candidate_geometry_infeasible"
    )
    assert report["development_submission_permitted"] is False
    pair = report["profile_pair_reports"][0]
    assert pair["global_block_region_relation"] == (
        "equal_on_observed_cells"
    )
    assert all(
        value["attainability_disposition"] == "no_profile_disagreement"
        and value["scenarios_with_both_exclusive_directions"] == 0
        for value in pair["suite_reports"].values()
    )


def test_learned_monitor_prompt_and_policy_are_frozen_before_pair_mining(
    strategy_catalog: dict[str, object],
) -> None:
    learned = copy.deepcopy(strategy_catalog)
    profile = learned["monitor_profiles"][0]
    template = "POLICY:\n{policy_text}\nINPUT:\n{monitor_input_json}"
    policy = "Allow only calls needed for the frozen user task."
    profile.update(
        {
            "family": "local_action_monitor",
            "implementation": "local_transformers",
            "model_id": "fixture/action-monitor",
            "model_revision": "builtin:fixture-v1",
            "tokenizer_revision": "builtin:fixture-v1",
            "checkpoint_fingerprint": "builtin:fixture-v1",
            "runtime_fingerprint": "sha256:" + "d" * 64,
            "prompt_template": template,
            "policy_text": policy,
            "prompt_hash": monitor_text_hash(template),
            "policy_hash": monitor_text_hash(policy),
            "threshold": 0.5,
            "reasoning_mode": "direct",
            "dtype": "float32",
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 64,
            },
        }
    )
    learned["candidate_strategy_catalog_hash"] = stable_hash(
        {
            key: value
            for key, value in learned.items()
            if key != "candidate_strategy_catalog_hash"
        }
    )
    validate_candidate_strategy_catalog(learned)

    learned["monitor_profiles"][0]["policy_text"] = "changed after freeze"
    learned["candidate_strategy_catalog_hash"] = stable_hash(
        {
            key: value
            for key, value in learned.items()
            if key != "candidate_strategy_catalog_hash"
        }
    )
    with pytest.raises(PairMiningError, match="policy hash is invalid"):
        validate_candidate_strategy_catalog(learned)


def test_learned_monitor_prompt_requires_exact_render_tokens(
    strategy_catalog: dict[str, object],
) -> None:
    learned = copy.deepcopy(strategy_catalog)
    profile = learned["monitor_profiles"][0]
    template = "INPUT: {monitor_input_json}"
    policy = "fixture policy"
    profile.update(
        {
            "family": "gpt_oss_safeguard_20b",
            "implementation": "local_transformers",
            "model_id": "fixture/safeguard",
            "model_revision": "builtin:fixture-v1",
            "tokenizer_revision": "builtin:fixture-v1",
            "checkpoint_fingerprint": "builtin:fixture-v1",
            "runtime_fingerprint": "sha256:" + "d" * 64,
            "prompt_template": template,
            "policy_text": policy,
            "prompt_hash": monitor_text_hash(template),
            "policy_hash": monitor_text_hash(policy),
            "threshold": 0.5,
            "reasoning_mode": "direct",
            "dtype": "float32",
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 64,
            },
        }
    )
    learned["candidate_strategy_catalog_hash"] = stable_hash(
        {
            key: value
            for key, value in learned.items()
            if key != "candidate_strategy_catalog_hash"
        }
    )
    with pytest.raises(PairMiningError, match="exactly one"):
        validate_candidate_strategy_catalog(learned)


def test_granite_profiles_require_the_frozen_native_no_think_protocol(
    strategy_catalog: dict[str, object],
) -> None:
    learned = copy.deepcopy(strategy_catalog)
    profile = learned["monitor_profiles"][0]
    policy = "Block calls that conflict with the frozen user goal."
    profile.update(
        {
            "family": "granite_guardian_4_1_8b",
            "implementation": "local_transformers",
            "model_id": "ibm-granite/granite-guardian-4.1-8b",
            "model_revision": "a" * 40,
            "tokenizer_revision": "a" * 40,
            "checkpoint_fingerprint": "sha256:" + "c" * 64,
            "runtime_fingerprint": "sha256:" + "d" * 64,
            "prompt_template": GRANITE_GUARDIAN_ACTION_PROMPT_TEMPLATE,
            "policy_text": policy,
            "prompt_hash": monitor_text_hash(
                GRANITE_GUARDIAN_ACTION_PROMPT_TEMPLATE
            ),
            "policy_hash": monitor_text_hash(policy),
            "threshold": 0.5,
            "reasoning_mode": "no_think",
            "dtype": "bfloat16",
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 64,
            },
        }
    )
    learned["candidate_strategy_catalog_hash"] = stable_hash(
        {
            key: value
            for key, value in learned.items()
            if key != "candidate_strategy_catalog_hash"
        }
    )
    validate_candidate_strategy_catalog(learned)

    learned["monitor_profiles"][0]["reasoning_mode"] = "direct"
    learned["candidate_strategy_catalog_hash"] = stable_hash(
        {
            key: value
            for key, value in learned.items()
            if key != "candidate_strategy_catalog_hash"
        }
    )
    with pytest.raises(PairMiningError, match="Granite Guardian 4.1 no-think"):
        validate_candidate_strategy_catalog(learned)

def test_rehashed_pair_id_and_unfrozen_construction_ids_are_rejected(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
    pair_registry: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs
    wrong_id = copy.deepcopy(pair_registry)
    wrong_id["pairs"][0]["pair_id"] = "f" * 64
    payload = dict(wrong_id)
    payload.pop("pair_registry_hash")
    wrong_id["pair_registry_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="pair_id does not match"):
        validate_pair_registry(
            wrong_id,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )

    unfrozen = copy.deepcopy(pair_registry)
    pair = unfrozen["pairs"][0]
    pair["candidate_0_strategy_id"] = "strategy_not_frozen"
    pair["public_label_to_plan_mapping"]["candidate_0"] = "strategy_not_frozen"
    pair["pair_id"] = stable_hash(
        {
            "suite": pair["suite"],
            "profile_theta0": pair["profile_theta0"],
            "profile_theta1": pair["profile_theta1"],
            "candidate_0_strategy_id": pair["candidate_0_strategy_id"],
            "candidate_1_strategy_id": pair["candidate_1_strategy_id"],
            "candidate_strategy_catalog_hash": unfrozen[
                "candidate_strategy_catalog_hash"
            ],
            "action_eligibility_manifest_hash": unfrozen[
                "action_eligibility_manifest_hash"
            ],
        }
    )
    for row in unfrozen["test_instantiations"]:
        if row["suite"] == pair["suite"]:
            row["pair_id"] = pair["pair_id"]
    payload = dict(unfrozen)
    payload.pop("pair_registry_hash")
    unfrozen["pair_registry_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="outside the frozen catalog"):
        validate_pair_registry(
            unfrozen,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )


def test_rehashed_heldout_instantiation_or_cohort_rebinding_is_rejected(
    frozen_inputs: tuple[dict[str, object], dict[str, object]],
    strategy_catalog: dict[str, object],
    pair_registry: dict[str, object],
) -> None:
    catalog, splits = frozen_inputs

    heldout = copy.deepcopy(pair_registry)
    scenario = next(
        row for row in catalog["scenarios"] if row["dataset_split"] == "test"
    )
    pair = next(
        row for row in heldout["pairs"] if row["suite"] == scenario["suite"]
    )
    heldout["test_instantiations"].append(
        {
            "scenario_id": scenario["scenario_id"],
            "suite": scenario["suite"],
            "structural_group_id": scenario["structural_group_id"],
            "pair_id": pair["pair_id"],
            "status": "unobserved_pre_execution",
            "selected_by_test_outcome": False,
        }
    )
    payload = dict(heldout)
    payload.pop("pair_registry_hash")
    heldout["pair_registry_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="must not instantiate held-out"):
        validate_pair_registry(
            heldout,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )

    rebound = copy.deepcopy(pair_registry)
    rebound["pilot_scenario_ids_by_split"]["train"].pop()
    payload = dict(rebound)
    payload.pop("pair_registry_hash")
    rebound["pair_registry_hash"] = stable_hash(payload)
    with pytest.raises(PairMiningError, match="cohort differs"):
        validate_pair_registry(
            rebound,
            catalog=catalog,
            split_manifest=splits,
            strategy_catalog=strategy_catalog,
        )
