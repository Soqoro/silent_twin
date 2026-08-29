from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from silenttwin.agentdojo.assembly import _profile_decisions
from silenttwin.agentdojo.config import AGENTDOJO_SUITES, stable_hash
from silenttwin.agentdojo.grid import (
    AgentDojoGrid,
    AgentDojoGridError,
    GridCell,
    GridTask,
    _experiment_cells,
    _models_from_plan,
    _validate_recipient_separation_models,
    _validate_preregistered_cells,
    is_estimation_only_protocol_disposition,
)
from silenttwin.agentdojo.recipient_separation import (
    RECIPIENT_SEPARATION_ATTACKER_IDENTITY,
    RECIPIENT_SEPARATION_DISPOSITION,
    RECIPIENT_SEPARATION_PROFILE_IDS,
    RecipientSeparationError,
    _scientific_v6_artifacts_from_validated_inputs,
    validate_recipient_separation_candidate_catalog,
    validate_recipient_separation_pair_registry,
    validate_recipient_separation_protocol,
)
from tests.unit.test_agentdojo_successor_design import (
    _minimal_v5_design_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / (
    "configs/silenttwin/agentdojo/"
    "scientific-v6-recipient-separation-protocol-v1.json"
)
ANALYSIS_PATH = REPO_ROOT / (
    "configs/silenttwin/agentdojo/analysis/recipient-separation-v1.json"
)
GRID_TEMPLATE_PATH = REPO_ROOT / (
    "configs/silenttwin/agentdojo/grid-plans/"
    "recipient-separation-train-template-v1.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rehash(value: dict, field: str) -> None:
    payload = dict(value)
    payload.pop(field, None)
    value[field] = stable_hash(payload)


def _fixture_artifacts() -> tuple[dict, dict, dict]:
    _, _, predecessor = _minimal_v5_design_fixture()
    eligibility_payload = {
        "schema_version": "fixture-action-eligibility.v1",
        "pilot_scenario_ids_by_split": {
            "train": ["train-excluded", "train-selected"],
            "development": [
                "development-excluded",
                "development-selected",
            ],
            "test": [],
        },
    }
    eligibility = {
        **eligibility_payload,
        "action_eligibility_manifest_hash": stable_hash(eligibility_payload),
    }
    predecessor["scenario_cohort"]["action_eligibility_manifest_hash"] = (
        eligibility["action_eligibility_manifest_hash"]
    )
    _rehash(predecessor["scenario_cohort"], "cohort_hash")
    _rehash(predecessor, "candidate_strategy_catalog_hash")

    audit_payload = {
        "schema_version": "silenttwin.agentdojo.train_pair_design_audit.v1",
        "fixture": True,
    }
    audit = {
        **audit_payload,
        "train_pair_design_audit_hash": stable_hash(audit_payload),
    }
    protocol = _load(PROTOCOL_PATH)
    protocol["upstream_bindings"] = {
        "catalog_hash": stable_hash("fixture-catalog"),
        "split_manifest_hash": stable_hash("fixture-splits"),
        "action_eligibility_manifest_hash": eligibility[
            "action_eligibility_manifest_hash"
        ],
        "predecessor_candidate_strategy_catalog_hash": predecessor[
            "candidate_strategy_catalog_hash"
        ],
        "predecessor_train_pair_design_audit_hash": audit[
            "train_pair_design_audit_hash"
        ],
        "analysis_plan_hash": stable_hash("fixture-analysis"),
    }
    _rehash(protocol, "protocol_hash")
    strategy, pair = _scientific_v6_artifacts_from_validated_inputs(
        protocol=protocol,
        predecessor_strategy_catalog=predecessor,
        predecessor_train_design_audit=audit,
        action_eligibility_manifest=eligibility,
        authoring_source_tree_hash=stable_hash("fixture-source"),
    )
    return protocol, strategy, pair


def test_checked_recipient_separation_protocol_is_self_consistent() -> None:
    protocol = _load(PROTOCOL_PATH)

    assert validate_recipient_separation_protocol(protocol) == protocol[
        "protocol_hash"
    ]
    assert protocol["access_policy"]["execution_permitted_splits"] == [
        "train"
    ]
    assert protocol["experiments"]["r4_clean_repair"]["status"].startswith(
        "blocked_"
    )
    attacker = dict(protocol["primary_attacker_identity"])
    attacker.pop("runtime_fingerprint_binding")
    assert attacker == RECIPIENT_SEPARATION_ATTACKER_IDENTITY


def test_recipient_separation_artifacts_encode_authored_complementarity() -> None:
    protocol, strategy, pair = _fixture_artifacts()

    validate_recipient_separation_candidate_catalog(strategy)
    validate_recipient_separation_pair_registry(
        pair, strategy_catalog=strategy
    )
    profiles = strategy["monitor_profiles"]
    strategy_ids = protocol["private_authorization_construction"][
        "candidate_strategy_ids"
    ]
    assert [row["profile_id"] for row in profiles] == list(
        RECIPIENT_SEPARATION_PROFILE_IDS
    )
    assert profiles[0]["decision_registry"] == {
        strategy_ids[0]: "allow",
        strategy_ids[1]: "block",
    }
    assert profiles[1]["decision_registry"] == {
        strategy_ids[0]: "block",
        strategy_ids[1]: "allow",
    }
    assert pair["selection_protocol"].startswith("authored_")
    assert pair["pair_selection_used_learned_monitor_outcomes"] is False
    assert pair["execution_permitted_splits"] == ["train"]
    assert pair["test_instantiations"] == []
    assert {row["suite"] for row in pair["pairs"]} == set(AGENTDOJO_SUITES)


def test_recipient_separation_profile_tampering_is_rejected_even_if_rehashed() -> None:
    _, strategy, _ = _fixture_artifacts()
    tampered = copy.deepcopy(strategy)
    tampered["monitor_profiles"][0]["decision_registry"] = {
        key: "allow"
        for key in tampered["monitor_profiles"][0]["decision_registry"]
    }
    _rehash(tampered["monitor_profiles"][0], "profile_hash")
    _rehash(tampered, "candidate_strategy_catalog_hash")

    with pytest.raises(RecipientSeparationError, match="profile drifted"):
        validate_recipient_separation_candidate_catalog(tampered)


def test_recipient_separation_profiles_bind_public_candidate_and_probe_labels() -> None:
    protocol, strategy, _ = _fixture_artifacts()
    strategy_ids = protocol["private_authorization_construction"][
        "candidate_strategy_ids"
    ]
    labels = {
        "candidate_0": strategy_ids[0],
        "candidate_1": strategy_ids[1],
    }

    theta0 = _profile_decisions(
        strategy["monitor_profiles"][0],
        scenario_id="train-selected",
        label_to_strategy=labels,
    )
    theta1 = _profile_decisions(
        strategy["monitor_profiles"][1],
        scenario_id="train-selected",
        label_to_strategy=labels,
    )

    assert theta0["candidate_0"] == theta0["probe_candidate_0"] == "allow"
    assert theta0["candidate_1"] == theta0["probe_candidate_1"] == "block"
    assert theta1["candidate_0"] == theta1["probe_candidate_0"] == "block"
    assert theta1["candidate_1"] == theta1["probe_candidate_1"] == "allow"


def test_recipient_separation_pair_cannot_claim_development_permission() -> None:
    _, strategy, pair = _fixture_artifacts()
    tampered = copy.deepcopy(pair)
    tampered["execution_permitted_splits"] = ["train", "development"]
    tampered["development_submission_permitted"] = True
    _rehash(tampered, "pair_registry_hash")

    with pytest.raises(RecipientSeparationError, match="train-only boundary"):
        validate_recipient_separation_pair_registry(
            tampered, strategy_catalog=strategy
        )


def test_recipient_separation_query_matrix_matches_analysis_plan() -> None:
    analysis = _load(ANALYSIS_PATH)
    template = _load(GRID_TEMPLATE_PATH)

    e1 = _experiment_cells(template, "e1")
    e2 = _experiment_cells(template, "e2")
    _validate_preregistered_cells("e1", e1, analysis_plan=analysis)
    _validate_preregistered_cells("e2", e2, analysis_plan=analysis)
    assert {row["query_budget"] for row in e1} == {0, 4, 16}
    assert len(e1) == 36
    assert len(e2) == 13


def test_recipient_separation_grid_rejects_attacker_substitution() -> None:
    template = _load(GRID_TEMPLATE_PATH)
    template["models"][0]["runtime_fingerprint"] = "sha256:" + "a" * 64
    models = _models_from_plan(template)
    _validate_recipient_separation_models(models)

    substituted = copy.deepcopy(template)
    substituted["models"][0]["model_id"] = "another/model"
    with pytest.raises(AgentDojoGridError, match="frozen protocol identity"):
        _validate_recipient_separation_models(
            _models_from_plan(substituted)
        )


def test_recipient_separation_grid_metadata_is_estimation_only() -> None:
    cell = GridCell(0, {}, stable_hash("configuration"), stable_hash("shard"))
    task = GridTask(0, "workspace", stable_hash("bundle"), 0, (cell,))
    grid = AgentDojoGrid(
        experiment_id="e1",
        tier2_track="controlled",
        dataset_split="train",
        tasks=(task,),
        upstream_binding_hash=stable_hash("upstream"),
        protocol_disposition=RECIPIENT_SEPARATION_DISPOSITION,
        action_eligibility_manifest_hash=stable_hash("eligibility"),
    )

    assert is_estimation_only_protocol_disposition(
        RECIPIENT_SEPARATION_DISPOSITION
    )
    assert grid.metadata()["confirmatory_suite_coverage_eligible"] is False
    with pytest.raises(AgentDojoGridError, match="held-out"):
        AgentDojoGrid(
            experiment_id="e1",
            tier2_track="controlled",
            dataset_split="test",
            tasks=(task,),
            upstream_binding_hash=stable_hash("upstream"),
            protocol_disposition=RECIPIENT_SEPARATION_DISPOSITION,
            action_eligibility_manifest_hash=stable_hash("eligibility"),
        )
