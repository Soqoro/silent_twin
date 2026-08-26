from __future__ import annotations

import json
from pathlib import Path

import pytest

from silenttwin.agentdojo.config import stable_hash
from silenttwin.agentdojo.gates import (
    AgentDojoGateError,
    evaluate_gates,
    validate_upstream_e1_gate_manifest,
)
from silenttwin.agentdojo.statistics import (
    attack_error_accounting,
    collapse_repeated_measurements,
    paired_scenario_contrast,
    suite_stratified_cluster_binary_upper_bound,
    suite_stratified_cluster_bootstrap_ci,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _analysis_plan() -> dict[str, object]:
    return json.loads(
        (REPO_ROOT / "configs/silenttwin/agentdojo/analysis/controlled-v1.json").read_text()
    )


def _contrast(estimate: float, lower: float, upper: float, level: float = 0.95) -> dict[str, float]:
    return {
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": level,
    }


def test_repeated_rows_collapse_before_scenario_inference() -> None:
    records = [
        {
            "agentdojo_suite": "workspace",
            "structural_group_id": "g1",
            "condition": "genuine",
            "attack_success": value,
        }
        for value in (0, 1, 1, 0)
    ]
    collapsed = collapse_repeated_measurements(
        records, metric="attack_success", condition_fields=("condition",)
    )
    assert collapsed == [
        {
            "agentdojo_suite": "workspace",
            "structural_group_id": "g1",
            "condition": "genuine",
            "attack_success": 0.5,
            "nested_row_count": 4,
        }
    ]


def test_primary_contrast_uses_equal_suite_weight_with_task_weighted_sensitivity() -> None:
    target = [
        {"agentdojo_suite": "workspace", "structural_group_id": "w0", "y": 1.0},
        *[
            {"agentdojo_suite": "travel", "structural_group_id": f"t{i}", "y": 0.0}
            for i in range(9)
        ],
    ]
    reference = [{**row, "y": 0.0} for row in target]
    result = paired_scenario_contrast(
        target,
        reference,
        metric="y",
        contrast_id="equal_suite_test",
        resamples=100,
    )
    assert result["estimate"] == pytest.approx(0.5)
    assert result["task_weighted_sensitivity_estimate"] == pytest.approx(0.1)
    assert result["suite_weighting"] == "equal_suite"
    assert result["task_level_paired_estimates"][0] == {
        "agentdojo_suite": "travel",
        "structural_group_id": "t0",
        "target": 0.0,
        "reference": 0.0,
        "difference": 0.0,
    }
    assert len(result["task_level_paired_estimates"]) == 10


def test_cluster_bootstrap_keeps_nested_rows_together_and_is_deterministic() -> None:
    observations = [0.0, 0.0, 1.0, 1.0]
    clusters = ["a", "a", "b", "b"]
    suites = ["workspace"] * 4
    first = suite_stratified_cluster_bootstrap_ci(
        observations,
        clusters,
        suites,
        lambda rows: sum(rows) / len(rows),
        resamples=200,
        seed=7,
    )
    second = suite_stratified_cluster_bootstrap_ci(
        observations,
        clusters,
        suites,
        lambda rows: sum(rows) / len(rows),
        resamples=200,
        seed=7,
    )
    assert first == second
    assert first[0] == 0.0
    assert first[1] == 1.0


def test_conservative_asr_counts_every_invalid_run_as_attack_success() -> None:
    accounting = attack_error_accounting(
        [
            {"run_valid": True, "attack_success": False},
            {"run_valid": True, "attack_success": True},
            {"run_valid": False, "attack_success": None},
        ]
    )
    assert accounting["valid_run_asr"] == 0.5
    assert accounting["conservative_asr"] == pytest.approx(2 / 3)
    assert accounting["error_rate"] == pytest.approx(1 / 3)


def test_all_error_accounting_is_json_safe_and_has_no_valid_denominator() -> None:
    accounting = attack_error_accounting(
        [
            {"run_valid": False, "attack_success": None},
            {"run_valid": False, "attack_success": None},
        ]
    )
    assert accounting["valid_run_asr"] is None
    assert accounting["conservative_asr"] == 1.0
    assert accounting["error_rate"] == 1.0


def test_finite_sample_cluster_upper_bound_is_positive_for_all_zero_sample() -> None:
    one_cluster = suite_stratified_cluster_binary_upper_bound(
        [0.0, 0.0],
        ["g0", "g0"],
        ["workspace", "workspace"],
    )
    assert one_cluster["estimate"] == 0.0
    assert one_cluster["ci_upper"] == pytest.approx(0.95)
    assert one_cluster["independent_unit_count"] == 1
    assert "clopper_pearson" in one_cluster["ci_method"]

    many_clusters = suite_stratified_cluster_binary_upper_bound(
        [0.0] * 400,
        [f"g{index}" for index in range(400)],
        ["workspace"] * 400,
    )
    assert 0.0 < many_clusters["ci_upper"] < 0.01


def test_cluster_upper_bound_counts_any_nested_violation_once() -> None:
    bound = suite_stratified_cluster_binary_upper_bound(
        [0.0, 1.0, 0.0, 0.0],
        ["g0", "g0", "g1", "g1"],
        ["workspace"] * 4,
    )
    assert bound["suite_bounds"]["workspace"]["cluster_count"] == 2
    assert bound["suite_bounds"]["workspace"]["event_cluster_count"] == 1
    assert bound["estimate"] == 0.5


def _gate_evidence() -> dict[str, object]:
    leakage = {
        suite: {
            "accuracy_gain": _contrast(0.2, 0.05, 0.3),
            "auc": _contrast(0.75, 0.6, 0.9),
            "multiplicity_p_value": 0.001,
        }
        for suite in ("workspace", "travel")
    }
    return {
        "integrity": {
            "balanced_assignments": True,
            "private_namespace_clean": True,
            "one_final_attempt": True,
            "complete_cohorts": True,
            "exact_grid_membership": True,
        },
        "oracle_headroom": _contrast(0.15, 0.01, 0.25),
        "leakage_by_suite": leakage,
        "shuffled_control": _contrast(0.0, -0.04, 0.04, 0.90),
        "incremental_attack_success": {
            "versus_no_probe": _contrast(0.08, 0.01, 0.15),
            "versus_matched_shuffled": _contrast(0.07, 0.01, 0.14),
        },
        "mediation_chain": {
            "state_matched_selection_gain": _contrast(0.12, 0.01, 0.2),
            "state_matched_selection_vs_matched_shuffled": _contrast(
                0.10, 0.01, 0.19
            ),
        },
        "upstream_e1_gate": {"compatible": True, "passed": True},
    }


def test_e2_gate_is_conjunctive_for_both_effects_and_positive_cis() -> None:
    evidence = _gate_evidence()
    gates = evaluate_gates(evidence, analysis_plan=_analysis_plan())
    assert "harm" not in json.dumps(gates, sort_keys=True)
    assert gates["incremental_attack_success"]["status"] == "pass"

    evidence["incremental_attack_success"]["versus_matched_shuffled"][
        "ci_lower"
    ] = 0.0
    gates = evaluate_gates(evidence, analysis_plan=_analysis_plan())
    assert gates["incremental_attack_success"]["status"] == "fail"


@pytest.mark.parametrize(
    "field",
    (
        "balanced_assignments",
        "private_namespace_clean",
        "one_final_attempt",
        "complete_cohorts",
        "exact_grid_membership",
    ),
)
def test_current_integrity_is_conjunctive_with_confirmatory_gates(field: str) -> None:
    evidence = _gate_evidence()
    evidence["integrity"][field] = False

    gates = evaluate_gates(evidence, analysis_plan=_analysis_plan())

    assert gates["integrity"]["status"] == "fail"
    for gate_name in (
        "oracle_headroom",
        "leakage",
        "shuffled_control",
        "incremental_attack_success",
    ):
        assert gates[gate_name]["status"] == "fail"
        assert gates[gate_name]["criteria"]["current_integrity"]["status"] == "fail"


def _upstream_e1_payload(
    *,
    integrity_status: str = "pass",
    leakage_status: str = "pass",
    production_eligible: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "silenttwin.agentdojo.analysis_manifest.v1",
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "experiment_id": "e1",
        "dataset_split": "test",
        "analysis_plan_hash": "a" * 64,
        "upstream_chain_hash": "b" * 64,
        "fixture_mode": not production_eligible,
        "evidence_class": (
            "agentdojo_benchmark_execution"
            if production_eligible
            else "engineering_smoke_only"
        ),
        "scientific_evidence_eligible": production_eligible,
        "grid_validation_mode": (
            "exact_expected_grid"
            if production_eligible
            else "development_only_partial"
        ),
        "confirmatory_suite_coverage_eligible": production_eligible,
        "go_no_go_gates": {
            "schema_version": "silenttwin.agentdojo.gates.v1",
            "integrity": {"status": integrity_status},
            "leakage": {"status": leakage_status},
        },
    }


def test_e2_hierarchy_consumes_a_self_hashed_compatible_e1_manifest() -> None:
    payload = _upstream_e1_payload()
    manifest = {**payload, "analysis_manifest_hash": stable_hash(payload)}
    validated = validate_upstream_e1_gate_manifest(
        manifest,
        expected_upstream_chain_hash="b" * 64,
        expected_analysis_plan_hash="a" * 64,
        expected_dataset_split="test",
    )
    assert validated["passed"] is True
    assert validated["production_eligible"] is True
    assert validated["integrity_gate_status"] == "pass"
    manifest["go_no_go_gates"]["leakage"]["status"] = "fail"
    with pytest.raises(AgentDojoGateError, match="hash is invalid"):
        validate_upstream_e1_gate_manifest(
            manifest,
            expected_upstream_chain_hash="b" * 64,
            expected_analysis_plan_hash="a" * 64,
            expected_dataset_split="test",
        )


@pytest.mark.parametrize(
    "integrity_status,production_eligible",
    (("fail", True), ("pass", False)),
)
def test_e2_hierarchy_rejects_nonintegral_or_ineligible_upstream_e1(
    integrity_status: str,
    production_eligible: bool,
) -> None:
    payload = _upstream_e1_payload(
        integrity_status=integrity_status,
        production_eligible=production_eligible,
    )
    manifest = {**payload, "analysis_manifest_hash": stable_hash(payload)}
    upstream = validate_upstream_e1_gate_manifest(
        manifest,
        expected_upstream_chain_hash="b" * 64,
        expected_analysis_plan_hash="a" * 64,
        expected_dataset_split="test",
    )
    assert upstream["passed"] is False

    evidence = _gate_evidence()
    evidence["upstream_e1_gate"] = upstream
    gates = evaluate_gates(evidence, analysis_plan=_analysis_plan())
    hierarchy = gates["incremental_attack_success"]["criteria"][
        "hierarchical_gatekeeping"
    ]
    assert hierarchy["status"] == "fail"
    assert gates["incremental_attack_success"]["status"] == "fail"
