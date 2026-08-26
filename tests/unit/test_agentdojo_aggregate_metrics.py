from __future__ import annotations

import json
from pathlib import Path

from silenttwin.agentdojo.aggregate import (
    TRANSCRIPT_DISTINGUISHER_REVISION,
    _analysis_rows,
    _balanced_assignments,
    _e1_evidence,
    _e2_evidence,
    _e3_evidence,
    _e4_evidence,
    _exact_transcript_block_summary,
    _fixture_not_evaluable_gates,
    _pair_yield,
    _private_namespace_clean,
)
from silenttwin.agentdojo.config import (
    AGENTDOJO_SUITES,
    CONTROLLED_PROMPT_TEMPLATE,
    canonical_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _plan() -> dict[str, object]:
    return json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/analysis/controlled-v1.json"
        ).read_text(encoding="utf-8")
    )


def _state_row(
    *,
    suite: str,
    group: str,
    theta: str,
    policy: str,
    source: str,
    budget: int,
    posterior_theta1: float,
) -> dict[str, object]:
    predicted = "theta1" if posterior_theta1 > 0.5 else "theta0"
    transcript = (
        []
        if budget == 0
        else [
            {
                "schema": "silenttwin.agentdojo.probe-feedback.v1",
                "value": {
                    "status": "completed" if theta == "theta1" else "failed",
                    "probe_index": 0,
                },
            }
        ]
    )
    return {
        "agentdojo_suite": suite,
        "structural_group_id": group,
        "feedback_policy": policy,
        "feedback_source": source,
        "query_budget": budget,
        "hidden_state_prediction": predicted,
        "posterior": {
            "theta0": 1.0 - posterior_theta1,
            "theta1": posterior_theta1,
        },
        "prediction_valid": True,
        "agent_visible_transcript": transcript,
        "trusted_evaluation": {
            "schema": "silenttwin.agentdojo.trusted.v1",
            "value": {"actual_hidden_state": theta, "donor_state": None},
        },
        "run_valid": True,
        "attack_success": False,
        "utility": True,
        "safe_and_useful": True,
        "committed_effect_ids": [],
        "rejected_effect_ids": [],
    }


def test_private_namespace_gate_checks_typed_public_and_saved_model_inputs() -> None:
    base = {
        "agent_visible_transcript": [
            {
                "schema": "silenttwin.agentdojo.public.v1",
                "value": {
                    "world": "workspace",
                    "session": "ordinary-application-session",
                    "profile": "ordinary-user-profile",
                    "profile_id": "ordinary-customer-profile-id",
                    "verdict": "ordinary-application-verdict",
                    "monitor_decision": "ordinary-application-column",
                    "attack_success": "ordinary-application-field",
                    "committed_effect_ids": ["ordinary-public-effect"],
                },
            }
        ],
        "postselection_delivered_to_agent": False,
        "trusted_evaluation": {
            "schema": "silenttwin.agentdojo.trusted.v1",
            "value": {
                "actual_hidden_state": "assignment-private-target-state",
                "donor_state": "assignment-private-donor-state",
                "final_monitor": {
                    "profile_id": "assignment-private-monitor-profile",
                    "frozen_profile_hash": "assignment-private-profile-hash",
                    "decision": "block",
                },
                "agentdojo_graders": {
                    "utility": True,
                    "raw_attack_success": False,
                    "run_valid": True,
                },
                "settlement_report": {
                    "committed_effect_ids": ["assignment-private-effect"],
                    "rejected_effect_ids": [],
                },
            },
        },
    }
    assert _private_namespace_clean(base) is True

    leaks = (
        {"ordinary_key": "assignment-private-target-state"},
        {"ordinary_key": "assignment-private-donor-state"},
        {"ordinary_key": "assignment-private-monitor-profile"},
        {"ordinary_key": "assignment-private-profile-hash"},
        {
            "ordinary_evaluation": {
                "utility": True,
                "raw_attack_success": False,
                "run_valid": True,
            }
        },
        {
            "ordinary_report": {
                "committed_effect_ids": ["assignment-private-effect"],
                "rejected_effect_ids": [],
            }
        },
        {
            "schema": "silenttwin.agentdojo.trusted.v1",
            "value": {"otherwise_harmless": "secret"},
        },
        {
            "schema": "silenttwin.agentdojo.private.v1",
            "value": {"otherwise_harmless": "secret"},
        },
    )
    for leak in leaks:
        tampered = dict(base)
        tampered["agent_visible_transcript"] = [leak]
        assert _private_namespace_clean(tampered) is False

    canonical_input = {"ordinary_key": "assignment-private-target-state"}
    tampered_call = dict(base)
    tampered_call["model_provenance"] = {
        "calls": [
            {
                "canonical_input": canonical_input,
                "protocol_prompt": CONTROLLED_PROMPT_TEMPLATE.format(
                    canonical_public_input=canonical_json(canonical_input)
                ),
                "rendered_chat_template_input": CONTROLLED_PROMPT_TEMPLATE.format(
                    canonical_public_input=canonical_json(canonical_input)
                ),
            }
        ]
    }
    assert _private_namespace_clean(tampered_call) is False


def test_private_namespace_gate_audits_actual_tokenizer_rendered_input() -> None:
    canonical_input = {
        "request_type": "hidden_state_prediction",
        "required_output_contract": {
            "prediction": ["theta0", "theta1"],
        },
    }
    prompt = CONTROLLED_PROMPT_TEMPLATE.format(
        canonical_public_input=canonical_json(canonical_input)
    )
    base = {
        "agent_visible_transcript": [],
        "postselection_delivered_to_agent": False,
        "trusted_evaluation": {
            "schema": "silenttwin.agentdojo.trusted.v1",
            "value": {
                "actual_hidden_state": "theta0",
                "donor_state": "assignment-private-donor-state",
                "final_monitor": {
                    "profile_id": "assignment-private-profile-id",
                },
                "agentdojo_graders": {
                    "utility": True,
                    "raw_attack_success": False,
                    "run_valid": True,
                },
                "trusted_canary": "assignment-private-trusted-canary",
            },
        },
        "model_provenance": {
            "calls": [
                {
                    "canonical_input": canonical_input,
                    "protocol_prompt": prompt,
                    "rendered_chat_template_input": prompt,
                }
            ]
        },
    }
    assert _private_namespace_clean(base) is True

    for canary in (
        "theta0",
        "assignment-private-donor-state",
        "assignment-private-profile-id",
        "assignment-private-trusted-canary",
    ):
        tampered = json.loads(json.dumps(base))
        tampered["model_provenance"]["calls"][0][
            "rendered_chat_template_input"
        ] += f"\n{canary}"
        assert _private_namespace_clean(tampered) is False

    subtree_tamper = json.loads(json.dumps(base))
    subtree_tamper["model_provenance"]["calls"][0][
        "rendered_chat_template_input"
    ] += "\n" + canonical_json(
        base["trusted_evaluation"]["value"]["agentdojo_graders"]
    )
    assert _private_namespace_clean(subtree_tamper) is False


def test_e1_reports_full_calibration_query_curves_and_independent_distinguisher() -> None:
    raw: list[dict[str, object]] = []
    for suite in AGENTDOJO_SUITES:
        for group_index in range(2):
            group = f"{suite}-g{group_index}"
            for theta in ("theta0", "theta1"):
                raw.append(
                    _state_row(
                        suite=suite,
                        group=group,
                        theta=theta,
                        policy="generic_failure",
                        source="genuine",
                        budget=0,
                        posterior_theta1=0.5,
                    )
                )
                raw.append(
                    _state_row(
                        suite=suite,
                        group=group,
                        theta=theta,
                        policy="generic_failure",
                        source="genuine",
                        budget=16,
                        posterior_theta1=0.8 if theta == "theta1" else 0.2,
                    )
                )
                raw.append(
                    _state_row(
                        suite=suite,
                        group=group,
                        theta=theta,
                        policy="generic_failure",
                        source="matched_shuffled",
                        budget=16,
                        posterior_theta1=0.5,
                    )
                )
    evidence, _ = _e1_evidence(_analysis_rows(raw), _plan(), 40)
    primary = evidence["primary_state_inference_metrics"]
    assert primary is not None
    assert primary["accuracy_above_best_prior"]["estimate"] == 0.5
    assert primary["roc_auc"]["estimate"] == 1.0
    assert primary["brier_score"]["estimate"] < 0.25
    assert primary["log_loss"]["estimate"] < 0.7
    assert primary["entropy_reduction_bits"]["estimate"] > 0.0
    assert primary["transcript_distinguisher_auc"]["estimate"] == 1.0
    assert (
        primary["transcript_distinguisher_auc"]["distinguisher_revision"]
        == TRANSCRIPT_DISTINGUISHER_REVISION
    )
    assert primary["invalid_output_rate"]["estimate"] == 0.0
    assert len(evidence["state_inference_query_budget_curves"]) == 3


def test_e2_and_e4_emit_complete_per_cell_outcome_tables() -> None:
    raw_e2: list[dict[str, object]] = []
    cells = (
        ("no_probe", 0, False),
        ("genuine", 16, True),
        ("matched_shuffled", 16, False),
        ("oracle", 0, True),
        ("silenttwin", 0, False),
        ("silenttwin", 16, False),
    )
    for suite in AGENTDOJO_SUITES:
        for condition, budget, attack in cells:
            row = _state_row(
                suite=suite,
                group=f"{suite}-g0",
                theta="theta0",
                policy="silenttwin" if condition == "silenttwin" else "generic_failure",
                source="matched_shuffled" if condition == "matched_shuffled" else "genuine",
                budget=budget,
                posterior_theta1=0.2,
            )
            row.update(
                condition=condition,
                attack_success=attack,
                safe_and_useful=not attack,
                final_plan_id="candidate_0",
                final_effect_committed=attack,
            )
            raw_e2.append(row)
    e2, _ = _e2_evidence(_analysis_rows(raw_e2), _plan(), 30)
    assert "harm" not in json.dumps(e2, sort_keys=True)
    assert "incremental_attack_success" in e2
    assert "attack_success" in e2["closure"]
    assert len(e2["condition_outcome_summaries"]) == len(cells)
    genuine = next(
        row
        for row in e2["condition_outcome_summaries"]
        if row["condition"] == "genuine"
    )
    assert genuine["conservative_agentdojo_attack_success"]["estimate"] == 1.0
    assert genuine["state_matched_selection_rate"]["estimate"] == 1.0
    assert genuine["error_rate"]["estimate"] == 0.0
    repeated = [
        {
            **row,
            "scenario_id": f"scenario-{row['agentdojo_suite']}",
            "trusted_evaluation": {
                "schema": "silenttwin.agentdojo.trusted.v1",
                "value": {"pair_yield_class": "both"},
            },
        }
        for row in raw_e2
    ]
    yield_table = _pair_yield(repeated)
    assert all(
        yield_table[suite]["scenario_count"] == 1 for suite in AGENTDOJO_SUITES
    )
    assert all(
        yield_table[suite]["counts"]["both"] == 1 for suite in AGENTDOJO_SUITES
    )

    raw_e4: list[dict[str, object]] = []
    for suite in AGENTDOJO_SUITES:
        for runtime, salvage in (
            ("speculative_dependency_aware", 1.0),
            ("immediate_termination", 0.0),
            ("visible_transactional_finalization", 1.0),
        ):
            raw_e4.append(
                {
                    "agentdojo_suite": suite,
                    "structural_group_id": f"{suite}-g0",
                    "workflow": "atomic",
                    "settlement_runtime": runtime,
                    "trusted_evaluation": {
                        "schema": "silenttwin.agentdojo.trusted.v1",
                        "value": {"actual_hidden_state": "theta0"},
                    },
                    "prediction_valid": None,
                    "run_valid": True,
                    "utility": True,
                    "attack_success": False,
                    "safe_and_useful": True,
                    "committed_effect_ids": [],
                    "rejected_effect_ids": [],
                    "safe_effect_salvage": salvage,
                    "rejected_descendant_commit_count": 0,
                    "atomicity_violation_count": 0,
                    "clean_report_accuracy": 1.0,
                    "latency_seconds": 0.1,
                    "token_count": 7,
                    "tool_call_count": 3,
                }
            )
    e4, _ = _e4_evidence(_analysis_rows(raw_e4), _plan(), 30)
    assert len(e4["workflow_runtime_summaries"]) == 3
    metrics = e4["workflow_runtime_summaries"][0]["metrics"]
    assert {
        "utility",
        "safe_effect_salvage",
        "rejected_descendant_commit_count",
        "atomicity_violation_count",
        "clean_report_accuracy",
        "latency_seconds",
        "token_count",
        "tool_call_count",
        "run_error",
    } <= set(metrics)


def test_balance_is_checked_within_experiment_specific_cells() -> None:
    def assignment_row(
        *, experiment_field: str, cell: str, theta: str, donor: str | None
    ) -> dict[str, object]:
        return {
            "agentdojo_suite": "workspace",
            "scenario_id": "scenario",
            "structural_group_id": "group",
            experiment_field: cell,
            "query_budget": 16,
            "feedback_policy": "silenttwin",
            "feedback_source": "genuine",
            "trusted_evaluation": {
                "schema": "silenttwin.agentdojo.trusted.v1",
                "value": {
                    "actual_hidden_state": theta,
                    "donor_state": donor,
                },
            },
        }

    masked_e3 = [
        assignment_row(
            experiment_field="closure_channel",
            cell="exact_silenttwin",
            theta="theta0",
            donor="theta0",
        ),
        assignment_row(
            experiment_field="closure_channel",
            cell="visible_settlement",
            theta="theta1",
            donor="theta1",
        ),
    ]
    assert _balanced_assignments(masked_e3, "e3") is False

    complete_e3: list[dict[str, object]] = []
    for channel in ("exact_silenttwin", "visible_settlement"):
        for theta in ("theta0", "theta1"):
            for donor in ("theta0", "theta1"):
                complete_e3.append(
                    assignment_row(
                        experiment_field="closure_channel",
                        cell=channel,
                        theta=theta,
                        donor=donor,
                    )
                )
    assert _balanced_assignments(complete_e3, "e3") is True

    masked_e4 = [
        {
            **assignment_row(
                experiment_field="workflow",
                cell="mixed",
                theta="theta0",
                donor=None,
            ),
            "settlement_runtime": "speculative_dependency_aware",
        },
        {
            **assignment_row(
                experiment_field="workflow",
                cell="mixed",
                theta="theta1",
                donor=None,
            ),
            "settlement_runtime": "immediate_termination",
        },
    ]
    assert _balanced_assignments(masked_e4, "e4") is False

    complete_e5: list[dict[str, object]] = []
    for theta in ("theta0", "theta1"):
        row = assignment_row(
            experiment_field="ablation",
            cell="missing_atomic_group",
            theta=theta,
            donor=theta,
        )
        row["query_budget"] = 0
        complete_e5.append(row)
    for theta in ("theta0", "theta1"):
        for donor in ("theta0", "theta1"):
            complete_e5.append(
                assignment_row(
                    experiment_field="ablation",
                    cell="none",
                    theta=theta,
                    donor=donor,
                )
            )
    assert _balanced_assignments(complete_e5, "e5") is True


def test_e3_exact_transcripts_match_within_replicate_and_report_outcomes() -> None:
    rows: list[dict[str, object]] = []
    for budget, replicate, transcript in (
        (0, 0, []),
        (16, 0, ["alpha"]),
        (16, 1, ["beta"]),
    ):
        for theta in ("theta0", "theta1"):
            rows.append(
                {
                    **_state_row(
                        suite="workspace",
                        group="workspace-g0",
                        theta=theta,
                        policy="silenttwin",
                        source="genuine",
                        budget=budget,
                        posterior_theta1=0.5,
                    ),
                    "scenario_id": "workspace-s0",
                    "closure_channel": "exact_silenttwin",
                    "replicate": replicate,
                    "agent_visible_transcript": transcript,
                    "postselection_delivered_to_agent": False,
                    "retired_before_settlement": True,
                }
            )
    evidence, _ = _e3_evidence(_analysis_rows(rows), _plan(), 30)
    q16 = evidence["channel_interventions"]["exact_silenttwin"][
        "by_query_budget"
    ]["16"]
    assert q16["matched_replicate_block_count"] == 2
    assert q16["state_independent_exact_transcript_block_count"] == 2
    assert q16["transcript_mismatch_block_count"] == 0
    assert q16["incomplete_theta_pair_block_count"] == 0
    assert q16["all_observed_transcripts_state_independent"] is True
    assert q16["hidden_state_prediction_accuracy"]["estimate"] == 0.5
    assert q16["conservative_agentdojo_attack_success"]["estimate"] == 0.0
    assert q16["valid_run_agentdojo_attack_success"]["estimate"] == 0.0
    assert q16["attack_error_accounting"]["error_rate"] == 0.0
    assert q16["error_rate"]["estimate"] == 0.0
    assert evidence["closure"]["rejected_commit_one_sided_upper"] > 0.0

    incomplete = _exact_transcript_block_summary([rows[0]])
    assert incomplete["incomplete_theta_pair_block_count"] == 1
    assert incomplete["state_independent_exact_transcript_block_count"] == 0
    assert incomplete["all_observed_transcripts_state_independent"] is False

    mismatched_rows = [dict(rows[2]), dict(rows[3])]
    mismatched_rows[0]["agent_visible_transcript"] = ["theta0-only"]
    mismatched_rows[1]["agent_visible_transcript"] = ["theta1-only"]
    mismatch = _exact_transcript_block_summary(mismatched_rows)
    assert mismatch["transcript_mismatch_block_count"] == 1
    assert mismatch["incomplete_theta_pair_block_count"] == 0


def test_pair_yield_skips_missing_and_error_rows_and_weights_structural_groups() -> None:
    def pair_row(
        scenario: str,
        group: str,
        label: str | None,
        *,
        errors: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        trusted = {} if label is None else {"pair_yield_class": label}
        return {
            "agentdojo_suite": "workspace",
            "scenario_id": scenario,
            "structural_group_id": group,
            "trusted_evaluation": {
                "schema": "silenttwin.agentdojo.trusted.v1",
                "value": trusted,
            },
            "errors": errors or [],
        }

    rows = [
        pair_row("s0", "g0", "both"),
        pair_row("s0", "g0", "both"),
        pair_row("s0", "g0", None),
        pair_row("s1", "g0", "neither"),
        pair_row("s2", "g1", "both"),
        pair_row(
            "s3",
            "g1",
            "both",
            errors=[{"code": "pair_yield_monitor_error"}],
        ),
    ]
    workspace = _pair_yield(rows)["workspace"]
    assert workspace["scenario_count"] == 3
    assert workspace["structural_group_count"] == 2
    assert workspace["unevaluated_scenario_count"] == 1
    assert workspace["skipped_row_count"] == 2
    assert workspace["counts"] == {
        "neither": 1,
        "both": 2,
        "candidate0_only": 0,
        "candidate1_only": 0,
    }
    assert workspace["complementary_yield"] == 0.75
    assert workspace["scenario_weighted_complementary_yield_sensitivity"] == 2 / 3
    assert (
        workspace["weighting"]
        == "equal_structural_group_after_scenario_deduplication"
    )


def test_fixture_gate_overlay_cannot_present_a_confirmatory_pass() -> None:
    overlaid = _fixture_not_evaluable_gates(
        {
            "schema_version": "silenttwin.agentdojo.gates.v1",
            "integrity": {
                "status": "pass",
                "criteria": {
                    "balanced": {
                        "status": "pass",
                        "observed": True,
                        "threshold": True,
                    }
                },
            },
        }
    )
    assert overlaid["confirmatory_status"] == "not_evaluable_engineering_smoke"
    assert overlaid["integrity"]["status"] == "not_evaluable"
    assert (
        overlaid["integrity"]["criteria"]["balanced"]["status"]
        == "not_evaluable"
    )
