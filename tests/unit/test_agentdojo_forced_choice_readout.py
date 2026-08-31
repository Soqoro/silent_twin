from __future__ import annotations

from copy import deepcopy

import pytest

from silenttwin.agentdojo.config import stable_hash
from silenttwin.agentdojo.forced_choice_readout import (
    ANSWER_TOKEN_IDS,
    CONTEXT_ORDER,
    SURFACE_ORDER,
    _auc_summary,
    _context_auc_contrast,
    audit_answer_tokens,
    build_prompt,
    collapse_surfaces,
    expanded_jobs,
    validate_protocol,
)


def _protocol() -> dict:
    payload = {
        "schema_version": "silenttwin.agentdojo.forced_choice_readout_protocol.v1",
        "access_policy": {
            "execution_permitted_splits": ["train"],
            "development_outcomes_inspected": False,
            "test_outcomes_inspected": False,
            "development_submission_permitted": False,
            "held_out_evaluation_permitted": False,
            "confirmatory_claim_permitted": False,
        },
        "design": {
            "context_order": list(CONTEXT_ORDER),
            "surface_order": list(SURFACE_ORDER),
            "expected_input_records": 744,
            "expected_model_calls": 2976,
        },
        "model": {
            "answer_token_ids": dict(ANSWER_TOKEN_IDS),
            "scoring_mode": "next_token_forced_choice",
        },
        "analysis": {
            "independent_unit": "structural_group_id",
            "suite_weighting": "equal_suite",
            "bootstrap_resamples": 5000,
            "minimum_paired_readout_validity": 1.0,
            "chance_auc_equivalence_margin": [0.45, 0.55],
        },
    }
    return {**payload, "protocol_hash": stable_hash(payload)}


def _record() -> dict:
    public = {
        "protocol": "silenttwin.controlled-json.v1",
        "request_type": "hidden_state_prediction",
        "required_output_contract": {
            "prediction": "theta0 or theta1",
            "posterior": {"theta0": "number", "theta1": "number"},
        },
        "scenario": {
            "candidate_plans": [
                {
                    "label": "candidate_0",
                    "plan_id": "candidate_0",
                    "description": "candidate strategy 0",
                },
                {
                    "label": "candidate_1",
                    "plan_id": "candidate_1",
                    "description": "candidate strategy 1",
                },
            ],
            "tool_schemas": [{"name": "public_tool"}],
        },
        "transcript": [
            {
                "event_type": "probe_feedback",
                "plan_id": "probe_candidate_0",
                "payload": {"status": "completed"},
            }
        ],
    }
    return {
        "input_id": "input-0",
        "upstream_input_id": "upstream-0",
        "upstream_trial_id": "trial-0",
        "suite": "travel",
        "structural_group_id": "group-0",
        "scenario_id": "scenario-0",
        "feedback_source": "genuine",
        "actual_state": "theta1",
        "donor_state": None,
        "prediction_seed": 7,
        "canonical_public_input": public,
    }


def test_protocol_is_self_hashed_and_train_only() -> None:
    protocol = _protocol()

    assert validate_protocol(protocol) == protocol["protocol_hash"]
    assert protocol["access_policy"]["execution_permitted_splits"] == ["train"]
    assert protocol["access_policy"]["held_out_evaluation_permitted"] is False


def test_job_expansion_pairs_two_contexts_and_two_surface_permutations() -> None:
    jobs = expanded_jobs([_record()])

    assert len(jobs) == 4
    assert {(job["context"], job["surface"]) for job in jobs} == {
        (context, surface)
        for context in CONTEXT_ORDER
        for surface in SURFACE_ORDER
    }
    assert len({job["job_id"] for job in jobs}) == 4


def test_prompts_relabel_candidates_without_exposing_trusted_state() -> None:
    record = _record()

    minimal_ab, mapping_ab = build_prompt(
        record, context="minimal_transcript", surface="ab"
    )
    full_ba, mapping_ba = build_prompt(
        record, context="full_scenario", surface="ba"
    )

    assert mapping_ab == {"candidate_0": "A", "candidate_1": "B"}
    assert mapping_ba == {"candidate_0": "B", "candidate_1": "A"}
    assert "probe_A" in minimal_ab
    assert "probe_B" in full_ba
    assert "public_tool" not in minimal_ab
    assert "public_tool" in full_ba
    for prompt in (minimal_ab, full_ba):
        assert "candidate_0" not in prompt
        assert "candidate_1" not in prompt
        assert "theta0" not in prompt
        assert "theta1" not in prompt
        assert "actual_state" not in prompt
        assert "donor_state" not in prompt


def _surface_row(surface: str, theta1_probability: float, raw_b: float) -> dict:
    return {
        "input_id": "input-0",
        "context": "minimal_transcript",
        "surface": surface,
        "suite": "travel",
        "structural_group_id": "group-0",
        "scenario_id": "scenario-0",
        "feedback_source": "genuine",
        "actual_state": "theta1",
        "donor_state": None,
        "model_error": None,
        "readout_valid": True,
        "theta1_probability": theta1_probability,
        "conditional_probabilities": {"A": 1.0 - raw_b, "B": raw_b},
        "candidate_probability_mass": 0.8,
        "greedy_allowed": True,
    }


def test_surface_collapse_maps_back_then_averages() -> None:
    rows = [
        _surface_row("ab", theta1_probability=0.8, raw_b=0.8),
        _surface_row("ba", theta1_probability=0.6, raw_b=0.4),
    ]

    collapsed = collapse_surfaces(rows)

    assert len(collapsed) == 1
    assert collapsed[0]["theta1_probability"] == pytest.approx(0.7)
    assert collapsed[0]["surface_absolute_difference"] == pytest.approx(0.2)
    assert collapsed[0]["raw_b_probability_mean"] == pytest.approx(0.6)
    assert collapsed[0]["paired_readout_valid"] == 1.0


class _TokenAuditTokenizer:
    chat_template = "fixture-template"

    def encode(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        return {"A": [32], "B": [33]}[value]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        if add_generation_prompt:
            assert len(messages) == 1
            return "PREFIX"
        assert len(messages) == 2
        return "PREFIX" + messages[-1]["content"] + "END"

    def __call__(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        suffix = value.removeprefix("PREFIX")
        ids = [1, 2]
        if suffix:
            ids.extend([{"A": 32, "B": 33}[suffix[0]], 9])
        return {"input_ids": ids}


def test_token_audit_binds_single_first_assistant_tokens() -> None:
    audit = audit_answer_tokens(_TokenAuditTokenizer())

    assert audit["answer_token_ids"] == {"A": 32, "B": 33}
    payload = deepcopy(audit)
    recorded = payload.pop("token_audit_hash")
    assert recorded == stable_hash(payload)


def test_clustered_auc_and_paired_context_contrast_use_mapped_scores() -> None:
    rows = []
    for suite in ("workspace", "travel", "banking", "slack"):
        for context in CONTEXT_ORDER:
            for state, score in (("theta0", 0.1), ("theta1", 0.9)):
                rows.append(
                    {
                        "input_id": f"{suite}:{context}:{state}",
                        "context": context,
                        "suite": suite,
                        "structural_group_id": f"{suite}-group",
                        "scenario_id": f"{suite}-scenario",
                        "feedback_source": "genuine",
                        "actual_state": state,
                        "donor_state": None,
                        "theta1_probability": score,
                    }
                )

    minimal = [row for row in rows if row["context"] == "minimal_transcript"]
    auc = _auc_summary(
        minimal,
        label_field="actual_state",
        seed=7,
        resamples=20,
        confidence=0.95,
    )
    contrast = _context_auc_contrast(
        rows,
        source="genuine",
        label_field="actual_state",
        seed=11,
        resamples=20,
    )

    assert auc["estimate"] == 1.0
    assert all(value["estimate"] == 1.0 for value in auc["by_suite"].values())
    assert contrast["estimate"] == 0.0
    assert contrast["ci_95"] == [0.0, 0.0]
