from __future__ import annotations

import json
from pathlib import Path

from silenttwin.agentdojo.interface_realization import (
    ARM_ORDER,
    SOURCE_ARMS,
    _minimal_public_input,
    _prompt_from_public_input,
    build_prompt,
    expanded_jobs,
    length_matched_prompt,
    opaque_label_mapping,
    parse_response,
    rendered_token_count,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / (
    "configs/silenttwin/agentdojo/"
    "scientific-v6-interface-realization-train-v1.json"
)


def _record(source: str = "genuine", *, input_id: str = "input-0") -> dict:
    public = {
        "protocol": "silenttwin.controlled-json.v1",
        "request_type": "hidden_state_prediction",
        "required_output_contract": {},
        "scenario": {"tool_schemas": [{"name": "private_free_tool"}]},
        "transcript": [
            {
                "event_type": "probe_feedback",
                "payload": {"status": "completed"},
                "plan_id": "probe_candidate_0",
                "probe_index": 0,
                "timing_class": "normalized",
            }
        ],
    }
    return {
        "input_id": input_id,
        "upstream_trial_id": f"trial-{input_id}",
        "suite": "travel",
        "structural_group_id": "group-public",
        "scenario_id": "scenario-public",
        "feedback_source": source,
        "actual_state": "theta1",
        "donor_state": "theta0" if source == "matched_shuffled" else None,
        "prediction_seed": 7,
        "original_input_tokens": 500,
        "canonical_public_input": public,
        "original_protocol_prompt": "ORIGINAL",
        "original_raw_response_hash": "a" * 64,
    }


def test_checked_protocol_is_self_hashed_and_train_only() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert validate_protocol(protocol) == protocol["protocol_hash"]
    assert protocol["access_policy"]["execution_permitted_splits"] == ["train"]
    assert protocol["access_policy"]["development_submission_permitted"] is False
    assert protocol["design"]["expected_model_calls"] == 1860


def test_source_arm_expansion_is_fixed_and_paired() -> None:
    records = [
        _record("genuine", input_id="g"),
        _record("matched_shuffled", input_id="m"),
        _record("constant", input_id="c"),
    ]

    jobs = expanded_jobs(records)

    assert len(jobs) == len(ARM_ORDER) + 2
    assert {job["arm"] for job in jobs if job["input_id"] == "g"} == set(ARM_ORDER)
    assert {job["arm"] for job in jobs if job["input_id"] == "m"} == set(
        SOURCE_ARMS["matched_shuffled"]
    )
    assert len({job["job_id"] for job in jobs}) == len(jobs)


def test_minimal_prompt_retains_transcript_but_excludes_trusted_assignment() -> None:
    record = _record()

    prompt, mapping = build_prompt(record, "minimal_explicit_exact")

    assert mapping == {"theta0": "theta0", "theta1": "theta1"}
    assert "probe_candidate_0" in prompt
    assert "private_free_tool" not in prompt
    assert "actual_state" not in prompt
    assert "donor_state" not in prompt


def test_opaque_mapping_is_public_group_deterministic() -> None:
    first = opaque_label_mapping("same-public-group")
    second = opaque_label_mapping("same-public-group")

    assert first == second
    assert set(first) == {"theta0", "theta1"}
    assert set(first.values()) == {"state_lantern", "state_kestrel"}


def test_parser_separates_posterior_from_namespace_contract_failure() -> None:
    raw = json.dumps(
        {
            "prediction": "candidate_1",
            "posterior": {"theta0": 0.2, "theta1": 0.8},
        }
    )

    parsed = parse_response(
        raw,
        label_mapping={"theta0": "theta0", "theta1": "theta1"},
        posterior_only=False,
        actual_state="theta1",
    )

    assert parsed["contract_valid"] is False
    assert parsed["posterior_valid"] is True
    assert parsed["posterior_classification_correct"] is True
    assert parsed["namespace_substitution"] is True
    assert parsed["conservative_state_correct"] is False


def test_posterior_only_opaque_contract_maps_back_for_scoring() -> None:
    mapping = {"theta0": "state_kestrel", "theta1": "state_lantern"}
    raw = json.dumps(
        {"posterior": {"state_kestrel": 0.1, "state_lantern": 0.9}}
    )

    parsed = parse_response(
        raw,
        label_mapping=mapping,
        posterior_only=True,
        actual_state="theta1",
    )

    assert parsed["contract_valid"] is True
    assert parsed["posterior_theta1"] == 0.9
    assert parsed["conservative_state_correct"] is True


class _CharacterTokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, rendered, *, add_special_tokens=True):
        assert add_special_tokens is True
        return {"input_ids": list(rendered)}


def test_length_padding_matches_rendered_token_count_exactly() -> None:
    record = _record()
    tokenizer = _CharacterTokenizer()
    public = _minimal_public_input(
        record,
        labels=("theta0", "theta1"),
        explicit_semantics=True,
        posterior_only=False,
    )
    unpadded = _prompt_from_public_input(public)
    target = rendered_token_count(tokenizer, unpadded) + 240

    padded = length_matched_prompt(public, target_tokens=target, tokenizer=tokenizer)

    assert rendered_token_count(tokenizer, padded) == target
    assert "irrelevant_schema_padding" in padded
