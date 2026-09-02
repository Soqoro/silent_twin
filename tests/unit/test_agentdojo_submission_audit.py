from __future__ import annotations

import json

import pytest

from silenttwin.agentdojo.submission_audit import (
    ARTIFACT_SPECS,
    CLAIM_BINDINGS,
    MANUSCRIPT_RESULT_FRAGMENTS,
    SubmissionAuditError,
    _Checks,
    _same_value,
    classify_native_failure,
    extract_top_level_errors,
    manuscript_claim_audit,
    native_repair_taxonomy,
    strict_repair_taxonomy,
)


def test_expected_artifacts_are_exact_train_only_sha256_bindings() -> None:
    assert len(ARTIFACT_SPECS) == 13
    assert len({row.artifact_id for row in ARTIFACT_SPECS}) == 13
    assert len({row.relative_path for row in ARTIFACT_SPECS}) == 13
    for row in ARTIFACT_SPECS:
        assert len(row.sha256) == 64
        int(row.sha256, 16)
        assert not row.relative_path.startswith("/")
        assert "development" not in row.relative_path
        assert "/test" not in row.relative_path


def test_numeric_comparison_is_exact_except_for_float_roundoff() -> None:
    assert _same_value(0.1 + 0.2, 0.3)
    assert not _same_value(0.300001, 0.3)
    assert not _same_value(True, 1.0)
    assert _same_value({"a": 1}, {"a": 1})


def test_extract_top_level_errors_avoids_parsing_large_nested_fields() -> None:
    row = {
        "earlier": {"errors": ["nested"], "payload": "x" * 1000},
        "errors": [
            {
                "code": "invalid_hidden_state_prediction",
                "message": "contains a quoted \\\"errors\\\" token",
            }
        ],
        "evidence_class": "agentdojo_benchmark_execution",
        "later": [1, 2, 3],
    }
    # Production rows are canonical and therefore sorted by top-level key.
    line = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert extract_top_level_errors(line) == row["errors"]


@pytest.mark.parametrize(
    "line",
    [
        b'{"evidence_class":"x"}\n',
        b'{"errors":{},"evidence_class":"x"}\n',
        b'{"errors":[}\n',
    ],
)
def test_extract_top_level_errors_fails_closed(line: bytes) -> None:
    with pytest.raises(SubmissionAuditError):
        extract_top_level_errors(line)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "NativeToolInterfaceError:native tool-call payload is not strict JSON",
            "malformed_tool_call_json",
        ),
        (
            "ValueError: File with ID 'invented' not found.",
            "invalid_entity_identifier",
        ),
        (
            "ValueError: Event with ID 'invented' not found.",
            "invalid_entity_identifier",
        ),
        (
            "ValueError: Transaction with ID 5 not found.",
            "invalid_entity_identifier",
        ),
        (
            "ValueError: Channel External not found in the channels list",
            "invalid_entity_identifier",
        ),
        ("ValueError: No files found with the given query", "unsuccessful_search"),
        ("ValueError: No emails found.", "unsuccessful_search"),
        ("value is not a valid email address", "invalid_email_argument"),
        ("channel Field required [type=missing]", "missing_required_argument"),
        ("something new", "unknown"),
    ],
)
def test_native_failure_classifier(message: str, expected: str) -> None:
    assert classify_native_failure(message) == expected


def test_strict_repair_taxonomy_uses_only_learned_episodes() -> None:
    rows = [
        {"learned_arm": False, "tool_loop_error": None},
        *[
            {
                "learned_arm": True,
                "tool_loop_error": (
                    'PipelineProtocolError:tool-loop output must be exactly '
                    '{"content","tool_calls"}'
                ),
            }
            for _ in range(2)
        ],
        {
            "learned_arm": True,
            "tool_loop_error": "PipelineProtocolError:tool-loop output is not JSON",
        },
        {
            "learned_arm": True,
            "tool_loop_error": (
                "PipelineProtocolError:a tool-call turn cannot also contain final content"
            ),
        },
    ]
    assert strict_repair_taxonomy(rows)["categories"] == {
        "missing_required_content_or_exact_keys": 2,
        "non_json_fenced": 1,
        "mixed_tool_calls_and_final_content": 1,
    }


def test_strict_repair_taxonomy_rejects_a_new_failure_class() -> None:
    with pytest.raises(SubmissionAuditError, match="unknown strict-repair"):
        strict_repair_taxonomy(
            [{"learned_arm": True, "tool_loop_error": "unexpected"}]
        )


def test_native_taxonomy_keeps_protocol_and_prompt_binding_distinct() -> None:
    rows = [
        {
            "model_protocol_valid": False,
            "prompt_binding_valid": True,
            "tool_loop_error": "ValueError: File with ID 'x' not found.",
            "agentdojo_suite": "workspace",
        },
        {
            "model_protocol_valid": True,
            "prompt_binding_valid": False,
            "tool_loop_error": None,
            "agentdojo_suite": "travel",
        },
    ]
    result = native_repair_taxonomy(rows)
    assert result["invalid_episode_count"] == 1
    assert result["categories"] == {"invalid_entity_identifier": 1}
    assert result["prompt_binding_failure_count"] == 1
    assert result["prompt_binding_failure_suites"] == {"travel": 1}
    assert result["otherwise_protocol_valid_prompt_binding_failures"] == 1


def _bound_manuscript() -> str:
    return "\n".join(
        [
            r"\begin{abstract}",
            *(str(row["abstract_fragment"]) for row in CLAIM_BINDINGS),
            r"\end{abstract}",
            r"\subsection{Frozen Train-Only Claim Ledger}",
            *(str(row["ledger_fragment"]) for row in CLAIM_BINDINGS),
            r"\subsection{Feedback Policies, Attackers, and Baselines}",
            r"\section{Conclusion}",
            *(str(row["conclusion_fragment"]) for row in CLAIM_BINDINGS),
            r"\begin{thebibliography}{1}",
        ]
    )


def test_manuscript_claim_audit_maps_abstract_conclusion_and_ledger() -> None:
    checks = _Checks()
    bindings = manuscript_claim_audit(_bound_manuscript(), checks)
    assert len(bindings) == len(CLAIM_BINDINGS) == 6
    assert all(check["passed"] for check in checks.rows)


def test_manuscript_claim_audit_detects_a_missing_conclusion_binding() -> None:
    checks = _Checks()
    missing = str(CLAIM_BINDINGS[0]["conclusion_fragment"])
    manuscript_claim_audit(_bound_manuscript().replace(missing, ""), checks)
    failed = [row for row in checks.rows if not row["passed"]]
    assert [row["check_id"] for row in failed] == [
        "manuscript.claim.source_aligned_private_state_information.conclusion"
    ]


def test_result_fragments_are_unique_and_cover_each_empirical_stage() -> None:
    assert len(MANUSCRIPT_RESULT_FRAGMENTS) == len(set(MANUSCRIPT_RESULT_FRAGMENTS))
    joined = "\n".join(MANUSCRIPT_RESULT_FRAGMENTS)
    for marker in ("8,928", "2,976", "4,836", "0.7443", "14 invalid"):
        assert marker in joined
