"""Offline integration contract for the exact AgentDojo Tier-2 environment."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import sys

import pytest


try:
    _AGENTDOJO_VERSION = metadata.version("agentdojo")
except metadata.PackageNotFoundError:
    _AGENTDOJO_VERSION = None

_PINNED_RUNTIME = sys.version_info[:2] == (3, 11) and _AGENTDOJO_VERSION == "0.1.35"
pytestmark = pytest.mark.skipif(
    not _PINNED_RUNTIME,
    reason="requires the locked Python 3.11 / agentdojo==0.1.35 Tier-2 runtime",
)


@pytest.fixture(scope="module")
def frozen_documents() -> tuple[dict[str, object], dict[str, object]]:
    from silenttwin.agentdojo.catalog import build_catalog
    from silenttwin.agentdojo.splits import build_split_manifest

    catalog = build_catalog()
    return catalog, build_split_manifest(catalog)


def test_release_pin_public_api_and_resolved_suite_versions() -> None:
    from silenttwin.agentdojo.compat import (
        EXPECTED_ATTACKS,
        EXPECTED_DEFENSES,
        EXPECTED_INTERNAL_BENCHMARK_VERSIONS,
        EXPECTED_RELEASE_COUNTS,
        assert_compatible,
    )

    report = assert_compatible()
    assert report.package_version == "0.1.35"
    assert report.source_revision == "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"
    assert report.benchmark_version == "v1.2.2"
    assert report.defenses == EXPECTED_DEFENSES
    assert report.attacks == EXPECTED_ATTACKS
    for suite in report.suites:
        assert suite.benchmark_version == EXPECTED_INTERNAL_BENCHMARK_VERSIONS[suite.name]
        assert {
            "user_tasks": suite.user_task_count,
            "injection_tasks": suite.injection_task_count,
            "tools": suite.tool_count,
            "vectors": suite.injection_vector_count,
        } == EXPECTED_RELEASE_COUNTS[suite.name]


def test_lock_and_python_pin_match_the_deployed_environment() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert (repository / "configs/silenttwin/agentdojo/python-version.txt").read_text(
        encoding="utf-8"
    ).strip() == "3.11"
    lock_lines = [
        line.strip()
        for line in (repository / "requirements-tier2-agentdojo.lock").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # agentdojo's declared pydantic[email] extra makes this 71, not 69.
    assert len(lock_lines) == 71
    for line in lock_lines:
        name, expected = line.split("==", 1)
        assert metadata.version(name) == expected


def test_complete_catalog_is_deterministic_and_exposure_derived(
    frozen_documents: tuple[dict[str, object], dict[str, object]],
) -> None:
    from silenttwin.agentdojo.catalog import build_catalog
    from silenttwin.agentdojo.compat import (
        get_injection_candidates,
        get_user_task,
        load_suite,
    )

    catalog, _ = frozen_documents
    repeated = build_catalog()
    assert repeated["catalog_hash"] == catalog["catalog_hash"]
    assert catalog["eligible_combination_count"] == 1467
    assert catalog["eligible_combination_count_by_suite"] == {
        "workspace": 840,
        "banking": 144,
        "slack": 140,
        "travel": 343,
    }
    resolved = catalog["resolved_suites"]
    assert resolved["workspace"]["unexposed_default_vectors"] == [
        "calendar_fake_deadline_description",
        "email_events_follow_up_injection",
    ]
    assert resolved["slack"]["unexposed_default_vectors"] == ["injection_phishing_1"]

    # Re-run the released exposure method and compare it with every frozen
    # per-user candidate set.  No default vector is assumed reachable.
    for suite_name, suite_record in resolved.items():
        suite = load_suite(suite_name)
        recorded = suite_record["candidate_vectors_by_user_task"]
        for user_id, vector_ids in recorded.items():
            user_task = get_user_task(suite, user_id)
            assert sorted(get_injection_candidates(suite, user_task)) == vector_ids


def test_split_manifest_exhaustive_and_entity_disjoint(
    frozen_documents: tuple[dict[str, object], dict[str, object]],
) -> None:
    from silenttwin.agentdojo.splits import validate_split_manifest

    catalog, splits = frozen_documents
    validate_split_manifest(splits, catalog=catalog)
    seen_users: set[tuple[str, str]] = set()
    seen_injections: set[tuple[str, str]] = set()
    for split_name in ("train", "development", "test"):
        entry = splits["splits"][split_name]
        current_users = {
            (suite, task_id)
            for suite, identifiers in entry["user_task_ids_by_suite"].items()
            for task_id in identifiers
        }
        current_injections = {
            (suite, task_id)
            for suite, identifiers in entry["injection_task_ids_by_suite"].items()
            for task_id in identifiers
        }
        assert not seen_users & current_users
        assert not seen_injections & current_injections
        seen_users.update(current_users)
        seen_injections.update(current_injections)


def test_released_ground_truth_pipeline_and_normalized_graders() -> None:
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.task_suite import (
        functions_stack_trace_from_messages,
        model_output_from_messages,
    )
    from silenttwin.agentdojo.compat import (
        clone_environment,
        get_injection_task,
        get_user_task,
        grade_attack_success,
        grade_utility,
        load_environment,
        load_suite,
    )
    from silenttwin.agentdojo.config import AGENTDOJO_SUITES

    for suite_name in AGENTDOJO_SUITES:
        suite = load_suite(suite_name)
        user = get_user_task(suite, sorted(suite.user_tasks)[0])
        injection = get_injection_task(suite, sorted(suite.injection_tasks)[0])

        environment = load_environment(suite, user, {})
        pre_environment = clone_environment(environment)
        _, _, post_environment, messages, _ = GroundTruthPipeline(user).query(
            user.PROMPT, FunctionsRuntime(suite.tools), environment
        )
        output = model_output_from_messages(messages)
        traces = functions_stack_trace_from_messages(messages)
        assert grade_utility(user, output or [], pre_environment, post_environment, traces)

        environment = load_environment(suite, user, {})
        pre_environment = clone_environment(environment)
        _, _, post_environment, messages, _ = GroundTruthPipeline(injection).query(
            user.PROMPT, FunctionsRuntime(suite.tools), environment
        )
        output = model_output_from_messages(messages)
        traces = functions_stack_trace_from_messages(messages)
        # This True is attack success, not safety.
        assert grade_attack_success(
            injection, output or [], pre_environment, post_environment, traces
        )


def test_deterministic_fake_pipeline_and_single_call_facade() -> None:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.types import ChatAssistantMessage, text_content_block_from_string
    from silenttwin.agentdojo.compat import (
        clone_environment,
        environment_hash,
        execute_call,
        get_user_task,
        load_environment,
        load_suite,
    )

    class DeterministicNoOpPipeline(BasePipelineElement):
        def query(self, query, runtime, env, messages=(), extra_args=None):
            final = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("deterministic fake")],
                tool_calls=None,
            )
            return query, runtime, env, [*messages, final], dict(extra_args or {})

    suite = load_suite("workspace")
    user = get_user_task(suite, "user_task_0")
    starts = []
    results = []
    for _ in range(2):
        environment = load_environment(suite, user, {})
        starts.append(environment_hash(environment))
        results.append(
            suite.run_task_with_pipeline(
                DeterministicNoOpPipeline(), user, None, {}, environment=environment
            )
        )
    assert starts[0] == starts[1]
    assert results[0] == results[1]
    # The clean-run second result is AgentDojo's sentinel True, not an attack
    # result; the facade only exposes injection grading when a task is supplied.
    assert results[0][1] is True

    environment = load_environment(suite, user, {})
    ground_truth_call = user.ground_truth(clone_environment(environment))[0]
    outcome = execute_call(suite, environment, ground_truth_call)
    assert outcome.error is None
    assert len(outcome.trace) == 1
    assert len(outcome.pre_environment_hash) == 64
    assert len(outcome.post_environment_hash) == 64
