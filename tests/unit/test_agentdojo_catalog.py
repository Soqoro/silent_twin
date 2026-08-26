from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from silenttwin.agentdojo.catalog import build_catalog, validate_catalog
from silenttwin.agentdojo.config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
)
from silenttwin.agentdojo.scenarios import EXCLUDED_CROSS_SPLIT
from silenttwin.schemas import stable_digest


@dataclass
class _UserTask:
    ID: str
    PROMPT: str


@dataclass
class _InjectionTask:
    ID: str
    GOAL: str


class _Suite:
    def __init__(self, name: str) -> None:
        self.name = name
        self.benchmark_version = (1, 2, 2)
        self.user_tasks = {
            f"user_task_{index}": _UserTask(f"user_task_{index}", f"prompt {name} {index}")
            for index in range(6)
        }
        self.injection_tasks = {
            f"injection_task_{index}": _InjectionTask(
                f"injection_task_{index}", f"goal {name} {index}"
            )
            for index in range(5)
        }
        self.tools = (object(),)

    def get_injection_vector_defaults(self) -> dict[str, str]:
        return {"vector_0": "", "never_exposed": ""}


class _Metadata:
    def __init__(self, suite: _Suite) -> None:
        self.suite = suite

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.suite.name,
            "benchmark_version": [1, 2, 2],
            "user_task_count": 6,
            "injection_task_count": 5,
            "tool_count": 1,
            "injection_vector_count": 2,
        }


class _FakeCompat:
    def __init__(self) -> None:
        self.suites = {name: _Suite(name) for name in AGENTDOJO_SUITES}

    def assert_compatible(self, source: str, benchmark: str) -> dict[str, str]:
        assert source == AGENTDOJO_SOURCE_REVISION
        assert benchmark == AGENTDOJO_BENCHMARK_VERSION
        return {
            "package_version": AGENTDOJO_PACKAGE_VERSION,
            "source_revision": source,
            "source_revision_verification": "fake_test_assertion",
            "benchmark_version": benchmark,
            "wheel_sha256": "1" * 64,
        }

    def load_suite(self, name: str, **_: object) -> _Suite:
        return self.suites[name]

    def get_user_task(self, suite: _Suite, identifier: str) -> _UserTask:
        return suite.user_tasks[identifier]

    def get_injection_task(self, suite: _Suite, identifier: str) -> _InjectionTask:
        return suite.injection_tasks[identifier]

    def canonical_tool_schemas(self, suite: _Suite) -> tuple[dict[str, object], ...]:
        return ({"name": f"{suite.name}_tool", "description": "fake", "parameters": {}},)

    def load_environment(
        self, suite: _Suite, user_task: _UserTask, injections: dict[str, str]
    ) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        return suite.name, user_task.ID, tuple(sorted(injections.items()))

    def environment_hash(self, environment: object) -> str:
        return stable_digest(environment)

    def get_injection_candidates(
        self, suite: _Suite, user_task: _UserTask
    ) -> tuple[str, ...]:
        return ("vector_0",)

    def generate_attack_injections(
        self,
        suite: _Suite,
        user_task: _UserTask,
        injection_task: _InjectionTask,
        *,
        attack_name: str,
        target_pipeline_name: str,
    ) -> dict[str, str]:
        del suite, user_task
        assert attack_name == "direct"
        assert target_pipeline_name == "silenttwin-local-tool-loop"
        return {"vector_0": f"released direct rendering: {injection_task.GOAL}"}

    def suite_metadata(self, suite: _Suite) -> _Metadata:
        return _Metadata(suite)


def test_fake_release_catalog_is_deterministic_and_split_safe() -> None:
    adapter = _FakeCompat()
    first = build_catalog(compat=adapter, validate_release_drift=False)
    second = build_catalog(compat=adapter, validate_release_drift=False)
    assert first == second
    validate_catalog(first, validate_release_drift=False)
    assert len(first["eligible_combinations"]) == 4 * 6 * 5
    assert any(
        row["dataset_split"] == EXCLUDED_CROSS_SPLIT
        for row in first["eligible_combinations"]
    )
    assert all(
        row["dataset_split"] != EXCLUDED_CROSS_SPLIT for row in first["scenarios"]
    )
    assert all(value == 6 for value in first["structural_group_count_by_suite"].values())
    assert all(
        row["released_attack_name"] == "direct"
        and row["released_attack_target_pipeline"] == "silenttwin-local-tool-loop"
        and row["released_attack_initial_environment_hash"]
        != row["initial_environment_hash"]
        for row in first["eligible_combinations"]
    )


def test_committed_release_catalog_is_hash_valid_without_agentdojo() -> None:
    path = Path(__file__).resolve().parents[2] / "configs/silenttwin/agentdojo/catalog-v1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_catalog(document)
    assert document["eligible_combination_count"] == 1467
