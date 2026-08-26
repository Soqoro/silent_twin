from __future__ import annotations

import importlib
import sys

import pytest

from silenttwin.schemas import stable_digest


def test_import_is_agentdojo_dependency_free() -> None:
    before = set(sys.modules)
    module = importlib.import_module("silenttwin.agentdojo.compat")
    imported = set(sys.modules) - before
    assert "agentdojo" not in imported
    assert not any(name.startswith("agentdojo.") for name in imported)
    assert module.EXPECTED_PYTHON == (3, 11)


class _FakeEnvironment:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self.value

    def model_copy(self, *, deep: bool) -> "_FakeEnvironment":
        assert deep is True
        return _FakeEnvironment({**self.value})


def test_environment_clone_and_hash_are_stable() -> None:
    from silenttwin.agentdojo.compat import clone_environment, environment_hash

    environment = _FakeEnvironment({"b": 2, "a": [1]})
    clone = clone_environment(environment)
    assert clone is not environment
    assert environment_hash(environment) == stable_digest({"a": [1], "b": 2})
    assert environment_hash(clone) == environment_hash(environment)


def test_canonical_call_excludes_provider_id() -> None:
    from silenttwin.agentdojo.compat import canonical_call

    assert canonical_call(
        {"function": "send_email", "args": {"to": "a@example.test"}, "id": "provider-1"}
    ) == {"function": "send_email", "args": {"to": "a@example.test"}}


def test_task_metadata_hashes_text_without_exposing_it() -> None:
    from silenttwin.agentdojo.compat import task_metadata

    task = type(
        "Task",
        (),
        {"ID": "user_task_3", "PROMPT": "private-ish fixture text", "DIFFICULTY": "easy"},
    )()
    value = task_metadata(task)
    assert value.task_id == "user_task_3"
    assert value.task_kind == "user"
    assert value.text_hash == stable_digest("private-ish fixture text")


def test_actual_tier2_use_fails_closed_outside_python_311() -> None:
    from silenttwin.agentdojo.compat import AgentDojoCompatibilityError, assert_compatible

    assert_compatible.cache_clear()
    if sys.version_info[:2] == (3, 11):
        with pytest.raises(AgentDojoCompatibilityError, match="source revision"):
            assert_compatible("0" * 40)
    else:
        with pytest.raises(AgentDojoCompatibilityError, match="Python 3.11 exactly"):
            assert_compatible()
