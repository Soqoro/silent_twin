"""Fail-closed AgentDojo grader adapter."""

from __future__ import annotations

import importlib
from typing import Any, Sequence

from silenttwin.backends.base import BackendError, BackendErrorStage, BackendGrades

from .results import make_grades


def _compat_module(compat: Any | None) -> Any:
    if compat is not None:
        return compat
    return importlib.import_module("silenttwin.agentdojo.compat")


def _call_grader(
    function: Any,
    task: Any,
    output_text: str,
    pre_environment: Any,
    post_environment: Any,
    traces: Sequence[Any],
) -> bool | None:
    # The compatibility facade freezes this exact signature.  Never retry a
    # TypeError: it may have been raised inside an effectful grader and a
    # second invocation would make error accounting non-idempotent.
    return function(task, output_text, pre_environment, post_environment, traces)


def grade_agentdojo(
    *,
    user_task: Any,
    injection_task: Any | None,
    output_text: str,
    pre_environment: Any,
    post_environment: Any,
    traces: Sequence[Any],
    compat: Any | None = None,
) -> BackendGrades:
    """Run utility and attack graders without conflating failures with safety."""

    adapter = _compat_module(compat)
    errors: list[BackendError] = []
    utility: bool | None = None
    raw_attack_success: bool | None = False if injection_task is None else None
    try:
        utility = _call_grader(
            adapter.grade_utility,
            user_task,
            output_text,
            pre_environment,
            post_environment,
            traces,
        )
        if utility is not None and not isinstance(utility, bool):
            raise TypeError("utility grader returned a non-boolean value")
    except Exception as exc:
        errors.append(
            BackendError(
                BackendErrorStage.UTILITY_GRADER,
                "utility_grader_error",
                f"{type(exc).__name__}:{exc}",
            )
        )
    if injection_task is not None:
        try:
            raw_attack_success = _call_grader(
                adapter.grade_attack_success,
                injection_task,
                output_text,
                pre_environment,
                post_environment,
                traces,
            )
            if raw_attack_success is not None and not isinstance(
                raw_attack_success, bool
            ):
                raise TypeError("attack grader returned a non-boolean value")
        except Exception as exc:
            raw_attack_success = None
            errors.append(
                BackendError(
                    BackendErrorStage.ATTACK_GRADER,
                    "attack_grader_error",
                    f"{type(exc).__name__}:{exc}",
                )
            )
    return make_grades(
        utility=utility,
        raw_security_result=raw_attack_success,
        errors=errors,
    )


__all__ = ["grade_agentdojo"]
