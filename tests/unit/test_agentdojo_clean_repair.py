from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from silenttwin.agentdojo.canonical import CanonicalToolCall, CanonicalToolSchema
from silenttwin.agentdojo.clean_repair import (
    ARMS,
    CHECKPOINT_SCHEMA,
    EXPECTED_JOB_COUNT,
    EXPECTED_MODEL_EPISODE_COUNT,
    EXPECTED_SUITE_GROUP_COUNTS,
    INPUT_SCHEMA,
    QWEN_REPAIR_IDENTITY,
    RUN_MANIFEST_SCHEMA,
    CleanRepairError,
    analyze_benchmark,
    expanded_jobs,
    run_trial,
    validate_protocol,
    validate_task_records,
)
from silenttwin.agentdojo.config import stable_hash
from silenttwin.io.jsonl import (
    atomic_write_json,
    atomic_write_objects_jsonl,
    sha256_file,
)
from tests.fakes.fake_backend import FakeCallOutcome, FakeEnvironment


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / (
    "configs/silenttwin/agentdojo/scientific-v6-clean-repair-train-v1.json"
)


def _load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rehash(value: dict[str, Any]) -> None:
    payload = dict(value)
    payload.pop("protocol_hash", None)
    value["protocol_hash"] = stable_hash(payload)


def _task_record(*, suite: str, index: int, protocol_hash: str) -> dict[str, Any]:
    call = CanonicalToolCall(f"oracle-{index}", "candidate_0", {}).to_dict()
    identity = {
        "protocol_hash": protocol_hash,
        "suite": suite,
        "structural_group_id": stable_hash([suite, index, "group"]),
        "source_scenario_id": stable_hash([suite, index, "scenario"]),
        "user_task_id": f"{suite}-user-{index}",
    }
    return {
        "record_type": "clean_repair_task",
        "task_id": stable_hash(identity),
        **identity,
        "dataset_split": "train",
        "selection_rule": "lexicographically_first_scenario_per_structural_group",
        "user_prompt_hash": stable_hash([suite, index, "prompt"]),
        "clean_initial_environment_hash": stable_hash([suite, index, "environment"]),
        "full_tool_schema_hash": stable_hash([suite, "schemas"]),
        "full_tool_names": ["candidate_0"],
        "restricted_tool_names": ["candidate_0"],
        "restricted_tool_scope_source": (
            "released_user_task_ground_truth_function_names"
        ),
        "restricted_scope_is_benchmark_oracle": True,
        "oracle_calls": [call],
        "oracle_calls_hash": stable_hash([call]),
        "oracle_output": "done",
        "oracle_output_hash": stable_hash("done"),
        "oracle_validation": {
            "executed_call_count": 1,
            "tool_error_count": 0,
            "strict_utility": True,
            "trace_hash": stable_hash("trace"),
            "final_environment_hash": stable_hash("final"),
            "action_schema_execution_audit_hash": stable_hash("audit"),
        },
        "private_or_adversarial_fields_present": False,
    }


def _corpus(protocol_hash: str) -> list[dict[str, Any]]:
    return [
        _task_record(suite=suite, index=index, protocol_hash=protocol_hash)
        for suite, count in EXPECTED_SUITE_GROUP_COUNTS.items()
        for index in range(count)
    ]


def test_checked_clean_repair_protocol_is_self_consistent() -> None:
    protocol = _load_protocol()

    assert validate_protocol(protocol) == protocol["protocol_hash"]
    assert protocol["design"]["arms"] == list(ARMS)
    assert protocol["design"]["expected_job_count"] == EXPECTED_JOB_COUNT
    assert (
        protocol["design"]["expected_model_episode_count"]
        == EXPECTED_MODEL_EPISODE_COUNT
    )
    assert protocol["model"] == dict(QWEN_REPAIR_IDENTITY)
    assert protocol["access_policy"]["execution_permitted_splits"] == ["train"]
    assert "generic_e4_authored_effect_salvage" in protocol["explicitly_not_opened"]


def test_clean_repair_protocol_cannot_silently_open_development() -> None:
    protocol = copy.deepcopy(_load_protocol())
    protocol["access_policy"]["execution_permitted_splits"] = [
        "train",
        "development",
    ]
    protocol["access_policy"]["development_submission_permitted"] = True
    _rehash(protocol)

    with pytest.raises(CleanRepairError, match="train-only access"):
        validate_protocol(protocol)


def test_clean_repair_task_corpus_is_exact_and_rejects_private_fields() -> None:
    protocol = _load_protocol()
    corpus = _corpus(protocol["protocol_hash"])

    validate_task_records(corpus, protocol_hash=protocol["protocol_hash"])

    tampered = copy.deepcopy(corpus)
    tampered[0]["injection_goal"] = "private payload"
    with pytest.raises(CleanRepairError, match="private or adversarial"):
        validate_task_records(tampered, protocol_hash=protocol["protocol_hash"])


class _RepairTask:
    ID = "workspace-user-0"
    PROMPT = "Complete the permitted report."
    GROUND_TRUTH_OUTPUT = "done"

    @staticmethod
    def ground_truth(environment: FakeEnvironment) -> list[dict[str, Any]]:
        del environment
        return [{"function": "candidate_0", "arguments": {}}]


class _RepairSuite:
    name = "workspace"


class _RepairCompat:
    @staticmethod
    def load_suite(name: str) -> _RepairSuite:
        assert name == "workspace"
        return _RepairSuite()

    @staticmethod
    def get_user_task(suite: _RepairSuite, task_id: str) -> _RepairTask:
        del suite
        assert task_id == _RepairTask.ID
        return _RepairTask()

    @staticmethod
    def canonical_tool_schemas(suite: _RepairSuite) -> tuple[CanonicalToolSchema, ...]:
        del suite
        return (
            CanonicalToolSchema(
                "candidate_0",
                "Commit the permitted fixture report.",
                {"type": "object", "properties": {}},
            ),
        )

    @staticmethod
    def load_environment(
        suite: _RepairSuite, task: _RepairTask, injections: Mapping[str, str]
    ) -> FakeEnvironment:
        del suite, task
        assert injections == {}
        return FakeEnvironment({"effects": []})

    @staticmethod
    def clone_environment(environment: FakeEnvironment) -> FakeEnvironment:
        return environment.model_copy(deep=True)

    @staticmethod
    def environment_hash(environment: FakeEnvironment) -> str:
        return stable_hash(environment.model_dump(mode="json"))

    @classmethod
    def execute_call(
        cls,
        suite: _RepairSuite,
        environment: FakeEnvironment,
        call: Mapping[str, Any],
    ) -> FakeCallOutcome:
        del suite
        before = cls.environment_hash(environment)
        assert call["function"] == "candidate_0"
        environment.state["effects"].append("candidate_0")
        after = cls.environment_hash(environment)
        return FakeCallOutcome(
            {"status": "ok"},
            None,
            ({"function": "candidate_0", "args": {}},),
            before,
            after,
        )

    @staticmethod
    def grade_utility(
        task: _RepairTask,
        output_text: str,
        pre_environment: FakeEnvironment,
        post_environment: FakeEnvironment,
        traces: tuple[Any, ...],
        *,
        strict: bool,
    ) -> bool:
        del task, pre_environment, traces
        assert strict is True
        return output_text == "done" and post_environment.state["effects"] == [
            "candidate_0"
        ]


class _Response:
    def __init__(self, text: str, metadata: Mapping[str, Any]) -> None:
        self.text = text
        self.metadata = metadata
        self.usage = None


class _RepairModel:
    def __init__(self, *, metadata_overrides: Mapping[str, Any] | None = None) -> None:
        self._metadata_overrides = dict(metadata_overrides or {})

    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> _Response:
        del seed, max_tokens
        request = json.loads(prompt.split("TOOL_LOOP_INPUT:\n", 1)[1])
        has_tool_result = any(
            message.get("role") == "tool" for message in request["messages"]
        )
        value = (
            {"content": "done", "tool_calls": []}
            if has_tool_result
            else {
                "content": None,
                "tool_calls": [
                    {
                        "call_id": "repair-call",
                        "function": "candidate_0",
                        "arguments": {},
                    }
                ],
            }
        )
        metadata = {
            "client": "local_transformers",
            "model_id": QWEN_REPAIR_IDENTITY["model_id"],
            "requested_model_revision": QWEN_REPAIR_IDENTITY["model_revision"],
            "model_revision": QWEN_REPAIR_IDENTITY["model_revision"],
            "requested_tokenizer_revision": QWEN_REPAIR_IDENTITY[
                "tokenizer_revision"
            ],
            "tokenizer_revision": QWEN_REPAIR_IDENTITY["tokenizer_revision"],
            "local_checkpoint_fingerprint": QWEN_REPAIR_IDENTITY[
                "checkpoint_fingerprint"
            ],
            "local_checkpoint_verification_mode": "full_tree_sha256_audit",
            "local_checkpoint_manifest_hash": "a" * 64,
            "local_checkpoint_path": "/frozen/qwen",
            "dtype": QWEN_REPAIR_IDENTITY["dtype"],
            "device": "cuda:0",
            "temperature": QWEN_REPAIR_IDENTITY["temperature"],
            "top_p": QWEN_REPAIR_IDENTITY["top_p"],
            "batch_size": 1,
            "local_files_only": True,
            "external_api_calls": 0,
            "gpu_name": "NVIDIA H200",
            "input_prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        metadata.update(self._metadata_overrides)
        return _Response(json.dumps(value), metadata)


def _live_record(protocol_hash: str) -> dict[str, Any]:
    record = _task_record(
        suite="workspace", index=0, protocol_hash=protocol_hash
    )
    environment = _RepairCompat.load_environment(_RepairSuite(), _RepairTask(), {})
    schemas = _RepairCompat.canonical_tool_schemas(_RepairSuite())
    call = CanonicalToolCall("call-9f", "candidate_0", {}).to_dict()
    # canonicalize_tool_call derives its ID from content; use the exact live form.
    from silenttwin.agentdojo.canonical import canonicalize_tool_call

    call = canonicalize_tool_call(
        {"function": "candidate_0", "arguments": {}}, default_id="oracle-0"
    ).to_dict()
    identity = {
        "protocol_hash": protocol_hash,
        "suite": "workspace",
        "structural_group_id": stable_hash("live-group"),
        "source_scenario_id": stable_hash("live-scenario"),
        "user_task_id": _RepairTask.ID,
    }
    record.update(
        {
            "task_id": stable_hash(identity),
            **identity,
            "user_prompt_hash": stable_hash(_RepairTask.PROMPT),
            "clean_initial_environment_hash": _RepairCompat.environment_hash(
                environment
            ),
            "full_tool_schema_hash": stable_hash(
                [schema.to_dict() for schema in schemas]
            ),
            "oracle_calls": [call],
            "oracle_calls_hash": stable_hash([call]),
        }
    )
    return record


def test_clean_repair_trial_uses_fresh_context_then_atomic_replay() -> None:
    protocol = _load_protocol()
    task = _live_record(protocol["protocol_hash"])
    corpus_hash = stable_hash("fixture-repair-corpus")

    repaired = run_trial(
        protocol=protocol,
        task_record=task,
        arm="fresh_repair_restricted_tools",
        model_client=_RepairModel(),
        compat=_RepairCompat,
        task_records_hash=corpus_hash,
    )
    terminated = run_trial(
        protocol=protocol,
        task_record=task,
        arm="no_repair",
        model_client=None,
        compat=_RepairCompat,
        task_records_hash=corpus_hash,
    )
    oracle = run_trial(
        protocol=protocol,
        task_record=task,
        arm="oracle_ground_truth",
        model_client=None,
        compat=_RepairCompat,
        task_records_hash=corpus_hash,
    )

    assert repaired["utility"] == 1
    assert repaired["run_valid"] is True
    assert repaired["model_provenance_valid"] is True
    assert repaired["locked_call_count"] == 1
    assert repaired["committed_call_count"] == 1
    assert repaired["exact_oracle_call_sequence"] is True
    assert repaired["restricted_function_scope_compliant"] is True
    assert repaired["repair_context_retired_before_final_replay"] is True
    assert repaired["sanitized_handoff_delivered"] is True
    assert repaired["injection_payload_present"] is False
    assert repaired["private_verdict_present"] is False
    assert terminated["utility"] == 0
    assert oracle["utility"] == 1


def test_clean_repair_trial_fails_closed_on_provenance_or_prompt_drift() -> None:
    protocol = _load_protocol()
    task = _live_record(protocol["protocol_hash"])
    corpus_hash = stable_hash("fixture-repair-corpus")

    wrong_gpu = run_trial(
        protocol=protocol,
        task_record=task,
        arm="fresh_repair_restricted_tools",
        model_client=_RepairModel(metadata_overrides={"gpu_name": "NVIDIA A100"}),
        compat=_RepairCompat,
        task_records_hash=corpus_hash,
    )
    wrong_prompt = run_trial(
        protocol=protocol,
        task_record=task,
        arm="fresh_repair_restricted_tools",
        model_client=_RepairModel(metadata_overrides={"input_prompt_hash": "0" * 64}),
        compat=_RepairCompat,
        task_records_hash=corpus_hash,
    )

    assert wrong_gpu["raw_strict_utility"] == 1
    assert wrong_gpu["utility"] == 0
    assert wrong_gpu["run_valid"] is False
    assert wrong_gpu["model_provenance_valid"] is False
    assert "invalid_model_provenance" in wrong_gpu["errors"]
    assert wrong_prompt["raw_strict_utility"] == 1
    assert wrong_prompt["utility"] == 0
    assert wrong_prompt["run_valid"] is False
    assert wrong_prompt["prompt_binding_valid"] is False
    assert "invalid_prompt_binding" in wrong_prompt["errors"]


def test_clean_repair_analysis_validates_complete_train_only_gate(
    tmp_path: Path,
) -> None:
    protocol = _load_protocol()
    protocol_hash = protocol["protocol_hash"]
    tasks = _corpus(protocol_hash)
    task_records_hash = stable_hash(tasks)
    input_path = tmp_path / "inputs.jsonl"
    metadata_payload = {
        "schema_version": INPUT_SCHEMA,
        "record_type": "metadata",
        "protocol_hash": protocol_hash,
        "task_count": len(tasks),
        "job_count": EXPECTED_JOB_COUNT,
        "model_episode_count": EXPECTED_MODEL_EPISODE_COUNT,
        "task_records_hash": task_records_hash,
        "private_or_adversarial_fields_present": False,
    }
    atomic_write_objects_jsonl(
        input_path,
        [{**metadata_payload, "metadata_hash": stable_hash(metadata_payload)}, *tasks],
    )

    rows = []
    for job in expanded_jobs(tasks):
        arm = job["arm"]
        learned = arm in {
            "clean_start_full_tools",
            "fresh_repair_full_tools",
            "fresh_repair_restricted_tools",
        }
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "trial_id": job["job_id"],
            "task_records_hash": task_records_hash,
            "task_id": job["task_id"],
            "arm": arm,
            "dataset_split": "train",
            "agentdojo_suite": job["task"]["suite"],
            "structural_group_id": job["task"]["structural_group_id"],
            "utility": int(arm != "no_repair"),
            "run_valid": True,
            "model_protocol_valid": True,
            "model_provenance_valid": True,
            "prompt_binding_valid": True,
            "exact_oracle_call_sequence": arm == "oracle_ground_truth",
            "restricted_function_scope_compliant": True,
            "locked_call_count": int(arm != "no_repair"),
            "model_call_count": int(learned),
            "token_count": 10 * int(learned),
            "unauthorized_function_count": 0,
            "tool_loop_error": None,
            "prior_context_present": False,
            "prior_transcript_present": False,
            "prior_draft_present": False,
            "injection_payload_present": False,
            "private_verdict_present": False,
            "private_reason_present": False,
            "external_api_calls": 0,
            "development_outcomes_inspected": False,
            "test_outcomes_inspected": False,
        }
        rows.append({**payload, "checkpoint_hash": stable_hash(payload)})

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    result_path = run_directory / "result.jsonl"
    atomic_write_objects_jsonl(result_path, rows)
    manifest_payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": "complete",
        "protocol_hash": protocol_hash,
        "input_file_sha256": sha256_file(input_path),
        "task_records_hash": task_records_hash,
        "expected_job_count": EXPECTED_JOB_COUNT,
        "expected_task_count": len(tasks),
        "expected_model_episode_count": EXPECTED_MODEL_EPISODE_COUNT,
        "model": dict(QWEN_REPAIR_IDENTITY),
        "source_tree_hash": "1" * 64,
        "code_revision": "0" * 40,
        "result_file": result_path.name,
        "result_sha256": sha256_file(result_path),
        "development_outcomes_inspected": False,
        "test_outcomes_inspected": False,
    }
    atomic_write_json(
        run_directory / "run_manifest.json",
        {
            **manifest_payload,
            "run_manifest_hash": stable_hash(manifest_payload),
        },
    )

    output_path = tmp_path / "analysis.json"
    summary = analyze_benchmark(
        protocol_path=PROTOCOL_PATH,
        input_path=input_path,
        run_directory=run_directory,
        output_path=output_path,
    )
    analysis = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["train_component_feasibility_supported"] is True
    assert analysis["train_component_feasibility_supported"] is True
    assert all(analysis["preregistered_feasibility_criteria"].values())
    assert analysis["development_submission_permitted"] is False
    assert analysis["held_out_evaluation_permitted"] is False
    assert analysis["confirmatory_claim_permitted"] is False
