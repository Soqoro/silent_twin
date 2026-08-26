from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from silenttwin.agentdojo.config import (
    AGENTDOJO_RESULT_SCHEMA,
    AgentDojoExperimentConfig,
    ModelIdentity,
    bundle_hash,
    stable_hash,
)
from silenttwin.agentdojo.runtime_integrity import (
    LearnedRuntimeReport,
    learned_runtime_manifest_fingerprint,
    make_learned_runtime_provenance,
)
from silenttwin.agentdojo.storage import (
    AgentDojoCheckpointStore,
    CHECKPOINT_MANIFEST_FILENAME,
    FAILURES_FILENAME,
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    bind_evidence_boundary,
    _validate_model_provenance,
    publish_completed_run,
    trial_checkpoint_id,
    validate_completed_run,
)
from silenttwin.io.jsonl import ResultValidationError, sha256_file
from silenttwin.schemas import stable_digest


SOURCE_HASH = stable_hash("fixture-source-tree")
GRID_HASH = stable_hash("fixture-grid")


def _learned_runtime_provenance() -> dict[str, object]:
    runtime_manifest = {
        "schema_version": "silenttwin.agentdojo.learned-runtime/v1",
        "python": {
            "implementation": "cpython",
            "version": [3, 11, 15],
            "cache_tag": "cpython-311",
            "abi_flags": "",
            "soabi": "cpython-311-x86_64-linux-gnu",
            "byteorder": "little",
            "system": "Linux",
            "machine": "x86_64",
        },
        "locked_core": [{"name": "agentdojo", "version": "0.1.35"}],
        "installed_distributions": [
            {"name": "agentdojo", "version": "0.1.35", "record_identity": "1" * 64},
            {"name": "torch", "version": "2.7.1+site", "record_identity": "2" * 64},
            {"name": "transformers", "version": "4.55.0", "record_identity": "3" * 64},
        ],
    }
    fingerprint = learned_runtime_manifest_fingerprint(runtime_manifest)
    return make_learned_runtime_provenance(
        LearnedRuntimeReport(
            fingerprint=fingerprint,
            distribution_count=3,
            manifest=runtime_manifest,
        )
    )


def _config(output_dir: Path, *, feedback_policy: str = "generic_failure") -> AgentDojoExperimentConfig:
    scenario_ids = ("scenario-0", "scenario-1")
    structural_ids = ("group-0", "group-1")
    return AgentDojoExperimentConfig(
        experiment_id="catalog",
        tier2_track="controlled",
        agentdojo_suite="workspace",
        dataset_split="train",
        agentdojo_catalog_hash=stable_hash("catalog"),
        scenario_registry_revision="fixture-registry-v1",
        scenario_registry_hash=stable_hash("registry"),
        split_manifest_hash=stable_hash("splits"),
        candidate_strategy_catalog_hash=stable_hash("strategies"),
        pair_registry_hash=stable_hash("pairs"),
        scenario_bundle_hash=bundle_hash(
            suite="workspace",
            dataset_split="train",
            scenario_ids=scenario_ids,
            structural_group_ids=structural_ids,
        ),
        scenario_ids=scenario_ids,
        structural_group_ids=structural_ids,
        analysis_plan_hash=stable_hash("analysis"),
        dependency_lock_hash=stable_hash("lock"),
        monitor_profile_hash=stable_hash("monitor"),
        system_prompt_hash=stable_hash("system"),
        feedback_policy=feedback_policy,
        output_dir=output_dir,
    )


def _sample(config: AgentDojoExperimentConfig, trial_id: str, index: int) -> dict[str, object]:
    return bind_evidence_boundary(
        {
            "schema_version": AGENTDOJO_RESULT_SCHEMA,
            "record_type": "sample",
            "trial_id": trial_id,
            "environment_backend": "agentdojo",
            "experiment_id": config.experiment_id,
            "tier2_track": config.tier2_track,
            "agentdojo_suite": config.agentdojo_suite,
            "scenario_id": config.scenario_ids[index],
            "structural_group_id": config.structural_group_ids[index],
            "agent_visible_transcript": [],
            "postselection_output": [],
            "trusted_evaluation": {
                "schema": "silenttwin.agentdojo.trusted.v1",
                "value": {},
            },
            "fixture_value": index,
        },
        config=config,
    )


def _store(output_dir: Path) -> tuple[AgentDojoCheckpointStore, AgentDojoExperimentConfig]:
    config = _config(output_dir)
    store = AgentDojoCheckpointStore(
        output_dir,
        config,
        ("trial-0", "trial-1"),
        provenance_hash=SOURCE_HASH,
    )
    return store, config


def _publish(output_dir: Path) -> tuple[dict[str, object], AgentDojoExperimentConfig]:
    store, config = _store(output_dir)
    store.initialize()
    store.save(_sample(config, "trial-0", 0))
    store.save(_sample(config, "trial-1", 1))
    manifest = publish_completed_run(
        store=store,
        failures=(),
        started_at="2026-08-24T00:00:00Z",
        completed_at="2026-08-24T00:01:00Z",
        grid_hash=GRID_HASH,
        grid_task_id=7,
        shard_id="fixture-shard",
        grid_batch_hash=stable_hash("fixture-batch"),
        provenance={
            "source_tree_hash": SOURCE_HASH,
            "code_revision": "fixture",
            "scheduler": {"job_id": "fixture-job"},
        },
    )
    return manifest, config


def test_checkpoint_resume_idempotence_and_collision(tmp_path: Path) -> None:
    store, config = _store(tmp_path)
    store.initialize()
    first = _sample(config, "trial-0", 0)
    store.save(first)
    checkpoint_id = trial_checkpoint_id(config.configuration_hash, "trial-0")
    checkpoint_path = store.checkpoint_dir / f"{checkpoint_id}.json"
    original_bytes = checkpoint_path.read_bytes()

    # Saving the exact same scientific row is an idempotent retry.
    store.save(copy.deepcopy(first))
    assert checkpoint_path.read_bytes() == original_bytes

    changed = copy.deepcopy(first)
    changed["fixture_value"] = 99
    with pytest.raises(ResultValidationError, match="checkpoint collision"):
        store.save(changed)

    # Emulate the one supported crash window: the checkpoint rename completed
    # before the running manifest's advisory completed-ID update.
    manifest_path = tmp_path / CHECKPOINT_MANIFEST_FILENAME
    checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest["completed_checkpoint_ids"] = []
    payload = dict(checkpoint_manifest)
    payload.pop("checkpoint_manifest_hash")
    checkpoint_manifest["checkpoint_manifest_hash"] = stable_hash(payload)
    manifest_path.write_text(json.dumps(checkpoint_manifest), encoding="utf-8")

    resumed = AgentDojoCheckpointStore(
        tmp_path,
        config,
        ("trial-0", "trial-1"),
        provenance_hash=SOURCE_HASH,
    )
    resumed.initialize()
    assert tuple(resumed.load()) == ("trial-0",)
    repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert repaired["completed_checkpoint_ids"] == [checkpoint_id]

    resumed.save(_sample(config, "trial-1", 1))
    assert tuple(resumed.mark_complete()) == ("trial-0", "trial-1")

    collision = AgentDojoCheckpointStore(
        tmp_path,
        _config(tmp_path, feedback_policy="binary_denial"),
        ("trial-0", "trial-1"),
        provenance_hash=SOURCE_HASH,
    )
    with pytest.raises(ResultValidationError, match="another configuration"):
        collision.initialize()


def test_checkpoint_accepts_exact_versioned_public_envelopes(tmp_path: Path) -> None:
    store, config = _store(tmp_path)
    store.initialize()
    sample = _sample(config, "trial-0", 0)
    sample["agent_visible_transcript"] = [
        {
            "schema": "silenttwin.agentdojo.probe-feedback.v1",
            "value": {"status": "failed"},
        }
    ]
    sample["postselection_output"] = [
        {"schema": "silenttwin.agentdojo.public.v1", "value": "complete"}
    ]
    store.save(sample)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("trusted_schema_swap", "trusted-evaluation envelope schema"),
        ("public_schema_swap", "does not use a public"),
        ("postselection_private_schema", "does not use a public"),
        ("unknown_public_schema", "does not use a public"),
        ("raw_transcript_element", "not an exact public envelope"),
        ("legacy_transcript", "legacy public_transcript fallback"),
    ),
)
def test_checkpoint_rejects_v1_namespace_schema_swaps_and_raw_fallbacks(
    tmp_path: Path, mutation: str, match: str
) -> None:
    store, config = _store(tmp_path)
    store.initialize()
    sample = _sample(config, "trial-0", 0)
    if mutation == "trusted_schema_swap":
        sample["trusted_evaluation"]["schema"] = (
            "silenttwin.agentdojo.public.v1"
        )
    elif mutation == "public_schema_swap":
        sample["agent_visible_transcript"] = [
            {
                "schema": "silenttwin.agentdojo.trusted.v1",
                "value": {"private": True},
            }
        ]
    elif mutation == "postselection_private_schema":
        sample["postselection_output"] = [
            {
                "schema": "silenttwin.agentdojo.private.v1",
                "value": {"private": True},
            }
        ]
    elif mutation == "unknown_public_schema":
        sample["agent_visible_transcript"] = [
            {
                "schema": "silenttwin.agentdojo.unregistered-public.v1",
                "value": {"looks_public": True},
            }
        ]
    elif mutation == "raw_transcript_element":
        sample["agent_visible_transcript"] = [{"status": "legacy-raw"}]
    else:
        sample["public_transcript"] = sample.pop("agent_visible_transcript")

    with pytest.raises(ResultValidationError, match=match):
        store.save(sample)


def test_checkpoint_and_manifest_hash_tampering_is_rejected(tmp_path: Path) -> None:
    store, config = _store(tmp_path)
    store.initialize()
    store.save(_sample(config, "trial-0", 0))
    checkpoint_id = trial_checkpoint_id(config.configuration_hash, "trial-0")
    checkpoint_path = store.checkpoint_dir / f"{checkpoint_id}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["sample"]["fixture_value"] = 123
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="self-hash is invalid"):
        store.load()

    other_dir = tmp_path / "manifest-tamper"
    other, other_config = _store(other_dir)
    other.initialize()
    other.save(_sample(other_config, "trial-0", 0))
    manifest_path = other_dir / CHECKPOINT_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_tree_hash"] = stable_hash("tampered-source")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="manifest hash is invalid"):
        other.load()


def test_complete_publication_manifest_and_idempotent_validation(tmp_path: Path) -> None:
    manifest, config = _publish(tmp_path)
    assert manifest["status"] == "complete"
    assert manifest["expected_trial_count"] == 2
    assert manifest["actual_trial_count"] == 2
    assert manifest["provenance"]["learned_runtime"] == {
        "schema_version": "silenttwin.agentdojo.learned-runtime-provenance/v1",
        "status": "not_applicable",
        "runtime_fingerprint": "not_applicable",
        "manifest": "not_applicable",
    }
    first = validate_completed_run(
        tmp_path,
        expected_config=config,
        expected_grid_hash=GRID_HASH,
        expected_shard_id="fixture-shard",
    )
    second = validate_completed_run(
        tmp_path,
        expected_config=config,
        expected_grid_hash=GRID_HASH,
        expected_shard_id="fixture-shard",
    )
    assert first == second == manifest


def test_learned_run_manifest_retains_complete_runtime_provenance(
    tmp_path: Path,
) -> None:
    learned_runtime = _learned_runtime_provenance()
    identity = ModelIdentity(
        role="attacker",
        implementation="local_transformers",
        model_id="fixture/site-model",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        checkpoint_fingerprint="sha256:" + "c" * 64,
        runtime_fingerprint=str(learned_runtime["runtime_fingerprint"]),
        prompt_hash=stable_hash("fixture-prompt"),
    )
    config = replace(_config(tmp_path), models=(identity,))
    store = AgentDojoCheckpointStore(
        tmp_path,
        config,
        ("trial-0", "trial-1"),
        provenance_hash=SOURCE_HASH,
    )
    store.initialize()
    store.save(_sample(config, "trial-0", 0))
    store.save(_sample(config, "trial-1", 1))
    manifest = publish_completed_run(
        store=store,
        failures=(),
        started_at="2026-08-24T00:00:00Z",
        completed_at="2026-08-24T00:01:00Z",
        grid_hash=GRID_HASH,
        grid_task_id=7,
        shard_id="learned-shard",
        provenance={
            "source_tree_hash": SOURCE_HASH,
            "scheduler": {},
            "learned_runtime": learned_runtime,
        },
    )
    assert manifest["provenance"]["learned_runtime"] == learned_runtime
    assert len(
        manifest["provenance"]["learned_runtime"]["manifest"][
            "installed_distributions"
        ]
    ) == 3
    validate_completed_run(tmp_path, expected_config=config)


def test_complete_validator_rejects_stream_checkpoint_and_binding_tampering(
    tmp_path: Path,
) -> None:
    stream_dir = tmp_path / "stream"
    _publish(stream_dir)
    result_path = stream_dir / RESULT_FILENAME
    records = result_path.read_text(encoding="utf-8")
    result_path.write_text(records.replace('"fixture_value":0', '"fixture_value":9'), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="result digest mismatch"):
        validate_completed_run(stream_dir)

    checkpoint_dir = tmp_path / "checkpoint"
    _, config = _publish(checkpoint_dir)
    checkpoint_id = trial_checkpoint_id(config.configuration_hash, "trial-0")
    checkpoint_path = checkpoint_dir / "checkpoints" / f"{checkpoint_id}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["sample"]["fixture_value"] = 44
    payload = dict(checkpoint)
    payload.pop("checkpoint_hash")
    checkpoint["checkpoint_hash"] = stable_hash(payload)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="results differ"):
        validate_completed_run(checkpoint_dir)

    binding_dir = tmp_path / "binding"
    _publish(binding_dir)
    manifest_path = binding_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["agentdojo"]["catalog_hash"] = stable_hash("another-catalog")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="upstream binding"):
        validate_completed_run(binding_dir)

    runtime_dir = tmp_path / "runtime-provenance"
    _publish(runtime_dir)
    manifest_path = runtime_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["learned_runtime"]["status"] = "captured"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="learned-runtime provenance"):
        validate_completed_run(runtime_dir)


def test_rehashed_checkpoint_manifest_cannot_rebind_source_or_cohort(tmp_path: Path) -> None:
    _publish(tmp_path)
    checkpoint_path = tmp_path / CHECKPOINT_MANIFEST_FILENAME
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["source_tree_hash"] = stable_hash("another-source")
    payload = dict(checkpoint)
    payload.pop("checkpoint_manifest_hash")
    checkpoint["checkpoint_manifest_hash"] = stable_hash(payload)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="another source tree"):
        validate_completed_run(tmp_path)


def _e2_sample(
    config: AgentDojoExperimentConfig,
    *,
    actual_state: str = "theta0",
    donor_state: str = "theta1",
) -> dict[str, object]:
    scenario_id = config.scenario_ids[0]
    trial_id = stable_hash(
        {
            "protocol": "silenttwin.agentdojo.controlled.v1",
            "configuration_hash": config.configuration_hash,
            "scenario_id": scenario_id,
            "actual_state": actual_state,
            "donor_state": donor_state,
            "replicate": config.replicate,
        }
    )
    canonical_input = {"request_type": "hidden_state_prediction"}
    protocol_prompt = "fixture controlled prediction prompt"
    raw_response = '{"posterior":{"theta0":1.0,"theta1":0.0},"prediction":"theta0"}'
    parsed_output = {
        "posterior": {"theta0": 1.0, "theta1": 0.0},
        "prediction": "theta0",
    }
    final_canonical_input = {"request_type": "final_plan_selection"}
    final_prompt = "fixture controlled final prompt"
    final_raw_response = '{"plan_id":"candidate_0"}'
    final_parsed_output = {"plan_id": "candidate_0"}
    return bind_evidence_boundary(
        {
            "schema_version": AGENTDOJO_RESULT_SCHEMA,
            "record_type": "sample",
            "trial_id": trial_id,
            "environment_backend": "agentdojo",
            "experiment_id": "e2",
            "tier2_track": "controlled",
            "agentdojo_suite": config.agentdojo_suite,
            "scenario_id": scenario_id,
            "structural_group_id": config.structural_group_ids[0],
            "dataset_split": config.dataset_split,
            "threat_mode": config.threat_mode,
            "feedback_policy": config.feedback_policy,
            "feedback_source": config.feedback_source,
            "query_budget": config.query_budget,
            "condition": config.condition,
            "agent_visible_transcript": [],
            "postselection_output": [],
            "trusted_evaluation": {
                "schema": "silenttwin.agentdojo.trusted.v1",
                "value": {
                    "actual_hidden_state": actual_state,
                    "donor_state": donor_state,
                },
            },
            "model_provenance": {
                "controlled_protocol_revision": "silenttwin.agentdojo.controlled.v1",
                "attacker": {
                    "calls": [
                        {
                            "phase": "prediction",
                            "call_index": 0,
                            "seed": 0,
                            "canonical_input_hash": stable_hash(canonical_input),
                            "rendered_input_hash": hashlib.sha256(
                                protocol_prompt.encode("utf-8")
                            ).hexdigest(),
                            "raw_response_hash": hashlib.sha256(
                                raw_response.encode("utf-8")
                            ).hexdigest(),
                            "parsed_output_hash": stable_hash(parsed_output),
                            "canonical_input": canonical_input,
                            "protocol_prompt": protocol_prompt,
                            "rendered_chat_template_input": protocol_prompt,
                            "raw_response": raw_response,
                            "parsed_output": parsed_output,
                            "latency_ms": 0.1,
                            "failure_metadata": {},
                            "metadata": {},
                            "error": None,
                        },
                        {
                            "phase": "final",
                            "call_index": 1,
                            "seed": 1,
                            "canonical_input_hash": stable_hash(
                                final_canonical_input
                            ),
                            "rendered_input_hash": hashlib.sha256(
                                final_prompt.encode("utf-8")
                            ).hexdigest(),
                            "raw_response_hash": hashlib.sha256(
                                final_raw_response.encode("utf-8")
                            ).hexdigest(),
                            "parsed_output_hash": stable_hash(
                                final_parsed_output
                            ),
                            "canonical_input": final_canonical_input,
                            "protocol_prompt": final_prompt,
                            "rendered_chat_template_input": final_prompt,
                            "raw_response": final_raw_response,
                            "parsed_output": final_parsed_output,
                            "latency_ms": 0.1,
                            "failure_metadata": {},
                            "metadata": {},
                            "error": None,
                        }
                    ]
                },
            },
        },
        config=config,
    )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("empty", "ledger is empty"),
        ("rendered_hash", "rendered input hash is inconsistent"),
        ("raw_material", "raw response hash is inconsistent"),
        ("missing_parsed", "required provenance fields"),
        ("duplicate_index", "not unique and sequential"),
        ("failure_with_parsed", "failure improperly claims parsed output"),
    ),
)
def test_checkpoint_rejects_missing_or_unbound_model_call_provenance(
    tmp_path: Path, mutation: str, match: str
) -> None:
    config = replace(
        _config(tmp_path),
        experiment_id="e2",
        condition="no_probe",
        query_budget=0,
    )
    sample = _e2_sample(config)
    calls = sample["model_provenance"]["attacker"]["calls"]
    if mutation == "empty":
        calls.clear()
    elif mutation == "rendered_hash":
        calls[0]["rendered_input_hash"] = "0" * 64
    elif mutation == "raw_material":
        calls[0]["raw_response"] = "tampered raw response"
    elif mutation == "missing_parsed":
        calls[0].pop("parsed_output")
    elif mutation == "duplicate_index":
        calls[1]["call_index"] = 0
    else:
        calls[0]["error"] = "fixture failure"
    store = AgentDojoCheckpointStore(
        tmp_path,
        config,
        (str(sample["trial_id"]),),
        provenance_hash=SOURCE_HASH,
    )
    store.initialize()

    with pytest.raises(ResultValidationError, match=match):
        store.save(sample)


def test_storage_validates_unicode_model_inputs_with_protocol_digest(
    tmp_path: Path,
) -> None:
    """Protocol hashes preserve Unicode even though artifact hashes escape it."""

    config = replace(
        _config(tmp_path),
        experiment_id="e2",
        condition="no_probe",
        query_budget=0,
    )
    sample = _e2_sample(config)
    prediction_call = sample["model_provenance"]["attacker"]["calls"][0]
    canonical_input = {
        "request_type": "hidden_state_prediction",
        "user_prompt": "Réservez le café à Zürich — 你好",
    }
    prediction_call["canonical_input"] = canonical_input
    prediction_call["canonical_input_hash"] = stable_digest(canonical_input)
    assert prediction_call["canonical_input_hash"] != stable_hash(canonical_input)

    _validate_model_provenance(sample, config=config)


def test_checkpoint_rejects_malformed_nested_monitor_call_provenance(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        experiment_id="e2",
        condition="no_probe",
        query_budget=0,
    )
    sample = _e2_sample(config)
    sample["trusted_evaluation"]["value"]["probe_monitor_evaluations"] = [
        {
            "monitor": {
                "provenance": {
                    "call_evaluations": [
                        {"provenance": {"model_call": {"phase": "incomplete"}}}
                    ]
                }
            }
        }
    ]
    store = AgentDojoCheckpointStore(
        tmp_path,
        config,
        (str(sample["trial_id"]),),
        provenance_hash=SOURCE_HASH,
    )
    store.initialize()

    with pytest.raises(ResultValidationError, match="monitor provenance"):
        store.save(sample)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("missing_raw", "lacks raw_output"),
        ("input_hash", "input hash is inconsistent"),
        ("output_hash", "output hash is inconsistent"),
        ("latency", "invalid latency"),
    ),
)
def test_storage_binds_pi_detector_call_material(
    mutation: str, match: str
) -> None:
    detector_input = "fixture detector input"
    raw_output = "[true,0.2]"
    detector_call: dict[str, object] = {
        "detector_input_text": detector_input,
        "detector_input_hash": hashlib.sha256(
            detector_input.encode("utf-8")
        ).hexdigest(),
        "rendered_chat_template_input": {"status": "not_available"},
        "tokenized_input": {"status": "not_available"},
        "raw_output": raw_output,
        "raw_output_hash": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "raw_output_serialization": {"status": "canonical_json"},
        "usage": {"status": "not_available"},
        "latency_ms": 0.1,
        "failure": None,
    }
    if mutation == "missing_raw":
        detector_call.pop("raw_output")
    elif mutation == "input_hash":
        detector_call["detector_input_hash"] = "0" * 64
    elif mutation == "output_hash":
        detector_call["raw_output_hash"] = "0" * 64
    else:
        detector_call["latency_ms"] = -1.0
    row = {
        "experiment_id": "ecological",
        "tier2_track": "ecological",
        "error_stage": "setup",
        "model_provenance": {
            "calls": [],
            "content_detector_provenance": {"calls": [detector_call]},
        },
    }

    with pytest.raises(ResultValidationError, match=match):
        _validate_model_provenance(
            row,
            config=SimpleNamespace(
                models=(), monitor_family="deterministic_task_policy"
            ),
        )


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("condition", "matched_shuffled"),
        ("query_budget", 16),
        ("feedback_policy", "binary_denial"),
        ("threat_mode", "clean"),
    ),
)
def test_checkpoint_rejects_rehashed_scientific_row_mismatch(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    config = replace(
        _config(tmp_path),
        experiment_id="e2",
        condition="no_probe",
        query_budget=0,
    )
    sample = _e2_sample(config)
    sample[field] = bad_value
    store = AgentDojoCheckpointStore(
        tmp_path,
        config,
        (str(sample["trial_id"]),),
        provenance_hash=SOURCE_HASH,
    )
    store.initialize()
    with pytest.raises(ResultValidationError, match="frozen configuration"):
        store.save(sample)


def test_checkpoint_rejects_assignment_rebound_under_existing_trial_id(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        experiment_id="e2",
        condition="no_probe",
        query_budget=0,
    )
    sample = _e2_sample(config)
    sample["trusted_evaluation"]["value"]["actual_hidden_state"] = "theta1"
    store = AgentDojoCheckpointStore(
        tmp_path,
        config,
        (str(sample["trial_id"]),),
        provenance_hash=SOURCE_HASH,
    )
    store.initialize()
    with pytest.raises(ResultValidationError, match="trusted assignment"):
        store.save(sample)


@pytest.mark.parametrize(
    "field",
    ("fixture_mode", "evidence_class", "scientific_evidence_eligible"),
)
def test_checkpoint_requires_complete_config_bound_evidence_labels(
    tmp_path: Path,
    field: str,
) -> None:
    store, config = _store(tmp_path)
    store.initialize()
    sample = _sample(config, "trial-0", 0)
    sample.pop(field)
    with pytest.raises(ResultValidationError, match=field):
        store.save(sample)


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("fixture_mode", 0),
        ("evidence_class", "engineering_smoke_only"),
        ("scientific_evidence_eligible", False),
    ),
)
def test_checkpoint_rejects_wrong_config_bound_evidence_labels(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    store, config = _store(tmp_path)
    store.initialize()
    sample = _sample(config, "trial-0", 0)
    sample[field] = bad_value
    with pytest.raises(ResultValidationError, match=field):
        store.save(sample)


def test_fixture_publication_labels_rows_and_manifest_as_engineering_only(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path), fixture_mode=True)
    store = AgentDojoCheckpointStore(
        tmp_path,
        config,
        ("trial-0",),
        provenance_hash=SOURCE_HASH,
    )
    store.initialize()
    store.save(_sample(config, "trial-0", 0))
    expected = {
        "fixture_mode": True,
        "evidence_class": "engineering_smoke_only",
        "scientific_evidence_eligible": False,
    }
    failure = {
        "schema_version": "silenttwin.agentdojo.failure.v1",
        "trial_id": "trial-0",
        "configuration_hash": config.configuration_hash,
        "scenario_id": config.scenario_ids[0],
        **expected,
    }
    manifest = publish_completed_run(
        store=store,
        failures=(failure,),
        started_at="2026-08-24T00:00:00Z",
        completed_at="2026-08-24T00:01:00Z",
        grid_hash=GRID_HASH,
        grid_task_id=0,
        shard_id="fixture-engineering-shard",
        provenance={"source_tree_hash": SOURCE_HASH},
    )
    assert {field: manifest[field] for field in expected} == expected
    row = json.loads((tmp_path / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert {field: row[field] for field in expected} == expected
    failure_row = json.loads(
        (tmp_path / FAILURES_FILENAME).read_text(encoding="utf-8")
    )
    assert {field: failure_row[field] for field in expected} == expected
    assert validate_completed_run(tmp_path, expected_config=config) == manifest

    failure_row["fixture_mode"] = False
    failure_path = tmp_path / FAILURES_FILENAME
    failure_path.write_text(json.dumps(failure_row) + "\n", encoding="utf-8")
    manifest["failures_sha256"] = sha256_file(failure_path)
    (tmp_path / MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ResultValidationError, match="failure row 0 fixture_mode"):
        validate_completed_run(tmp_path, expected_config=config)


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("fixture_mode", True),
        ("evidence_class", "engineering_smoke_only"),
        ("scientific_evidence_eligible", False),
    ),
)
def test_completed_run_rejects_manifest_evidence_boundary_tampering(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    _publish(tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = bad_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ResultValidationError, match=field):
        validate_completed_run(tmp_path)
