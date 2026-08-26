from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from silenttwin.agentdojo.cli import main as artifact_cli_main
from silenttwin.agentdojo.aggregate import _development_power_status, aggregate
from silenttwin.agentdojo.config import (
    AGENTDOJO_SUITES,
    CONTROLLED_MODEL_PROMPT_HASH,
    AgentDojoExperimentConfig,
    stable_hash,
)
from silenttwin.agentdojo.freeze import (
    make_development_power_evidence,
    validate_agentdojo_sample_size_freeze,
    validate_development_power_evidence,
)
from silenttwin.agentdojo.grid import (
    AgentDojoGridError,
    build_grid,
    load_grid_manifest,
    write_manifest,
)
from silenttwin.agentdojo.runner import (
    AgentDojoRunnerError,
    _validate_runtime_sample_size_freeze,
    controlled_trial_id,
)
from silenttwin.agentdojo.runtime_integrity import (
    LearnedRuntimeReport,
    learned_runtime_manifest_fingerprint,
    make_learned_runtime_provenance,
)
from silenttwin.agentdojo.storage import (
    AgentDojoCheckpointStore,
    bind_evidence_boundary,
    publish_completed_run,
)
from tests.unit.test_agentdojo_grid import REPO_ROOT, _frozen_inputs


def _captured_learned_runtime() -> dict[str, object]:
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


def _controlled_model_call(
    *,
    phase: str,
    call_index: int,
    scenario_id: str,
) -> dict[str, object]:
    parsed_output = (
        {"plan_id": "probe_0"}
        if phase == "probe"
        else {
            "prediction": "theta0",
            "posterior": {"theta0": 0.8, "theta1": 0.2},
        }
        if phase == "prediction"
        else {"plan_id": "candidate_0"}
    )
    canonical_input = {
        "request_type": phase,
        "scenario_id": scenario_id,
        "call_index": call_index,
    }
    rendered_input = json.dumps(
        canonical_input,
        sort_keys=True,
        separators=(",", ":"),
    )
    raw_response = json.dumps(
        parsed_output,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "phase": phase,
        "call_index": call_index,
        "seed": call_index,
        "canonical_input": canonical_input,
        "canonical_input_hash": stable_hash(canonical_input),
        "protocol_prompt": f"unit-test {phase} prompt",
        "rendered_chat_template_input": rendered_input,
        "rendered_input_hash": hashlib.sha256(
            rendered_input.encode("utf-8")
        ).hexdigest(),
        "raw_response": raw_response,
        "raw_response_hash": hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest(),
        "parsed_output": parsed_output,
        "parsed_output_hash": stable_hash(parsed_output),
        "latency_ms": 0.0,
        "failure_metadata": {},
        "metadata": {"transport": "unit_test_fixture"},
        "error": None,
    }


def _development_manifest(inputs: object, experiment_id: str = "e2") -> dict[str, object]:
    plan = inputs.analysis_plan
    primary = plan["primary_contrasts"][experiment_id]
    paired_outcomes = [
        {
            "agentdojo_suite": suite,
            "structural_group_id": f"{suite}-development-group",
            "target": index % 2,
            "reference": 0,
        }
        for index, suite in enumerate(AGENTDOJO_SUITES)
    ]
    evidence_digest_payload = {
        "comparisons": [],
        "accounting": None,
        "pair_yield_headroom": None,
        "paired_power_outcomes": paired_outcomes,
    }
    evidence_hash = stable_hash(evidence_digest_payload)
    power = make_development_power_evidence(
        experiment_id=experiment_id,
        primary_contrast_id=primary,
        development_evidence_hash=evidence_hash,
        power_analysis_spec=plan["development_power_analysis"],
        paired_outcomes=paired_outcomes,
    )
    payload: dict[str, object] = {
        "schema_version": "silenttwin.agentdojo.analysis_manifest.v1",
        "environment_backend": "agentdojo",
        "tier2_track": "controlled",
        "experiment_id": experiment_id,
        "fixture_mode": False,
        "evidence_class": "agentdojo_benchmark_execution",
        "scientific_evidence_eligible": True,
        "dataset_split": "development",
        "analysis_plan_hash": inputs.upstream.analysis_plan_hash,
        "grid_hash": "e" * 64,
        "grid_validation_mode": "exact_expected_grid",
        "suite_coverage_status": "full_four_suite",
        "confirmatory_suite_coverage_eligible": True,
        "sample_size_freeze_eligible": True,
        "upstream_chain_hash": inputs.upstream.binding_hash,
        "independent_unit": "structural_group_id",
        "suite_strata": list(AGENTDOJO_SUITES),
        "suite_independent_unit_counts": {
            suite: 1 for suite in AGENTDOJO_SUITES
        },
        "current_evidence_digest_payload": evidence_digest_payload,
        "current_evidence_hash": evidence_hash,
        "development_evidence_hash": evidence_hash,
        "development_power_analysis": power,
    }
    return {**payload, "analysis_manifest_hash": stable_hash(payload)}


def _freeze_cli_arguments(paths: dict[str, Path], development: Path, output: Path) -> list[str]:
    return [
        "freeze-sample-size",
        "--experiment",
        "e2",
        "--catalog",
        str(paths["catalog"]),
        "--splits",
        str(paths["splits"]),
        "--strategy-catalog",
        str(paths["strategy"]),
        "--pair-registry",
        str(paths["pair"]),
        "--analysis-plan",
        str(paths["analysis"]),
        "--dependency-lock",
        str(paths["lock"]),
        "--development-analysis-manifest",
        str(development),
        "--output",
        str(output),
        "--assert-test-results-uninspected",
    ]


def _production_grid_plan(*, runtime_fingerprint: str) -> dict[str, object]:
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text(encoding="utf-8")
    )
    plan["base_configuration"]["fixture_mode"] = False
    for name in (
        "artifact_class",
        "evidence_class",
        "scientific_evidence_eligible",
        "required_fixture_artifact_hashes",
    ):
        plan.pop(name, None)
    plan["models"] = [
        {
            "role": "attacker",
            "implementation": "local_transformers",
            "model_id": "example/immutable-attacker",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "checkpoint_fingerprint": "sha256:" + "c" * 64,
            "runtime_fingerprint": runtime_fingerprint,
            "prompt_hash": CONTROLLED_MODEL_PROMPT_HASH,
            "reasoning_mode": "disabled",
            "dtype": "bfloat16",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 64,
        }
    ]
    return plan


def test_development_power_freeze_grid_and_runtime_chain_is_model_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _frozen_inputs(tmp_path)
    development_path = tmp_path / "development-analysis.json"
    development = _development_manifest(inputs)
    development_path.write_text(
        json.dumps(development, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze_path = tmp_path / "e2-freeze.json"
    assert artifact_cli_main(
        _freeze_cli_arguments(paths, development_path, freeze_path)
    ) == 0
    # Identical regeneration is idempotent, never an overwrite.
    assert artifact_cli_main(
        _freeze_cli_arguments(paths, development_path, freeze_path)
    ) == 0
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validated = validate_agentdojo_sample_size_freeze(
        freeze,
        experiment_id="e2",
        primary_contrast_id=inputs.analysis_plan["primary_contrasts"]["e2"],
        upstream=inputs.upstream,
    )
    assert validated["claim_disposition"] == "underpowered_estimation_only"
    assert validated["independent_unit_count_by_suite"] == {
        suite: 1 for suite in AGENTDOJO_SUITES
    }

    learned_runtime = _captured_learned_runtime()
    plan = _production_grid_plan(
        runtime_fingerprint=str(learned_runtime["runtime_fingerprint"])
    )
    grid = build_grid(
        inputs=inputs,
        grid_plan=plan,
        experiment_id="e2",
        tier2_track="controlled",
        dataset_split="test",
        sample_size_freeze=freeze,
    )
    manifest_path = tmp_path / "test-grid.jsonl"
    write_manifest(grid, manifest_path)
    loaded = load_grid_manifest(manifest_path)
    binding = loaded["metadata"]["heldout_freeze_binding"]
    assert binding["freeze_hash"] == freeze["freeze_hash"]
    assert binding["development_analysis_manifest_hash"] == development[
        "analysis_manifest_hash"
    ]
    assert loaded["metadata"]["suite_coverage_status"] == "full_four_suite"
    assert all(
        row["configuration"]["sample_size_freeze_hash"] == freeze["freeze_hash"]
        for row in loaded["members"]
    )

    monkeypatch.setenv("AGENTDOJO_SAMPLE_SIZE_FREEZE", str(freeze_path))
    monkeypatch.setenv(
        "AGENTDOJO_DEVELOPMENT_ANALYSIS_MANIFEST", str(development_path)
    )
    _validate_runtime_sample_size_freeze(
        grid_metadata=loaded["metadata"],
        selected_members=loaded["members"],
        upstream=inputs.upstream,
    )

    run_root = tmp_path / "heldout-runs"
    for cell in grid.cells:
        scientific = cell.configuration
        config = AgentDojoExperimentConfig.from_mapping(scientific)
        attack = config.condition in {"genuine", "oracle"}
        call_phases = [
            *(["probe"] * config.query_budget),
            "prediction",
            *(
                []
                if config.condition in {"oracle", "random_final"}
                else ["final"]
            ),
        ]
        attacker_calls = [
            _controlled_model_call(
                phase=phase,
                call_index=index,
                scenario_id=config.scenario_ids[0],
            )
            for index, phase in enumerate(call_phases)
        ]
        rows = []
        for actual_state, donor_state in (
            ("theta0", "theta0"),
            ("theta0", "theta1"),
            ("theta1", "theta0"),
            ("theta1", "theta1"),
        ):
            trial_id = controlled_trial_id(
                config,
                scenario_id=config.scenario_ids[0],
                actual_state=actual_state,
                donor_state=donor_state,
            )
            row = {
                "schema_version": "silenttwin.agentdojo.result.v1",
                "record_type": "sample",
                "trial_id": trial_id,
                "experiment_id": "e2",
                "environment_backend": "agentdojo",
                "tier2_track": "controlled",
                "dataset_split": "test",
                "threat_mode": config.threat_mode,
                "scenario_id": config.scenario_ids[0],
                "structural_group_id": config.structural_group_ids[0],
                "agentdojo_suite": config.agentdojo_suite,
                "condition": config.condition,
                "feedback_policy": config.feedback_policy,
                "feedback_source": config.feedback_source,
                "query_budget": config.query_budget,
                "hidden_state_prediction": actual_state,
                "posterior": {
                    "theta0": 0.8 if actual_state == "theta0" else 0.2,
                    "theta1": 0.2 if actual_state == "theta0" else 0.8,
                },
                "prediction_valid": True,
                "agent_visible_transcript": [],
                "postselection_output": [],
                "postselection_delivered_to_agent": False,
                "model_provenance": {
                    "attacker": {
                        "calls": attacker_calls,
                    }
                },
                "trusted_evaluation": {
                    "schema": "silenttwin.agentdojo.trusted.v1",
                    "value": {
                        "actual_hidden_state": actual_state,
                        "donor_state": donor_state,
                        "pair_yield_class": "both",
                        "final_plan_locked_before_settlement": True,
                    },
                },
                "exact_transcript_distribution": {
                    "status": "not_applicable",
                    "method": None,
                    "reason": "learned_non_enumerable_agentdojo_backend",
                },
                "raw_attack_success": attack,
                "attack_success": attack,
                "safe": not attack,
                "utility": True,
                "safe_and_useful": not attack,
                "run_valid": True,
                "final_plan_attempt_count": 1,
                "committed_effect_ids": ["effect"] if attack else [],
                "rejected_effect_ids": [] if attack else ["effect"],
            }
            rows.append(bind_evidence_boundary(row, config=config))
        leaf = run_root / cell.shard_id
        store = AgentDojoCheckpointStore(
            leaf,
            config,
            tuple(str(row["trial_id"]) for row in rows),
            provenance_hash="source-fixture-v1",
        )
        store.initialize()
        for row in rows:
            store.save(row)
        publish_completed_run(
            store=store,
            failures=(),
            started_at="2026-08-24T00:00:00Z",
            completed_at="2026-08-24T00:00:01Z",
            grid_hash=grid.grid_hash,
            grid_task_id=next(
                task.task_id for task in grid.tasks if cell in task.cells
            ),
            shard_id=cell.shard_id,
            provenance={
                "source_tree_hash": "source-fixture-v1",
                "learned_runtime": learned_runtime,
            },
        )
    summary = aggregate(
        input_root=run_root,
        output_dir=tmp_path / "heldout-aggregate",
        expected_grid_manifest=manifest_path,
        analysis_plan_path=paths["analysis"],
    )
    assert summary["heldout_claim_disposition"] == "underpowered_estimation_only"
    assert summary["go_no_go_gates"]["confirmatory_status"] == (
        "not_confirmatory_underpowered"
    )
    heldout_analysis = json.loads(
        (tmp_path / "heldout-aggregate/analysis_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert heldout_analysis["development_evidence_hash"] == development[
        "development_evidence_hash"
    ]
    assert heldout_analysis["current_evidence_hash"] != heldout_analysis[
        "development_evidence_hash"
    ]

    corrupted = dict(freeze)
    corrupted["development_evidence_hash"] = "f" * 64
    freeze_path.write_text(
        json.dumps(corrupted, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(AgentDojoRunnerError, match="invalid runtime sample-size freeze"):
        _validate_runtime_sample_size_freeze(
            grid_metadata=loaded["metadata"],
            selected_members=loaded["members"],
            upstream=inputs.upstream,
        )


def test_heldout_rejects_subset_suites_and_development_only_experiments(
    tmp_path: Path,
) -> None:
    inputs, _ = _frozen_inputs(tmp_path)
    plan = json.loads(
        (
            REPO_ROOT
            / "configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(AgentDojoGridError, match="exact four-suite"):
        build_grid(
            inputs=inputs,
            grid_plan=plan,
            experiment_id="e2",
            tier2_track="controlled",
            dataset_split="test",
            suites=("workspace",),
            sample_size_freeze={},
        )
    for experiment, track in (("e5", "controlled"), ("ecological", "ecological")):
        with pytest.raises(AgentDojoGridError, match="development-only"):
            build_grid(
                inputs=inputs,
                grid_plan=plan,
                experiment_id=experiment,
                tier2_track=track,
                dataset_split="test",
                sample_size_freeze={},
            )


def test_freeze_cli_refuses_conflicting_existing_output(tmp_path: Path) -> None:
    inputs, paths = _frozen_inputs(tmp_path)
    development_path = tmp_path / "development-analysis.json"
    development_path.write_text(
        json.dumps(_development_manifest(inputs), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text('{"conflict":true}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        artifact_cli_main(_freeze_cli_arguments(paths, development_path, freeze_path))
    assert error.value.code == 2
    assert json.loads(freeze_path.read_text(encoding="utf-8")) == {"conflict": True}


def test_development_aggregate_power_record_is_recomputable() -> None:
    plan = json.loads(
        (
            REPO_ROOT / "configs/silenttwin/agentdojo/analysis/controlled-v1.json"
        ).read_text(encoding="utf-8")
    )
    rows = [
        {
            "agentdojo_suite": suite,
            "structural_group_id": f"{suite}-group",
            "target": index % 2,
            "reference": 0,
        }
        for index, suite in enumerate(AGENTDOJO_SUITES)
    ]
    result = _development_power_status(
        experiment_id="e2",
        dataset_split="development",
        fixture_mode=False,
        validation_mode="exact_expected_grid",
        plan=plan,
        development_evidence_hash="a" * 64,
        paired_outcomes=rows,
        heldout_binding=None,
    )
    assert result["status"] == "estimated_not_frozen"
    assert result["required_sample_size"]["target_power"] >= 0.8
    assert result["power_evidence_hash"] == stable_hash(
        {key: value for key, value in result.items() if key != "power_evidence_hash"}
    )


def test_development_subset_is_explicitly_nonfreezable() -> None:
    plan = json.loads(
        (
            REPO_ROOT / "configs/silenttwin/agentdojo/analysis/controlled-v1.json"
        ).read_text(encoding="utf-8")
    )
    result = _development_power_status(
        experiment_id="e2",
        dataset_split="development",
        fixture_mode=False,
        validation_mode="exact_expected_grid",
        confirmatory_suite_coverage_eligible=False,
        plan=plan,
        development_evidence_hash="a" * 64,
        paired_outcomes=None,
        heldout_binding=None,
    )
    assert result["status"] == "not_evaluable_incomplete_suite_coverage"
    assert result["claim_disposition"] == "development_subset_nonconfirmatory"
    assert "power_evidence_hash" not in result


def test_power_evidence_and_freeze_cli_reject_one_suite_claims(
    tmp_path: Path,
) -> None:
    inputs, paths = _frozen_inputs(tmp_path)
    primary = inputs.analysis_plan["primary_contrasts"]["e2"]
    evidence_hash = "a" * 64
    one_suite = make_development_power_evidence(
        experiment_id="e2",
        primary_contrast_id=primary,
        development_evidence_hash=evidence_hash,
        power_analysis_spec=inputs.analysis_plan["development_power_analysis"],
        paired_outcomes=[
            {
                "agentdojo_suite": "workspace",
                "structural_group_id": "workspace-development-group",
                "target": 1,
                "reference": 0,
            }
        ],
    )
    with pytest.raises(ValueError, match="all four suites"):
        validate_development_power_evidence(
            one_suite,
            experiment_id="e2",
            primary_contrast_id=primary,
            development_evidence_hash=evidence_hash,
        )

    development = _development_manifest(inputs)
    development["suite_coverage_status"] = "development_subset_nonconfirmatory"
    development["confirmatory_suite_coverage_eligible"] = False
    development["sample_size_freeze_eligible"] = False
    development.pop("analysis_manifest_hash")
    development["analysis_manifest_hash"] = stable_hash(development)
    development_path = tmp_path / "subset-development-analysis.json"
    development_path.write_text(
        json.dumps(development, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as error:
        artifact_cli_main(
            _freeze_cli_arguments(
                paths, development_path, tmp_path / "forbidden-freeze.json"
            )
        )
    assert error.value.code == 2
