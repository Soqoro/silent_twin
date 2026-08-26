"""Executable AgentDojo Tier-2 worker and pair-freeze orchestration.

Grid selection and artifact validation occur before model construction.  The
worker has no provider/API fallback: a grid member either names an explicit
CPU smoke model or a locally available immutable model checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from silenttwin.io.jsonl import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)
from silenttwin.io.provenance import collect_provenance

from .assembly import (
    assemble_controlled_backend,
    assemble_useful_work_backend,
    model_client_from_identity,
)
from .advanced import (
    ablation_contract,
    apply_authored_graph_ablation,
    channel_intervention_audit,
    settle_authored_effect_graph,
)
from .config import (
    AgentDojoExperimentConfig,
    ECOLOGICAL_SYSTEM_PROMPT,
    ModelIdentity,
    canonical_json,
    stable_hash,
)
from .controlled import ControlledExperimentSpec, run_controlled_trial
from .ecological import EcologicalExperimentSpec, run_ecological_trial
from .grid import (
    AgentDojoGridError,
    load_grid_manifest,
    validate_grid_manifest_coverage,
)
from .freeze import validate_agentdojo_sample_size_freeze
from .pair_mining import mine_pair_registry
from .pair_observations import generate_pair_observation_set
from .pipeline import StructuredControlledAttacker
from .results import AgentDojoTrialResult, ExactTranscriptEvidence, make_grades
from .seeds import SeedSchedule
from .runtime_validation import (
    validate_environment_integrity,
    validate_persistent_runtime_paths,
    validate_runtime_artifacts,
)
from .runtime_integrity import (
    capture_learned_runtime_provenance,
    not_applicable_learned_runtime_provenance,
)
from .storage import (
    AgentDojoCheckpointStore,
    MANIFEST_FILENAME,
    bind_evidence_boundary,
    publish_completed_run,
    validate_completed_run,
)
from .visibility import public_value, trusted_value
from silenttwin.backends.base import (
    BackendActionResult,
    BackendError,
    BackendErrorStage,
    EnvironmentRole,
    GuardEvaluation,
)


class AgentDojoRunnerError(RuntimeError):
    """A selected frozen shard cannot be executed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_object(path: Path | str, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentDojoRunnerError(f"cannot read {label} {candidate}: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentDojoRunnerError(f"{label} must be one JSON object")
    return value


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise AgentDojoRunnerError(f"{name} must identify a frozen artifact")
    path = Path(value)
    if not path.is_file():
        raise AgentDojoRunnerError(f"{name} is not a readable file: {path}")
    return path


def _validate_runtime_sample_size_freeze(
    *,
    grid_metadata: Mapping[str, Any],
    selected_members: Sequence[Mapping[str, Any]],
    upstream: Any,
) -> None:
    """Re-read held-out freeze/evidence artifacts before compatibility/models."""

    if grid_metadata.get("dataset_split") != "test":
        return
    binding = grid_metadata.get("heldout_freeze_binding")
    if not isinstance(binding, Mapping):
        raise AgentDojoRunnerError("held-out grid lacks its freeze binding")
    freeze_path = _required_environment_path("AGENTDOJO_SAMPLE_SIZE_FREEZE")
    freeze = _read_object(freeze_path, label="sample-size freeze")
    primary = binding.get("primary_contrast_id")
    experiment = grid_metadata.get("experiment_id")
    if not isinstance(primary, str) or not isinstance(experiment, str):
        raise AgentDojoRunnerError("held-out grid lacks experiment/contrast identity")
    development_path = _required_environment_path(
        "AGENTDOJO_DEVELOPMENT_ANALYSIS_MANIFEST"
    )
    development_raw = _read_object(
        development_path, label="development analysis manifest"
    )
    try:
        validated = validate_agentdojo_sample_size_freeze(
            freeze,
            experiment_id=experiment,
            primary_contrast_id=primary,
            upstream=upstream,
            development_analysis_manifest=development_raw,
        )
    except ValueError as exc:
        raise AgentDojoRunnerError(f"invalid runtime sample-size freeze: {exc}") from exc
    expected_binding = {
        "freeze_hash": validated["freeze_hash"],
        "development_analysis_manifest_hash": validated[
            "development_analysis_manifest_hash"
        ],
        "development_evidence_hash": validated["development_evidence_hash"],
        "power_evidence_hash": validated["power_evidence_hash"],
        "primary_contrast_id": validated["primary_contrast_id"],
        "claim_disposition": validated["claim_disposition"],
        "independent_unit_count_by_suite": validated[
            "independent_unit_count_by_suite"
        ],
        "available_test_independent_unit_count_by_suite": validated[
            "available_test_independent_unit_count_by_suite"
        ],
        "selected_test_bundle_hash_by_suite": validated[
            "selected_test_bundle_hash_by_suite"
        ],
        "selected_structural_group_ids_by_suite": validated[
            "selected_structural_group_ids_by_suite"
        ],
        "structural_minimum_shortfalls": validated[
            "structural_minimum_shortfalls"
        ],
    }
    if dict(binding) != expected_binding:
        raise AgentDojoRunnerError(
            "runtime sample-size freeze differs from grid metadata"
        )
    for index, member in enumerate(selected_members):
        config = member.get("configuration")
        if not isinstance(config, Mapping):
            raise AgentDojoRunnerError(f"held-out grid member {index} lacks configuration")
        suite = str(config.get("agentdojo_suite"))
        if (
            config.get("sample_size_freeze_hash") != validated["freeze_hash"]
            or config.get("development_evidence_hash")
            != validated["development_evidence_hash"]
            or config.get("primary_contrast_id") != primary
            or config.get("frozen_independent_unit_count")
            != validated["independent_unit_count_by_suite"].get(suite)
            or config.get("selected_test_bundle_hash")
            != validated["selected_test_bundle_hash_by_suite"].get(suite)
            or list(config.get("structural_group_ids", ()))
            != validated["selected_structural_group_ids_by_suite"].get(suite)
        ):
            raise AgentDojoRunnerError(
                f"held-out grid member {index} differs from runtime freeze"
            )


@dataclass(frozen=True, slots=True)
class _SmokeResponse:
    text: str
    metadata: Mapping[str, Any]


class DeterministicSmokeModelClient:
    """Explicit CPU smoke model; never selected as a runtime fallback.

    The model observes only the rendered public prompt and is intentionally
    weak.  Its purpose is protocol/integration validation, never empirical
    evidence about leakage or attack success.
    """

    def complete(self, prompt: str, *, seed: int, max_tokens: int) -> _SmokeResponse:
        del max_tokens
        marker = "PUBLIC_INPUT:\n"
        tool_loop_marker = "TOOL_LOOP_INPUT:\n"
        tool_filter_marker = "TOOL_FILTER_INPUT:\n"
        if tool_filter_marker in prompt:
            value = json.loads(prompt.split(tool_filter_marker, 1)[1])
            tools = value.get("tools", [])
            names = [
                str(tool.get("name"))
                for tool in tools
                if isinstance(tool, Mapping) and tool.get("name")
            ]
            return _SmokeResponse(
                text=",".join(names),
                metadata={
                    "client": "deterministic_cpu_smoke",
                    "seed": int(seed),
                    "external_api_calls": 0,
                    "evidence_class": "engineering_smoke_only",
                },
            )
        if tool_loop_marker in prompt:
            # A terminal answer exercises messages, grading, retirement, and
            # fresh-target settlement without pretending to be a capable
            # ecological victim model.
            return _SmokeResponse(
                text=canonical_json(
                    {
                        "content": "Deterministic engineering smoke completed.",
                        "tool_calls": [],
                    }
                ),
                metadata={
                    "client": "deterministic_cpu_smoke",
                    "seed": int(seed),
                    "external_api_calls": 0,
                    "evidence_class": "engineering_smoke_only",
                },
            )
        try:
            public = json.loads(prompt.split(marker, 1)[1])
        except Exception:
            # Ecological protocol uses a different marker and is handled by
            # the ecological runner rather than this controlled smoke client.
            raise AgentDojoRunnerError("smoke model received an unknown protocol prompt")
        request = public.get("request_type")
        scenario = public.get("scenario", {})
        plans = scenario.get("candidate_plans", []) if isinstance(scenario, Mapping) else []
        if request == "probe_selection":
            allowed = [row["plan_id"] for row in plans if row.get("kind") == "probe"]
            if not allowed:
                payload = {"plan_id": "invalid"}
            else:
                index = int(public.get("probe_index", 0))
                payload = {"plan_id": sorted(allowed)[index % len(allowed)]}
        elif request == "hidden_state_prediction":
            # Public-transcript-only heuristic.  It deliberately has no
            # access to theta, donor assignment, monitor profile, or verdict.
            transcript = public.get("transcript", [])
            serialized = canonical_json(transcript)
            theta1 = serialized.count("candidate_1") > serialized.count("candidate_0")
            probability = 0.75 if theta1 else 0.25
            payload = {
                "prediction": "theta1" if probability > 0.5 else "theta0",
                "posterior": {"theta0": 1.0 - probability, "theta1": probability},
            }
        elif request == "final_plan_selection":
            allowed = [
                row["plan_id"]
                for row in plans
                if row.get("kind") in {"candidate", "workflow"}
            ]
            if not allowed:
                payload = {"plan_id": "invalid"}
            else:
                payload = {"plan_id": sorted(allowed)[int(seed) % len(allowed)]}
        elif request == "postselection_observation":
            payload = {"acknowledged": True}
        else:
            raise AgentDojoRunnerError(f"unknown controlled request type {request!r}")
        return _SmokeResponse(
            text=canonical_json(payload),
            metadata={
                "client": "deterministic_cpu_smoke",
                "seed": int(seed),
                "external_api_calls": 0,
                "evidence_class": "engineering_smoke_only",
            },
        )


def _selected_members(grid: Mapping[str, Any], task_id: int) -> list[dict[str, Any]]:
    metadata = grid["metadata"]
    total = int(metadata["total_tasks"])
    if task_id < 0 or task_id >= total:
        raise AgentDojoRunnerError(
            f"task ID {task_id} is out of range; valid range is 0-{total - 1}"
        )
    members = [
        dict(row) for row in grid["members"] if int(row.get("task_id", -1)) == task_id
    ]
    members.sort(key=lambda row: int(row["batch_offset"]))
    if not members or [int(row["batch_offset"]) for row in members] != list(
        range(len(members))
    ):
        raise AgentDojoRunnerError("selected grid task is missing contiguous batch members")
    return members


def _assignment_cells(
    config: AgentDojoExperimentConfig,
) -> tuple[tuple[str, str | None], ...]:
    if config.experiment_id == "e1" and config.feedback_source == "matched_shuffled":
        return (
            ("theta0", "theta0"),
            ("theta0", "theta1"),
            ("theta1", "theta0"),
            ("theta1", "theta1"),
        )
    if config.experiment_id == "e1":
        return (("theta0", None), ("theta1", None))
    if config.experiment_id == "e2":
        return (
            ("theta0", "theta0"),
            ("theta0", "theta1"),
            ("theta1", "theta0"),
            ("theta1", "theta1"),
        )
    raise AgentDojoRunnerError(
        f"controlled executable protocol for {config.experiment_id!r} is not available"
    )


def controlled_trial_id(
    config: AgentDojoExperimentConfig,
    *,
    scenario_id: str,
    actual_state: str,
    donor_state: str | None,
) -> str:
    return stable_hash(
        {
            "protocol": "silenttwin.agentdojo.controlled.v1",
            "configuration_hash": config.configuration_hash,
            "scenario_id": scenario_id,
            "actual_state": actual_state,
            "donor_state": donor_state,
            "replicate": config.replicate,
        }
    )


def ecological_trial_id(
    config: AgentDojoExperimentConfig, *, scenario_id: str
) -> str:
    return stable_hash(
        {
            "protocol": "silenttwin.agentdojo.tool-loop.v1",
            "configuration_hash": config.configuration_hash,
            "scenario_id": scenario_id,
            "threat_mode": config.threat_mode,
            "defense": config.ecological_defense,
            "replicate": config.replicate,
        }
    )


def advanced_trial_id(
    config: AgentDojoExperimentConfig,
    *,
    scenario_id: str,
    actual_state: str,
    donor_state: str,
) -> str:
    return stable_hash(
        {
            "protocol": "silenttwin.agentdojo.advanced.v1",
            "configuration_hash": config.configuration_hash,
            "scenario_id": scenario_id,
            "actual_state": actual_state,
            "donor_state": donor_state,
            "replicate": config.replicate,
        }
    )


def useful_work_trial_id(
    config: AgentDojoExperimentConfig,
    *,
    scenario_id: str,
    actual_state: str,
) -> str:
    return stable_hash(
        {
            "protocol": "silenttwin.agentdojo.useful-work.v1",
            "configuration_hash": config.configuration_hash,
            "scenario_id": scenario_id,
            "actual_state": actual_state,
            "workflow": config.workflow,
            "settlement_runtime": config.settlement_runtime,
            "replicate": config.replicate,
        }
    )


def _scenario_index(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = catalog.get("scenarios")
    if not isinstance(rows, list):
        raise AgentDojoRunnerError("catalog lacks a scenario registry")
    result = {str(row.get("scenario_id")): row for row in rows if isinstance(row, Mapping)}
    if len(result) != len(rows):
        raise AgentDojoRunnerError("catalog scenario registry has duplicate/invalid rows")
    return result


def _model_identity(config: AgentDojoExperimentConfig, role: str) -> ModelIdentity:
    matches = [model for model in config.models if model.role == role]
    if len(matches) != 1:
        raise AgentDojoRunnerError(f"configuration requires exactly one {role} model")
    return matches[0]


def _client_for_identity(
    identity: ModelIdentity,
    *,
    cache_dir: str | None,
    device: str | None,
) -> Any:
    if identity.implementation == "deterministic_fake":
        if not identity.model_revision.startswith("builtin:deterministic-fake"):
            raise AgentDojoRunnerError("fake model identity is not the frozen smoke implementation")
        return DeterministicSmokeModelClient()
    checkpoint_path = os.environ.get(
        f"AGENTDOJO_{identity.role.upper()}_CHECKPOINT"
    )
    return model_client_from_identity(
        identity,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        device=device,
    )


def _controlled_condition(config: AgentDojoExperimentConfig) -> str:
    if config.experiment_id == "e2":
        assert config.condition is not None
        return config.condition
    # In E1 the condition is the randomized feedback source.  SilentTwin is a
    # feedback policy and takes semantic-twin precedence inside the protocol.
    return config.feedback_source


def _failure_entry(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if not record.get("errors") and record.get("run_valid") is not False:
        return None
    fixture_mode = record.get("fixture_mode")
    evidence_class = record.get("evidence_class")
    scientific_evidence_eligible = record.get("scientific_evidence_eligible")
    expected_evidence_class = (
        "engineering_smoke_only"
        if fixture_mode is True
        else "agentdojo_benchmark_execution"
    )
    if (
        not isinstance(fixture_mode, bool)
        or evidence_class != expected_evidence_class
        or scientific_evidence_eligible is not (not fixture_mode)
    ):
        raise AgentDojoRunnerError(
            "AgentDojo failure row lacks its config-bound evidence labels"
        )
    return {
        "schema_version": "silenttwin.agentdojo.failure.v1",
        "trial_id": record.get("trial_id"),
        "configuration_hash": record.get("configuration_hash"),
        "scenario_id": record.get("scenario_id"),
        "fixture_mode": fixture_mode,
        "evidence_class": evidence_class,
        "scientific_evidence_eligible": scientific_evidence_eligible,
        "error_stage": record.get("error_stage"),
        "errors": record.get("errors", []),
        "conservative_attack_success": record.get(
            "conservative_attack_success", True
        ),
    }


def _run_controlled_member(
    *,
    config: AgentDojoExperimentConfig,
    scenario_index: Mapping[str, Mapping[str, Any]],
    strategy_catalog: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
    output_dir: Path,
    grid_hash: str,
    task_id: int,
    shard_id: str,
    batch_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    assignments = _assignment_cells(config)
    expected = tuple(
        controlled_trial_id(
            config,
            scenario_id=scenario_id,
            actual_state=theta,
            donor_state=donor,
        )
        for scenario_id in config.scenario_ids
        for theta, donor in assignments
    )
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return validate_completed_run(
            output_dir,
            expected_config=config,
            expected_grid_hash=grid_hash,
            expected_shard_id=shard_id,
            expected_source_tree_hash=str(provenance["source_tree_hash"]),
        )
    store = AgentDojoCheckpointStore(
        output_dir,
        config,
        expected,
        provenance_hash=str(provenance["source_tree_hash"]),
    )
    store.initialize()
    completed = store.load()
    attacker_identity = _model_identity(config, "attacker")
    cache_dir = os.environ.get("AGENTDOJO_MODEL_CACHE")
    attacker_client: Any | None = None
    if attacker_identity.implementation != "deterministic_fake":
        attacker_client = _client_for_identity(
            attacker_identity,
            cache_dir=cache_dir,
            device=os.environ.get("ATTACKER_DEVICE", "cuda"),
        )
    monitor_clients: dict[str, Any] = {}
    if config.monitor_family != "deterministic_task_policy":
        monitor_identity = _model_identity(config, "monitor")
        monitor_client = _client_for_identity(
            monitor_identity,
            cache_dir=cache_dir,
            device=os.environ.get("MONITOR_DEVICE", "cuda"),
        )
        for profile in strategy_catalog.get("monitor_profiles", ()):
            if isinstance(profile, Mapping):
                monitor_clients[str(profile.get("profile_id"))] = monitor_client
    started_at = _utc_now()
    failures: list[dict[str, Any]] = []
    for scenario_id in config.scenario_ids:
        try:
            scenario = scenario_index[scenario_id]
        except KeyError as exc:
            raise AgentDojoRunnerError(
                f"grid references unknown scenario {scenario_id!r}"
            ) from exc
        for theta, donor in assignments:
            trial_id = controlled_trial_id(
                config,
                scenario_id=scenario_id,
                actual_state=theta,
                donor_state=donor,
            )
            if trial_id in completed:
                failure = _failure_entry(completed[trial_id])
                if failure is not None:
                    failures.append(failure)
                continue
            backend = assemble_controlled_backend(
                config=config,
                scenario=scenario,
                strategy_catalog=strategy_catalog,
                pair_registry=pair_registry,
                monitor_clients=monitor_clients,
            )
            client = (
                DeterministicSmokeModelClient()
                if attacker_identity.implementation == "deterministic_fake"
                else attacker_client
            )
            attacker = StructuredControlledAttacker(
                client,
                immutable_model_revision=attacker_identity.model_revision,
                max_tokens=attacker_identity.max_new_tokens,
            )
            spec = ControlledExperimentSpec(
                trial_id=trial_id,
                experiment_id=config.experiment_id,
                condition=_controlled_condition(config),
                feedback_policy=config.feedback_policy,
                query_budget=config.query_budget,
                actual_state=theta,
                donor_state=donor,
                replicate=config.replicate,
                user_task_id=str(scenario["user_task_id"]),
                injection_task_id=str(scenario["injection_task_id"]),
                injection_vector_id=str(scenario["injection_vector_id"]),
                threat_mode=config.threat_mode,
            )
            record = run_controlled_trial(spec, backend, attacker).to_record()
            record["configuration_hash"] = config.configuration_hash
            record = bind_evidence_boundary(record, config=config)
            store.save(record)
            completed[trial_id] = record
            failure = _failure_entry(record)
            if failure is not None:
                failures.append(failure)
    return publish_completed_run(
        store=store,
        failures=failures,
        started_at=started_at,
        completed_at=_utc_now(),
        grid_hash=grid_hash,
        grid_task_id=task_id,
        shard_id=shard_id,
        grid_batch_hash=batch_hash,
        provenance=provenance,
    )


def _run_ecological_member(
    *,
    config: AgentDojoExperimentConfig,
    scenario_index: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    grid_hash: str,
    task_id: int,
    shard_id: str,
    batch_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    expected = tuple(
        ecological_trial_id(config, scenario_id=scenario_id)
        for scenario_id in config.scenario_ids
    )
    if (output_dir / MANIFEST_FILENAME).exists():
        return validate_completed_run(
            output_dir,
            expected_config=config,
            expected_grid_hash=grid_hash,
            expected_shard_id=shard_id,
            expected_source_tree_hash=str(provenance["source_tree_hash"]),
        )
    store = AgentDojoCheckpointStore(
        output_dir,
        config,
        expected,
        provenance_hash=str(provenance["source_tree_hash"]),
    )
    store.initialize()
    completed = store.load()
    victim_identity = _model_identity(config, "victim")
    shared_client: Any | None = None
    if victim_identity.implementation != "deterministic_fake":
        shared_client = _client_for_identity(
            victim_identity,
            cache_dir=os.environ.get("AGENTDOJO_MODEL_CACHE"),
            device=os.environ.get("VICTIM_DEVICE", "cuda"),
        )
    from . import compat

    compat.assert_compatible(
        config.agentdojo_source_revision, config.agentdojo_benchmark_version
    )
    suite = compat.load_suite(
        config.agentdojo_suite,
        deployment_source_revision=config.agentdojo_source_revision,
        benchmark_version=config.agentdojo_benchmark_version,
    )
    shared_pi_detector: Any | None = None
    if (
        config.ecological_defense == "transformers_pi_detector"
        and not config.fixture_mode
    ):
        from .ecological_defenses import ReleasedTransformersPIDetectorAdapter

        detector_identity = _model_identity(config, "monitor")
        shared_pi_detector = ReleasedTransformersPIDetectorAdapter.from_identity(
            detector_identity,
            compat=compat,
            checkpoint_path=os.environ.get("AGENTDOJO_MONITOR_CHECKPOINT"),
        )
    failures: list[dict[str, Any]] = []
    started_at = _utc_now()
    for scenario_id in config.scenario_ids:
        trial_id = ecological_trial_id(config, scenario_id=scenario_id)
        if trial_id in completed:
            failure = _failure_entry(completed[trial_id])
            if failure is not None:
                failures.append(failure)
            continue
        scenario = scenario_index.get(scenario_id)
        if scenario is None:
            raise AgentDojoRunnerError(f"unknown ecological scenario {scenario_id!r}")
        if scenario.get("suite") != config.agentdojo_suite:
            raise AgentDojoRunnerError("ecological scenario suite differs from the grid")
        user_task = compat.get_user_task(suite, str(scenario["user_task_id"]))
        attacked = config.threat_mode != "clean"
        if attacked:
            if scenario.get("released_attack_name") != config.released_attack_name:
                raise AgentDojoRunnerError(
                    "ecological released_attack_name differs from the frozen catalog row"
                )
            if (
                scenario.get("released_attack_target_pipeline")
                != config.released_attack_target_pipeline
            ):
                raise AgentDojoRunnerError(
                    "ecological attack target pipeline differs from the frozen catalog row"
                )
            expected_initial_hash = scenario.get(
                "released_attack_initial_environment_hash"
            )
            expected_rendering_hash = scenario.get(
                "released_attack_rendering_hash"
            )
        else:
            expected_initial_hash = scenario.get("clean_initial_environment_hash")
            expected_rendering_hash = None
        for label, value in (
            ("expected_initial_environment_hash", expected_initial_hash),
            ("expected_injection_rendering_hash", expected_rendering_hash),
        ):
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or value.lower() != value
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise AgentDojoRunnerError(
                    f"ecological catalog row has an invalid {label}"
                )
        if expected_initial_hash is None:
            raise AgentDojoRunnerError(
                "ecological catalog row lacks its mode-specific initial hash"
            )
        injection_task = (
            compat.get_injection_task(suite, str(scenario["injection_task_id"]))
            if attacked
            else None
        )
        client = (
            DeterministicSmokeModelClient()
            if victim_identity.implementation == "deterministic_fake"
            else shared_client
        )
        assert client is not None
        spec = EcologicalExperimentSpec(
            trial_id=trial_id,
            scenario_id=scenario_id,
            suite=config.agentdojo_suite,
            structural_group_id=str(scenario["structural_group_id"]),
            dataset_split=config.dataset_split,
            user_task_id=str(scenario["user_task_id"]),
            injection_task_id=(str(scenario["injection_task_id"]) if attacked else None),
            injection_vector_id=(str(scenario["injection_vector_id"]) if attacked else None),
            threat_mode=config.threat_mode,
            defense=str(config.ecological_defense or "none"),
            released_attack_name=(config.released_attack_name if attacked else None),
            released_attack_target_pipeline=(
                config.released_attack_target_pipeline if attacked else None
            ),
            expected_initial_environment_hash=str(expected_initial_hash),
            expected_injection_rendering_hash=(
                str(expected_rendering_hash) if attacked else None
            ),
            replicate=config.replicate,
            victim_model_revision=victim_identity.model_revision,
            system_prompt=ECOLOGICAL_SYSTEM_PROMPT,
        )
        tool_filter = None
        if config.ecological_defense == "tool_filter":
            from .ecological_defenses import AdaptedLocalToolFilter

            tool_filter = AdaptedLocalToolFilter(
                client,
                identity=victim_identity,
                seed=SeedSchedule(
                    scenario_id,
                    victim_identity.model_revision,
                    config.replicate,
                ).ecological_tool_filter(),
                released_prompt=compat.released_tool_filter_prompt(),
            )
        result = run_ecological_trial(
            spec,
            suite=suite,
            user_task=user_task,
            injection_task=injection_task,
            model_client=client,
            compat=compat,
            pi_detector=shared_pi_detector,
            tool_filter=tool_filter,
            package_version=config.agentdojo_package_version,
            source_revision=config.agentdojo_source_revision,
            benchmark_version=config.agentdojo_benchmark_version,
            catalog_hash=config.agentdojo_catalog_hash,
        )
        record = result.to_record()
        record["configuration_hash"] = config.configuration_hash
        record["ecological_attack"] = config.ecological_attack
        record["released_attack_name"] = config.released_attack_name
        record["released_attack_target_pipeline"] = (
            config.released_attack_target_pipeline
        )
        record["ecological_defense"] = config.ecological_defense
        record["settlement_runtime"] = config.settlement_runtime
        record = bind_evidence_boundary(record, config=config)
        store.save(record)
        completed[trial_id] = record
        failure = _failure_entry(record)
        if failure is not None:
            failures.append(failure)
    return publish_completed_run(
        store=store,
        failures=failures,
        started_at=started_at,
        completed_at=_utc_now(),
        grid_hash=grid_hash,
        grid_task_id=task_id,
        shard_id=shard_id,
        grid_batch_hash=batch_hash,
        provenance=provenance,
    )


_E5_CHANNELS = {
    "visible_settlement": "visible_settlement",
    "timing_leak": "timing",
    "identifier_leak": "identifiers",
    "error_schema_leak": "error_schema",
    "verdict_dependent_termination": "verdict_dependent_termination",
    "callback_leak": "callbacks",
    "later_world_read": "later_environment_read",
    "retained_memory": "retained_memory",
    "retained_kv_cache": "retained_kv_cache",
    "public_fidelity_degradation": "public_fidelity_degradation",
    "decision_dependent_twin_mismatch": "decision_dependent_twin_mismatch",
}

_E5_AUTHORED_GRAPH_ABLATIONS = {
    "incomplete_data_dependencies",
    "incomplete_control_dependencies",
    "missing_atomic_group",
}


def _advanced_record(
    *,
    config: AgentDojoExperimentConfig,
    base_result: Any,
) -> dict[str, Any]:
    if config.experiment_id == "e3":
        label = str(config.closure_channel)
        channel = label
        contract = None
    else:
        label = str(config.ablation)
        contract = ablation_contract(label, backend_exposes_kv_cache=False)
        channel = _E5_CHANNELS.get(label, "exact_silenttwin")
    trusted_document = (
        base_result.trusted_evaluation.to_trusted_dict()["value"]
        if base_result.trusted_evaluation is not None
        else {}
    )
    live_intervention = trusted_document.get("channel_intervention")
    if not isinstance(live_intervention, Mapping):
        raise AgentDojoRunnerError(
            "advanced row lacks its live controlled-trial channel evidence"
        )
    evidence = live_intervention.get("operational_evidence")
    if not isinstance(evidence, Mapping):
        raise AgentDojoRunnerError(
            "advanced row lacks operational channel-boundary evidence"
        )
    intervention = channel_intervention_audit(
        channel=channel,
        operational_evidence=evidence,
    )
    if dict(live_intervention) != intervention:
        raise AgentDojoRunnerError(
            "advanced row channel evidence changed after live execution"
        )
    trusted_document = {
        **dict(trusted_document),
        "advanced_protocol_revision": "silenttwin.agentdojo.advanced.v1",
        "channel_intervention": intervention,
        "ablation_contract": contract,
    }
    result = replace(
        base_result,
        experiment_id=config.experiment_id,
        condition=label,
        postselection_delivered_to_agent=bool(
            intervention["postselection_delivered_to_agent"]
        ),
        trusted_evaluation=trusted_value(trusted_document),
        model_provenance={
            **dict(base_result.model_provenance),
            "advanced_protocol_revision": "silenttwin.agentdojo.advanced.v1",
            "channel_intervention": intervention,
            "ablation_contract": contract,
        },
    )
    record = result.to_record()
    record["configuration_hash"] = config.configuration_hash
    if record["postselection_delivered_to_agent"] != bool(
        intervention["postselection_delivered_to_agent"]
    ):
        raise AgentDojoRunnerError(
            "result delivery flag differs from live channel evidence"
        )
    if config.experiment_id == "e3":
        record["closure_channel"] = label
        record["channel_intervention"] = intervention
    else:
        record["ablation"] = label
        record["ablation_contract"] = contract
    return bind_evidence_boundary(record, config=config)


def _run_advanced_member(
    *,
    config: AgentDojoExperimentConfig,
    scenario_index: Mapping[str, Mapping[str, Any]],
    strategy_catalog: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
    output_dir: Path,
    grid_hash: str,
    task_id: int,
    shard_id: str,
    batch_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if config.experiment_id not in {"e3", "e5"}:
        raise AgentDojoRunnerError("advanced runner supports E3 or E5")
    if config.feedback_policy != "silenttwin":
        raise AgentDojoRunnerError("E3/E5 advanced rows require SilentTwin feedback")
    structural_graph_ablation = (
        config.experiment_id == "e5"
        and config.ablation in _E5_AUTHORED_GRAPH_ABLATIONS
    )
    assignments = (
        (("theta0", "theta0"), ("theta1", "theta1"))
        if structural_graph_ablation
        else (
            ("theta0", "theta0"),
            ("theta0", "theta1"),
            ("theta1", "theta0"),
            ("theta1", "theta1"),
        )
    )
    expected = tuple(
        advanced_trial_id(
            config,
            scenario_id=scenario_id,
            actual_state=theta,
            donor_state=donor,
        )
        for scenario_id in config.scenario_ids
        for theta, donor in assignments
    )
    if (output_dir / MANIFEST_FILENAME).exists():
        return validate_completed_run(
            output_dir,
            expected_config=config,
            expected_grid_hash=grid_hash,
            expected_shard_id=shard_id,
            expected_source_tree_hash=str(provenance["source_tree_hash"]),
        )
    store = AgentDojoCheckpointStore(
        output_dir,
        config,
        expected,
        provenance_hash=str(provenance["source_tree_hash"]),
    )
    store.initialize()
    completed = store.load()
    attacker_identity = (
        None if structural_graph_ablation else _model_identity(config, "attacker")
    )
    shared_attacker_client: Any | None = None
    if (
        not structural_graph_ablation
        and attacker_identity is not None
        and attacker_identity.implementation != "deterministic_fake"
    ):
        shared_attacker_client = _client_for_identity(
            attacker_identity,
            cache_dir=os.environ.get("AGENTDOJO_MODEL_CACHE"),
            device=os.environ.get("ATTACKER_DEVICE", "cuda"),
        )
    monitor_clients: dict[str, Any] = {}
    if config.monitor_family != "deterministic_task_policy":
        monitor_identity = _model_identity(config, "monitor")
        monitor_client = _client_for_identity(
            monitor_identity,
            cache_dir=os.environ.get("AGENTDOJO_MODEL_CACHE"),
            device=os.environ.get("MONITOR_DEVICE", "cuda"),
        )
        for profile in strategy_catalog.get("monitor_profiles", ()):
            if isinstance(profile, Mapping):
                monitor_clients[str(profile.get("profile_id"))] = monitor_client
    failures: list[dict[str, Any]] = []
    started_at = _utc_now()
    for scenario_id in config.scenario_ids:
        scenario = scenario_index.get(scenario_id)
        if scenario is None:
            raise AgentDojoRunnerError(f"unknown advanced scenario {scenario_id!r}")
        for theta, donor in assignments:
            trial_id = advanced_trial_id(
                config,
                scenario_id=scenario_id,
                actual_state=theta,
                donor_state=donor,
            )
            if trial_id in completed:
                failure = _failure_entry(completed[trial_id])
                if failure is not None:
                    failures.append(failure)
                continue
            if structural_graph_ablation:
                assert config.ablation is not None
                record = _run_useful_work_trial(
                    config=config,
                    scenario=scenario,
                    actual_state=theta,
                    trial_id=trial_id,
                    strategy_catalog=strategy_catalog,
                    pair_registry=pair_registry,
                    monitor_clients=monitor_clients,
                    workflow_override="atomic",
                    graph_ablation=config.ablation,
                    donor_state=donor,
                )
            else:
                backend = assemble_controlled_backend(
                    config=config,
                    scenario=scenario,
                    strategy_catalog=strategy_catalog,
                    pair_registry=pair_registry,
                    monitor_clients=monitor_clients,
                )
                client = (
                    DeterministicSmokeModelClient()
                    if attacker_identity is not None
                    and attacker_identity.implementation == "deterministic_fake"
                    else shared_attacker_client
                )
                assert client is not None
                attacker = StructuredControlledAttacker(
                    client,
                    immutable_model_revision=attacker_identity.model_revision,
                    max_tokens=attacker_identity.max_new_tokens,
                )
                base_spec = ControlledExperimentSpec(
                    trial_id=trial_id,
                    experiment_id="e2",
                    condition="silenttwin",
                    feedback_policy="silenttwin",
                    query_budget=config.query_budget,
                    actual_state=theta,
                    donor_state=donor,
                    replicate=config.replicate,
                    user_task_id=str(scenario["user_task_id"]),
                    injection_task_id=str(scenario["injection_task_id"]),
                    injection_vector_id=str(scenario["injection_vector_id"]),
                    threat_mode=config.threat_mode,
                    channel_intervention=(
                        str(config.closure_channel)
                        if config.experiment_id == "e3"
                        else _E5_CHANNELS.get(
                            str(config.ablation), "exact_silenttwin"
                        )
                    ),
                )
                record = _advanced_record(
                    config=config,
                    base_result=run_controlled_trial(base_spec, backend, attacker),
                )
            store.save(record)
            completed[trial_id] = record
            failure = _failure_entry(record)
            if failure is not None:
                failures.append(failure)
    return publish_completed_run(
        store=store,
        failures=failures,
        started_at=started_at,
        completed_at=_utc_now(),
        grid_hash=grid_hash,
        grid_task_id=task_id,
        shard_id=shard_id,
        grid_batch_hash=batch_hash,
        provenance=provenance,
    )


def _run_useful_work_trial(
    *,
    config: AgentDojoExperimentConfig,
    scenario: Mapping[str, Any],
    actual_state: str,
    trial_id: str,
    strategy_catalog: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
    monitor_clients: Mapping[str, Any],
    workflow_override: str | None = None,
    graph_ablation: str | None = None,
    donor_state: str | None = None,
) -> dict[str, Any]:
    latency_started = time.perf_counter()
    backend, source_effects = assemble_useful_work_backend(
        config=config,
        scenario=scenario,
        strategy_catalog=strategy_catalog,
        pair_registry=pair_registry,
        monitor_clients=monitor_clients,
        workflow_override=workflow_override,
    )
    effects = source_effects
    ablation_audit: Mapping[str, Any] | None = None
    if graph_ablation is not None:
        effects, ablation_audit = apply_authored_graph_ablation(
            source_effects, ablation=graph_ablation
        )
    seed_base = int(stable_hash([trial_id, "e4-monitor"])[:15], 16)
    base = backend.fresh_environment(actual_state, EnvironmentRole.BASE, seed_base)
    initial_hash = backend.environment_hash(base)
    speculative = backend.clone_environment(base, EnvironmentRole.SEMANTIC_TWIN)
    staged_results: list[BackendActionResult] = []
    errors: list[BackendError] = []
    for index, effect in enumerate(effects):
        result = backend.execute_speculative(
            speculative, effect.plan_id, seed=seed_base + index
        )
        staged_results.append(result)
        errors.extend(result.errors)
    if backend.environment_hash(base) != initial_hash:
        raise AgentDojoRunnerError("E4 speculative workflow mutated the frozen target base")
    evaluations: dict[str, GuardEvaluation] = {}
    for index, effect in enumerate(effects):
        try:
            evaluation = backend.evaluate_guard(
                actual_state,
                effect.plan_id,
                seed=seed_base + 10_000 + index,
            )
        except Exception as exc:
            monitor_pair = getattr(backend, "monitor_pair", None)
            selected_monitor = (
                monitor_pair.for_theta(actual_state)
                if monitor_pair is not None
                and callable(getattr(monitor_pair, "for_theta", None))
                else None
            )
            profile = getattr(selected_monitor, "profile", None)
            evaluation = GuardEvaluation(
                plan_id=effect.plan_id,
                profile_id=str(getattr(profile, "profile_id", "monitor-error")),
                monitor_family=str(getattr(profile, "family", "unavailable")),
                decision="error",
                provenance={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "failed_model_call": getattr(
                        selected_monitor, "last_failure_provenance", None
                    ),
                    "prior_effect_evaluation_ids": list(evaluations),
                },
            )
            errors.append(
                BackendError(
                    BackendErrorStage.MONITOR,
                    "useful_work_monitor_exception",
                    f"{effect.effect_id}:{type(exc).__name__}:{exc}",
                )
            )
        evaluations[effect.effect_id] = evaluation
    for effect_id, evaluation in evaluations.items():
        if evaluation.decision == "error":
            errors.append(
                BackendError(
                    BackendErrorStage.MONITOR,
                    "useful_work_monitor_error",
                    f"authored effect {effect_id} received an error verdict",
                )
            )
    retired_before_settlement = True
    for label, environment in (("semantic_twin", speculative), ("base", base)):
        try:
            backend.retire_environment(environment)
        except Exception as exc:
            retired_before_settlement = False
            errors.append(
                BackendError(
                    BackendErrorStage.RETIREMENT,
                    "useful_work_context_retirement_error",
                    f"{label}:{type(exc).__name__}:{exc}",
                )
            )
    decisions = {
        effect.effect_id: (
            "error"
            if not retired_before_settlement or staged_results[index].errors
            else evaluations[effect.effect_id].decision
        )
        for index, effect in enumerate(effects)
    }
    source_report = (
        settle_authored_effect_graph(
            source_effects,
            monitor_decisions=decisions,
            runtime=config.settlement_runtime,
        )
        if ablation_audit is not None
        else None
    )
    report = settle_authored_effect_graph(
        effects,
        monitor_decisions=decisions,
        runtime=config.settlement_runtime,
        authored_source_effects=(source_effects if ablation_audit is not None else None),
        ablation_audit=ablation_audit,
    )
    final_target = backend.fresh_environment(
        actual_state, EnvironmentRole.FINAL_TARGET, seed_base
    )
    final_start_hash = backend.environment_hash(final_target)
    if final_start_hash != initial_hash:
        raise AgentDojoRunnerError("E4 final target does not match its frozen initial state")
    settlement_working = backend.clone_environment(
        final_target, EnvironmentRole.FINAL_TARGET
    )
    committed_set = set(report.committed_effect_ids)
    final_parts: list[BackendActionResult] = []
    for index, effect in enumerate(effects):
        if effect.effect_id not in committed_set:
            continue
        # Guard decisions are frozen above.  Final publication replays only
        # benchmark-authored tool semantics on a disposable clone; asking the
        # monitor again would introduce a second, differently seeded decision.
        part = backend.execute_speculative(
            settlement_working,
            effect.plan_id,
            seed=seed_base + 20_000 + index,
        )
        final_parts.append(part)
        errors.extend(part.errors)
        if part.errors:
            break
    replay_failed = any(part.errors for part in final_parts)
    if not replay_failed:
        final_target.environment = settlement_working.environment
    else:
        # Atomic publication across the selected authored branches.
        committed_set.clear()
    backend.retire_environment(settlement_working)
    staged_effect_ids = tuple(effect.effect_id for effect in effects)
    committed_effect_ids = tuple(
        effect.effect_id for effect in effects if effect.effect_id in committed_set
    )
    rejected_effect_ids = tuple(
        effect.effect_id
        for effect in effects
        if effect.effect_id not in committed_set
    )
    all_actions = tuple(
        replace(
            action,
            action_id=f"{effect.effect_id}:{action.action_id}",
            effect_id=effect.effect_id,
            status="error" if action.status == "error" else "committed",
            ground_truth_label=effect.ground_truth_label,
            data_dependencies=effect.data_dependencies,
            control_dependencies=effect.control_dependencies,
            atomic_group=effect.atomic_group,
        )
        for effect, part in zip(
            (effect for effect in effects if effect.effect_id in set(report.committed_effect_ids)),
            final_parts,
        )
        for action in part.trusted_actions
    )
    if replay_failed:
        # The authored workflow is one atomic publication boundary.  Calls
        # may have succeeded on the disposable settlement clone, but none of
        # those effects committed to the fresh target after a later error.
        all_actions = tuple(
                replace(
                    action,
                    status=(
                        "error"
                        if action.status == "error"
                        else "atomicity_discarded"
                    ),
                )
            for action in all_actions
        )
    raw_trace = tuple(
        trace
        for part in final_parts
        for trace in (part.raw_trace if isinstance(part.raw_trace, Sequence) else ())
    )
    grader_trace = () if replay_failed else raw_trace
    staged_actions = [
        replace(
            action,
            action_id=f"staged:{effect.effect_id}:{action.action_id}",
            effect_id=effect.effect_id,
            ground_truth_label=effect.ground_truth_label,
            data_dependencies=effect.data_dependencies,
            control_dependencies=effect.control_dependencies,
            atomic_group=effect.atomic_group,
        ).to_trusted_dict()
        for effect, staged in zip(effects, staged_results)
        for action in staged.trusted_actions
    ]
    tool_call_count = sum(
        len(part.raw_trace)
        if isinstance(part.raw_trace, Sequence)
        and not isinstance(part.raw_trace, (str, bytes, bytearray))
        else 0
        for part in (*staged_results, *final_parts)
    )
    monitor_token_count = sum(
        int(
            call.get("provenance", {})
            .get("model_metadata", {})
            .get("usage", {})
            .get("total_tokens", 0)
        )
        for evaluation in evaluations.values()
        for call in evaluation.provenance.get("call_evaluations", ())
        if isinstance(call, Mapping)
    )
    summary_guard = GuardEvaluation(
        plan_id="authored_mixed_workflow",
        profile_id="suite-monitor-pair",
        monitor_family=config.monitor_family,
        decision=(
            "error"
            if errors
            else "block"
            if any(value != "allow" for value in decisions.values())
            else "allow"
        ),
        provenance={
            "pair_registry_hash": config.pair_registry_hash,
            "evaluation_hash": stable_hash(
                {key: value.to_trusted_dict() for key, value in evaluations.items()}
            ),
        },
    )
    postselection = (
        public_value(
            {
                "event_type": "transaction_finalization",
                "clean_report": dict(report.clean_report),
            }
        ),
    )
    final_result = BackendActionResult(
        plan_id="authored_mixed_workflow",
        public_events=postselection,
        trusted_actions=all_actions,
        pre_environment_hash=final_start_hash,
        post_environment_hash=backend.environment_hash(final_target),
        trace_hash=stable_hash(
            {
                "authored_graph_hash": report.graph_hash,
                "committed": list(committed_effect_ids),
                "parts": [part.trace_hash for part in final_parts],
            }
        ),
        monitor=summary_guard,
        staged_effect_ids=staged_effect_ids,
        committed_effect_ids=committed_effect_ids,
        rejected_effect_ids=rejected_effect_ids,
        errors=tuple(errors),
        raw_trace=grader_trace,
        output_text=final_parts[-1].output_text if final_parts else "",
    )
    grades = backend.grade(final_target, final_result)
    if errors:
        grades = make_grades(
            utility=grades.utility,
            raw_security_result=grades.raw_attack_success,
            errors=tuple((*grades.errors, *errors)),
        )
    visible = tuple(
        event for result in staged_results for event in result.public_events
    )
    delivered_finalization = (
        config.settlement_runtime == "visible_transactional_finalization"
    )
    if delivered_finalization:
        visible = tuple((*visible, *postselection))
    permitted_staged = {
        effect.effect_id
        for effect in effects
        if effect.ground_truth_label == "permitted"
    }
    permitted_committed = permitted_staged & set(committed_effect_ids)
    safe_effect_salvage = (
        len(permitted_committed) / len(permitted_staged)
        if permitted_staged
        else 1.0
    )
    actual_clean_report = {
        effect.effect_id: (
            "committed"
            if effect.effect_id in committed_set
            else "rejected"
            if effect.effect_id in set(report.rejected_effect_ids)
            else "dependency_discarded"
            if effect.effect_id in set(report.dependency_discarded_effect_ids)
            else "atomicity_discarded"
        )
        for effect in effects
    }
    result = AgentDojoTrialResult(
        experiment_id=config.experiment_id,
        tier2_track="controlled",
        trial_id=trial_id,
        identity=backend.identity,
        scenario=backend.public_scenario,
        user_task_id=str(scenario["user_task_id"]),
        injection_task_id=str(scenario["injection_task_id"]),
        injection_vector_id=str(scenario["injection_vector_id"]),
        threat_mode=config.threat_mode,
        condition=(
            str(config.ablation)
            if config.experiment_id == "e5"
            else f"{workflow_override or config.workflow}:{config.settlement_runtime}"
        ),
        feedback_policy="silenttwin",
        feedback_source="genuine",
        query_budget=config.query_budget,
        actual_state=actual_state,
        donor_state=donor_state,
        prediction=None,
        posterior=None,
        prediction_valid=None,
        probe_plan_ids=(),
        final_plan_id="authored_mixed_workflow",
        final_plan_attempt_count=1,
        final_result=final_result,
        grades=grades,
        public_transcript=visible,
        postselection_output=postselection,
        initial_environment_hash=initial_hash,
        final_start_environment_hash=final_start_hash,
        final_environment_hash=backend.environment_hash(final_target),
        trace_hash=final_result.trace_hash,
        exact_transcript_distribution=ExactTranscriptEvidence.learned_not_applicable(),
        retired_before_settlement=retired_before_settlement,
        postselection_delivered_to_agent=delivered_finalization,
        model_provenance={
            "protocol_revision": (
                "silenttwin.agentdojo.authored-graph-ablation.v1"
                if graph_ablation is not None
                else "silenttwin.agentdojo.useful-work.v1"
            ),
            "generation_calls": 0,
            "tool_call_count": tool_call_count,
            "monitor_token_count": monitor_token_count,
        },
        trusted_evaluation=trusted_value(
            {
                "actual_hidden_state": actual_state,
                "authored_effect_graph": [effect.to_dict() for effect in source_effects],
                "effective_settlement_graph": [effect.to_dict() for effect in effects],
                "graph_ablation_audit": dict(ablation_audit) if ablation_audit else None,
                "source_settlement_report": (
                    source_report.to_dict() if source_report is not None else None
                ),
                "donor_state": donor_state,
                "authored_selection_report": report.to_dict(),
                "settlement_report": {
                    **report.to_dict(),
                    "committed_effect_ids": list(committed_effect_ids),
                    "rejected_effect_ids": list(rejected_effect_ids),
                    "clean_report": actual_clean_report,
                    "safe_effect_salvage": safe_effect_salvage,
                    "atomic_publication_error": replay_failed,
                },
                "monitor_evaluations": {
                    key: value.to_trusted_dict() for key, value in evaluations.items()
                },
                "speculative_staged_actions": staged_actions,
                "discarded_replay_trace": list(raw_trace) if replay_failed else [],
                "runtime_dependency_metadata_ignored": True,
                "final_plan_locked_before_settlement": True,
            }
        ),
        errors=tuple(errors),
    )
    record = result.to_record()
    record.update(
        {
            "configuration_hash": config.configuration_hash,
            "workflow": workflow_override or config.workflow,
            "settlement_runtime": config.settlement_runtime,
            "safe_effect_salvage": safe_effect_salvage,
            "rejected_descendant_commit_count": len(
                set(report.dependency_discarded_effect_ids) & set(committed_effect_ids)
            ),
            "atomicity_violation_count": len(
                set(report.atomicity_discarded_effect_ids) & set(committed_effect_ids)
            ),
            "clean_report_accuracy": sum(
                report.clean_report.get(effect.effect_id)
                == actual_clean_report[effect.effect_id]
                for effect in effects
            )
            / len(effects),
            "tool_call_count": tool_call_count,
            "token_count": monitor_token_count,
            "latency_seconds": time.perf_counter() - latency_started,
        }
    )
    if graph_ablation is not None:
        assert source_report is not None
        record["ablation"] = graph_ablation
        record["ablation_contract"] = {
            **dict(ablation_audit or {}),
            "source_committed_effect_ids": list(source_report.committed_effect_ids),
            "transformed_committed_effect_ids": list(report.committed_effect_ids),
            "settlement_outcome_changed": (
                source_report.committed_effect_ids != report.committed_effect_ids
            ),
        }
        record["source_safe_effect_salvage"] = source_report.safe_effect_salvage
        record["safe_effect_salvage_delta_from_complete_graph"] = (
            safe_effect_salvage - source_report.safe_effect_salvage
        )
    backend.retire_environment(final_target)
    return bind_evidence_boundary(record, config=config)


def _run_useful_work_member(
    *,
    config: AgentDojoExperimentConfig,
    scenario_index: Mapping[str, Mapping[str, Any]],
    strategy_catalog: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
    output_dir: Path,
    grid_hash: str,
    task_id: int,
    shard_id: str,
    batch_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    expected = tuple(
        useful_work_trial_id(
            config, scenario_id=scenario_id, actual_state=theta
        )
        for scenario_id in config.scenario_ids
        for theta in ("theta0", "theta1")
    )
    if (output_dir / MANIFEST_FILENAME).exists():
        return validate_completed_run(
            output_dir,
            expected_config=config,
            expected_grid_hash=grid_hash,
            expected_shard_id=shard_id,
            expected_source_tree_hash=str(provenance["source_tree_hash"]),
        )
    store = AgentDojoCheckpointStore(
        output_dir,
        config,
        expected,
        provenance_hash=str(provenance["source_tree_hash"]),
    )
    store.initialize()
    completed = store.load()
    monitor_clients: dict[str, Any] = {}
    if config.monitor_family != "deterministic_task_policy":
        monitor_identity = _model_identity(config, "monitor")
        client = _client_for_identity(
            monitor_identity,
            cache_dir=os.environ.get("AGENTDOJO_MODEL_CACHE"),
            device=os.environ.get("MONITOR_DEVICE", "cuda"),
        )
        for profile in strategy_catalog.get("monitor_profiles", ()):
            if isinstance(profile, Mapping):
                monitor_clients[str(profile.get("profile_id"))] = client
    failures: list[dict[str, Any]] = []
    started_at = _utc_now()
    for scenario_id in config.scenario_ids:
        scenario = scenario_index.get(scenario_id)
        if scenario is None:
            raise AgentDojoRunnerError(f"unknown E4 scenario {scenario_id!r}")
        for theta in ("theta0", "theta1"):
            trial_id = useful_work_trial_id(
                config, scenario_id=scenario_id, actual_state=theta
            )
            if trial_id in completed:
                failure = _failure_entry(completed[trial_id])
                if failure is not None:
                    failures.append(failure)
                continue
            record = _run_useful_work_trial(
                config=config,
                scenario=scenario,
                actual_state=theta,
                trial_id=trial_id,
                strategy_catalog=strategy_catalog,
                pair_registry=pair_registry,
                monitor_clients=monitor_clients,
            )
            store.save(record)
            completed[trial_id] = record
            failure = _failure_entry(record)
            if failure is not None:
                failures.append(failure)
    return publish_completed_run(
        store=store,
        failures=failures,
        started_at=started_at,
        completed_at=_utc_now(),
        grid_hash=grid_hash,
        grid_task_id=task_id,
        shard_id=shard_id,
        grid_batch_hash=batch_hash,
        provenance=provenance,
    )


def run_grid_task(*, grid_manifest: Path | str, task_id: int) -> list[dict[str, Any]]:
    grid = load_grid_manifest(grid_manifest)
    members = _selected_members(grid, task_id)
    metadata = grid["metadata"]
    grid_hash = str(metadata["grid_hash"])
    output_root = Path(
        os.environ.get(
            "AGENTDOJO_TASK_OUTPUT_DIR",
            f"outputs/silenttwin/agentdojo/{metadata['experiment_id']}/runs/task-{task_id}",
        )
    )
    dependency_lock_path = _required_environment_path(
        "AGENTDOJO_DEPENDENCY_LOCK"
    )
    frozen = validate_runtime_artifacts(
        catalog_path=_required_environment_path("AGENTDOJO_CATALOG"),
        splits_path=_required_environment_path("AGENTDOJO_SPLITS"),
        strategy_catalog_path=_required_environment_path(
            "AGENTDOJO_STRATEGY_CATALOG"
        ),
        pair_registry_path=_required_environment_path("AGENTDOJO_PAIR_REGISTRY"),
        analysis_plan_path=_required_environment_path("AGENTDOJO_ANALYSIS_PLAN"),
        dependency_lock_path=dependency_lock_path,
        grid_metadata=metadata,
        selected_members=members,
    )
    try:
        validate_grid_manifest_coverage(grid, frozen.analysis_plan)
    except AgentDojoGridError as error:
        raise AgentDojoRunnerError(
            f"grid violates preregistered run-stage coverage: {error}"
        ) from error
    _validate_runtime_sample_size_freeze(
        grid_metadata=metadata,
        selected_members=members,
        upstream=frozen.upstream,
    )
    catalog = frozen.catalog
    strategy_catalog = frozen.strategy_catalog
    pair_registry = frozen.pair_registry
    scenario_index = _scenario_index(catalog)
    # Release/API/suite validation is deliberately complete before any local
    # victim, attacker, or monitor checkpoint is constructed.
    from . import compat

    compat.assert_compatible(
        frozen.upstream.source_revision, frozen.upstream.benchmark_version
    )
    selected_scenario_ids = {
        str(scenario_id)
        for member in members
        for scenario_id in member["configuration"]["scenario_ids"]
    }
    suites: dict[str, Any] = {}
    for scenario_id in sorted(selected_scenario_ids):
        scenario = scenario_index[scenario_id]
        suite_name = str(scenario["suite"])
        suite = suites.get(suite_name)
        if suite is None:
            suite = compat.load_suite(
                suite_name,
                deployment_source_revision=frozen.upstream.source_revision,
                benchmark_version=frozen.upstream.benchmark_version,
            )
            suites[suite_name] = suite
        compat.validate_frozen_scenario_row(suite, scenario)
    learned_runtime_fingerprints = {
        str(model.get("runtime_fingerprint"))
        for member in members
        for model in member["configuration"].get("models", ())
        if isinstance(model, Mapping)
        and model.get("implementation")
        in {"local_transformers", "transformers_pi_detector"}
    }
    base_provenance = collect_provenance()
    task_learned_runtime = capture_learned_runtime_provenance(
        dependency_lock_path,
        expected_runtime_fingerprints=learned_runtime_fingerprints,
    )
    manifests: list[dict[str, Any]] = []
    for member in members:
        scientific = member.get("configuration")
        if not isinstance(scientific, Mapping):
            raise AgentDojoRunnerError("grid member lacks a scientific configuration")
        shard_id = str(member["shard_id"])
        output_dir = output_root / f"batch-{int(member['batch_offset']):03d}-{shard_id[:16]}"
        cache_paths = {
            key: value
            for key, value in {
                "model": os.environ.get("AGENTDOJO_MODEL_CACHE"),
                "huggingface": os.environ.get("HF_HOME"),
            }.items()
            if value
        }
        config = AgentDojoExperimentConfig(
            **dict(scientific),
            output_dir=output_dir,
            cache_paths=cache_paths,
            grid_hash=grid_hash,
            grid_task_id=task_id,
            shard_id=shard_id,
            overwrite=os.environ.get("AGENTDOJO_OVERWRITE", "0") == "1",
        )
        if config.configuration_hash != member.get("configuration_hash"):
            raise AgentDojoRunnerError("selected grid member configuration hash changed")
        member_has_learned_runtime = any(
            model.implementation
            in {"local_transformers", "transformers_pi_detector"}
            for model in config.models
        )
        provenance = {
            **base_provenance,
            "learned_runtime": (
                task_learned_runtime
                if member_has_learned_runtime
                else not_applicable_learned_runtime_provenance()
            ),
        }
        common = dict(
                config=config,
                scenario_index=scenario_index,
                output_dir=output_dir,
                grid_hash=grid_hash,
                task_id=task_id,
                shard_id=shard_id,
                batch_hash=stable_hash(
                    [
                        (row["configuration_hash"], row["shard_id"])
                        for row in members
                    ]
                ),
                provenance=provenance,
        )
        if config.experiment_id in {"e1", "e2"}:
            manifests.append(
                _run_controlled_member(
                    **common,
                    strategy_catalog=strategy_catalog,
                    pair_registry=pair_registry,
                )
            )
        elif config.experiment_id == "ecological":
            manifests.append(_run_ecological_member(**common))
        elif config.experiment_id in {"e3", "e5"}:
            manifests.append(
                _run_advanced_member(
                    **common,
                    strategy_catalog=strategy_catalog,
                    pair_registry=pair_registry,
                )
            )
        elif config.experiment_id == "e4":
            manifests.append(
                _run_useful_work_member(
                    **common,
                    strategy_catalog=strategy_catalog,
                    pair_registry=pair_registry,
                )
            )
        else:
            raise AgentDojoRunnerError(
                f"{config.experiment_id} runner is not implemented by this worker"
            )
    return manifests


def _mine_pairs(args: argparse.Namespace) -> dict[str, Any]:
    catalog = _read_object(args.catalog, label="catalog")
    splits = _read_object(args.splits, label="split manifest")
    strategies = _read_object(args.strategy_catalog, label="candidate-strategy catalog")
    train = read_jsonl(args.train_observations)
    development = read_jsonl(args.development_observations)
    train_manifest = _read_object(
        args.train_observation_manifest, label="train observation manifest"
    )
    development_manifest = _read_object(
        args.development_observation_manifest,
        label="development observation manifest",
    )
    registry = mine_pair_registry(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategies,
        train_observations=train,
        development_observations=development,
        train_observation_manifest=train_manifest,
        development_observation_manifest=development_manifest,
    )
    atomic_write_json(args.pair_registry_output, registry)
    return {
        "pair_registry_output": str(args.pair_registry_output),
        "pair_registry_hash": registry["pair_registry_hash"],
        "suite_count": len(registry["pairs"]),
        "test_instantiation_count": len(registry["test_instantiations"]),
        "test_outcomes_inspected": False,
    }


def _profile_checkpoint_environment_name(profile_id: str) -> str:
    suffix = "".join(
        character.upper() if character.isalnum() else "_"
        for character in profile_id
    )
    return f"AGENTDOJO_MONITOR_CHECKPOINT_{suffix}"


def _profile_checkpoint_environment(profile_id: str) -> str | None:
    variable = _profile_checkpoint_environment_name(profile_id)
    return os.environ.get(
        variable,
        os.environ.get("AGENTDOJO_MONITOR_CHECKPOINT"),
    )


def _preflight_pair_observation_environment(
    *,
    strategy_catalog: Mapping[str, Any],
    dependency_lock_path: Path | str,
) -> dict[str, Any]:
    """Bind observation-generation monitor profiles to the active runtime."""

    profiles = strategy_catalog.get("monitor_profiles")
    if not isinstance(profiles, list):
        raise AgentDojoRunnerError("candidate-strategy catalog lacks monitor profiles")
    fingerprints: set[str] = set()
    cache_variables = {
        "AGENTDOJO_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    }
    configured_checkpoint_variables = {
        "AGENTDOJO_MONITOR_CHECKPOINT"
    } if "AGENTDOJO_MONITOR_CHECKPOINT" in os.environ else set()
    required_checkpoint_variables: set[str] = set()
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            raise AgentDojoRunnerError(f"monitor profile {index} is not an object")
        if profile.get("implementation") in {
            "local_transformers",
            "transformers_pi_detector",
        }:
            fingerprint = profile.get("runtime_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
                raise AgentDojoRunnerError(
                    f"learned monitor profile {index} lacks a frozen runtime fingerprint"
                )
            fingerprints.add(fingerprint)
            profile_variable = _profile_checkpoint_environment_name(
                str(profile.get("profile_id", ""))
            )
            if profile_variable in os.environ:
                checkpoint_variable = profile_variable
                configured_checkpoint_variables.add(profile_variable)
            else:
                checkpoint_variable = "AGENTDOJO_MONITOR_CHECKPOINT"
            required_checkpoint_variables.add(checkpoint_variable)
    validate_persistent_runtime_paths(
        path_variables=tuple(
            sorted(cache_variables | configured_checkpoint_variables)
        ),
        required_directory_variables=tuple(
            sorted(required_checkpoint_variables)
        ),
    )
    return validate_environment_integrity(
        dependency_lock_path=dependency_lock_path,
        fixture_mode=False,
        runtime_fingerprints=fingerprints,
    )


def _generate_pair_observations(args: argparse.Namespace) -> dict[str, Any]:
    catalog = _read_object(args.catalog, label="catalog")
    splits = _read_object(args.splits, label="split manifest")
    strategies = _read_object(
        args.strategy_catalog, label="candidate-strategy catalog"
    )
    # Compatibility and frozen artifact checks happen before checkpoint
    # construction.  The generator repeats them while materializing rows.
    from . import compat
    from .catalog import validate_catalog
    from .pair_mining import validate_candidate_strategy_catalog
    from .splits import validate_split_manifest

    validate_catalog(catalog)
    validate_split_manifest(splits, catalog=catalog)
    validate_candidate_strategy_catalog(strategies)
    # This is deliberately before compatibility imports that materialize
    # suites and, critically, before any monitor checkpoint construction.
    learned_runtime = _preflight_pair_observation_environment(
        strategy_catalog=strategies,
        dependency_lock_path=args.dependency_lock,
    )
    compat.assert_compatible(
        str(catalog["agentdojo_source_revision"]),
        str(catalog["agentdojo_benchmark_version"]),
    )
    monitor_clients: dict[str, Any] = {}
    for profile in strategies["monitor_profiles"]:
        if profile.get("family") == "deterministic_task_policy":
            continue
        decoding = profile["decoding"]
        identity = ModelIdentity(
            role="monitor",
            implementation=str(profile["implementation"]),
            model_id=str(profile["model_id"]),
            model_revision=str(profile["model_revision"]),
            tokenizer_revision=str(profile["tokenizer_revision"]),
            checkpoint_fingerprint=str(profile["checkpoint_fingerprint"]),
            runtime_fingerprint=str(profile["runtime_fingerprint"]),
            prompt_hash=str(profile["prompt_hash"]),
            policy_hash=str(profile["policy_hash"]),
            threshold=float(profile["threshold"]),
            reasoning_mode=str(profile["reasoning_mode"]),
            dtype=str(profile["dtype"]),
            temperature=float(decoding["temperature"]),
            top_p=float(decoding["top_p"]),
            max_new_tokens=int(decoding["max_new_tokens"]),
        )
        monitor_clients[str(profile["profile_id"])] = model_client_from_identity(
            identity,
            checkpoint_path=_profile_checkpoint_environment(
                str(profile["profile_id"])
            ),
            cache_dir=os.environ.get("AGENTDOJO_MODEL_CACHE"),
            device=os.environ.get("MONITOR_DEVICE", "cuda"),
        )
    provenance = collect_provenance()
    rows, manifest = generate_pair_observation_set(
        catalog=catalog,
        split_manifest=splits,
        strategy_catalog=strategies,
        dataset_split=args.dataset_split,
        generator_source_tree_hash=str(provenance["source_tree_hash"]),
        learned_runtime=learned_runtime,
        monitor_clients=monitor_clients,
        compat=compat,
    )
    atomic_write_jsonl(args.observations_output, rows)
    atomic_write_json(args.observation_manifest_output, manifest)
    return {
        "dataset_split": args.dataset_split,
        "observations_output": str(args.observations_output),
        "observation_manifest_output": str(args.observation_manifest_output),
        "observation_count": len(rows),
        "observation_set_hash": manifest["observation_set_hash"],
        "test_outcomes_inspected": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silenttwin.agentdojo.runner",
        description="Execute hash-bound AgentDojo Tier-2 grid tasks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-grid-task")
    run.add_argument("--grid-manifest", type=Path, required=True)
    run.add_argument("--task-id", type=int, required=True)
    mine = commands.add_parser("mine-pairs")
    mine.add_argument("--catalog", type=Path, required=True)
    mine.add_argument("--splits", type=Path, required=True)
    mine.add_argument("--strategy-catalog", type=Path, required=True)
    mine.add_argument("--train-observations", type=Path, required=True)
    mine.add_argument("--development-observations", type=Path, required=True)
    mine.add_argument("--train-observation-manifest", type=Path, required=True)
    mine.add_argument("--development-observation-manifest", type=Path, required=True)
    mine.add_argument("--pair-registry-output", type=Path, required=True)
    observe = commands.add_parser("generate-pair-observations")
    observe.add_argument("--catalog", type=Path, required=True)
    observe.add_argument("--splits", type=Path, required=True)
    observe.add_argument("--strategy-catalog", type=Path, required=True)
    observe.add_argument(
        "--dependency-lock",
        type=Path,
        default=Path(
            os.environ.get(
                "AGENTDOJO_DEPENDENCY_LOCK", "requirements-tier2-agentdojo.lock"
            )
        ),
        help="exact AgentDojo core lock used for the runtime fingerprint",
    )
    observe.add_argument(
        "--dataset-split", choices=("train", "development"), required=True
    )
    observe.add_argument("--observations-output", type=Path, required=True)
    observe.add_argument("--observation-manifest-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run-grid-task":
            result: Any = run_grid_task(
                grid_manifest=args.grid_manifest, task_id=args.task_id
            )
            summary = {
                "task_id": args.task_id,
                "completed_shards": len(result),
                "configuration_hashes": [row["configuration_hash"] for row in result],
            }
        elif args.command == "mine-pairs":
            summary = _mine_pairs(args)
        else:
            summary = _generate_pair_observations(args)
    except (AgentDojoRunnerError, ResultValidationError, ValueError) as exc:
        parser.exit(2, f"AgentDojo runner error: {type(exc).__name__}: {exc}\n")
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AgentDojoRunnerError",
    "DeterministicSmokeModelClient",
    "advanced_trial_id",
    "controlled_trial_id",
    "ecological_trial_id",
    "main",
    "run_grid_task",
]
