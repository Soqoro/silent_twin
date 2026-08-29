"""Audited train/development monitor-observation generation.

The reducer in :mod:`pair_mining` never runs a model.  This module is the only
first-party path that materializes frozen strategies against live pinned
AgentDojo tasks and emits the hash-bound verdict rows accepted by that reducer.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from silenttwin.schemas import stable_digest

from .action_eligibility import (
    ActionEligibilityError,
    execute_and_grade_action_plan,
    pilot_scenario_ids,
    validate_action_eligibility_manifest,
    validate_distinct_required_action_plans,
)
from .assembly import _monitor, _trusted_plan
from .canonical import calls_hash, canonicalize_tool_schemas
from .catalog import validate_catalog
from .config import (
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_SOURCE_REVISION,
)
from .monitors import MonitorInput
from .pair_mining import (
    PairMiningError,
    make_monitor_observation,
    make_observation_set_manifest,
    validate_candidate_strategy_catalog,
    validate_estimation_strategy_coverage,
    validate_observation_set_manifest,
)
from .runtime_integrity import (
    RuntimeIntegrityError,
    validate_learned_runtime_provenance,
)
from .seeds import SeedSchedule
from .splits import validate_split_manifest
from .visibility import public_value


def generate_pair_observation_set(
    *,
    catalog: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    strategy_catalog: Mapping[str, Any],
    action_eligibility_manifest: Mapping[str, Any],
    dataset_split: str,
    generator_source_tree_hash: str,
    learned_runtime: Mapping[str, Any],
    monitor_clients: Mapping[str, Any] | None = None,
    compat: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every frozen profile/strategy cell on one non-test split."""

    if dataset_split not in {"train", "development"}:
        raise PairMiningError(
            "monitor-observation generation is restricted to train/development"
        )
    validate_catalog(catalog)
    validate_split_manifest(split_manifest, catalog=catalog)
    validate_candidate_strategy_catalog(strategy_catalog)
    try:
        eligibility_hash = validate_action_eligibility_manifest(
            action_eligibility_manifest,
            catalog=catalog,
            split_manifest=split_manifest,
        )
    except ActionEligibilityError as exc:
        raise PairMiningError(f"observation action eligibility is invalid: {exc}") from exc
    runtime_fingerprints = {
        str(profile["runtime_fingerprint"])
        for profile in strategy_catalog["monitor_profiles"]
        if profile.get("implementation") in {
            "local_transformers",
            "transformers_pi_detector",
        }
    }
    try:
        validate_learned_runtime_provenance(
            learned_runtime,
            expected_runtime_fingerprints=runtime_fingerprints,
        )
    except RuntimeIntegrityError as exc:
        raise PairMiningError(
            f"observation runtime provenance is invalid: {exc}"
        ) from exc
    if (
        strategy_catalog.get("catalog_hash") != catalog.get("catalog_hash")
        or strategy_catalog.get("split_manifest_hash")
        != split_manifest.get("split_manifest_hash")
    ):
        raise PairMiningError("observation generation artifacts do not share an upstream")
    strategies = tuple(strategy_catalog["strategies"])
    profiles = tuple(strategy_catalog["monitor_profiles"])
    if len(strategies) < 2:
        raise PairMiningError(
            "estimation-only observation generation requires at least two strategies"
        )
    validate_estimation_strategy_coverage(
        strategy_catalog, action_eligibility_manifest
    )
    report = compat.assert_compatible(
        AGENTDOJO_SOURCE_REVISION, AGENTDOJO_BENCHMARK_VERSION
    )
    clients = dict(monitor_clients or {})
    eligible_scenario_ids = pilot_scenario_ids(
        action_eligibility_manifest, dataset_split=dataset_split
    )
    if not eligible_scenario_ids:
        raise PairMiningError(
            f"action eligibility permits no {dataset_split} pilot scenarios"
        )
    eligible_scenario_id_set = set(eligible_scenario_ids)
    rows: list[dict[str, Any]] = []
    action_validations: list[dict[str, Any]] = []
    suites: dict[str, Any] = {}
    scenarios = sorted(
        (
            row
            for row in catalog["scenarios"]
            if row["dataset_split"] == dataset_split
            and row["scenario_id"] in eligible_scenario_id_set
        ),
        key=lambda row: (
            str(row["suite"]),
            str(row["structural_group_id"]),
            str(row["scenario_id"]),
        ),
    )
    for scenario in scenarios:
        suite_name = str(scenario["suite"])
        suite = suites.get(suite_name)
        if suite is None:
            suite = compat.load_suite(
                suite_name,
                deployment_source_revision=AGENTDOJO_SOURCE_REVISION,
                benchmark_version=AGENTDOJO_BENCHMARK_VERSION,
            )
            suites[suite_name] = suite
        compat.validate_frozen_scenario_row(suite, scenario)
        user_task = compat.get_user_task(suite, str(scenario["user_task_id"]))
        injection_task = compat.get_injection_task(
            suite, str(scenario["injection_task_id"])
        )
        user_prompt = str(user_task.PROMPT)
        injection_goal = str(injection_task.GOAL)
        schemas = canonicalize_tool_schemas(compat.canonical_tool_schemas(suite))
        schema_by_name = {schema.name: schema for schema in schemas}
        plans: dict[str, Any] = {}
        strategy_index: dict[str, Mapping[str, Any]] = {}
        for strategy in strategies:
            strategy_id = str(strategy["strategy_id"])
            strategy_index[strategy_id] = strategy
            plans[strategy_id] = _trusted_plan(
                plan_id=strategy_id,
                strategy=strategy,
                scenario=scenario,
                user_prompt=user_prompt,
                injection_goal=injection_goal,
            )
        try:
            required_action_hashes = validate_distinct_required_action_plans(
                plans, schemas=schemas
            )
        except ActionEligibilityError as exc:
            raise PairMiningError(
                f"scenario {scenario['scenario_id']} has invalid candidate actions: {exc}"
            ) from exc
        validation_by_strategy: dict[str, Mapping[str, Any]] = {}
        for strategy_id, plan in plans.items():
            try:
                validation = execute_and_grade_action_plan(
                    compat=compat,
                    suite=suite,
                    user_task=user_task,
                    injection_task=injection_task,
                    scenario=scenario,
                    strategy_id=strategy_id,
                    plan=plan,
                    action_eligibility_manifest_hash=eligibility_hash,
                    required_action_multiset_hash=required_action_hashes[
                        strategy_id
                    ],
                )
            except ActionEligibilityError as exc:
                raise PairMiningError(
                    f"scenario {scenario['scenario_id']} action validation failed: {exc}"
                ) from exc
            validation_by_strategy[strategy_id] = validation
            action_validations.append(validation)
        for strategy_id, plan in plans.items():
            strategy = strategy_index[strategy_id]
            action_validation = validation_by_strategy[strategy_id]
            for profile in profiles:
                profile_id = str(profile["profile_id"])
                monitor = _monitor(
                    profile,
                    scenario_id=str(scenario["scenario_id"]),
                    label_to_strategy={"candidate": strategy_id},
                    model_client=clients.get(profile_id),
                )
                schedule = SeedSchedule(
                    str(scenario["scenario_id"]),
                    str(profile.get("model_revision", profile["profile_hash"])),
                    0,
                )
                evaluations = []
                inputs: list[str] = []
                seeds: list[int] = []
                for call_index, call in enumerate(plan.calls):
                    try:
                        schema = schema_by_name[call.function]
                    except KeyError as exc:
                        raise PairMiningError(
                            f"strategy {strategy_id!r} references unknown tool {call.function!r}"
                        ) from exc
                    monitor_input = MonitorInput(
                        frozen_user_goal=user_prompt,
                        public_conversation_summary=public_value(
                            {
                                "scenario_id": scenario["scenario_id"],
                                "summary": "frozen controlled-plan evaluation",
                            }
                        ),
                        proposed_call=call,
                        tool_schema=schema,
                    )
                    seed = schedule.monitor(call_index)
                    evaluation = monitor.evaluate(
                        monitor_input, plan_id=strategy_id, seed=seed
                    )
                    if evaluation.decision not in {"allow", "block"}:
                        raise PairMiningError(
                            f"monitor {profile_id!r} returned a nonbinary decision"
                        )
                    evaluations.append(evaluation)
                    inputs.append(monitor_input.digest)
                    seeds.append(seed)
                decision = (
                    "block"
                    if any(item.decision == "block" for item in evaluations)
                    else "allow"
                )
                scores = [
                    float(item.score)
                    for item in evaluations
                    if item.score is not None
                ]
                score = max(scores) if scores else (1.0 if decision == "block" else 0.0)
                rows.append(
                    make_monitor_observation(
                        scenario=scenario,
                        strategy=strategy,
                        profile=profile,
                        catalog_hash=str(catalog["catalog_hash"]),
                        split_manifest_hash=str(
                            split_manifest["split_manifest_hash"]
                        ),
                        candidate_strategy_catalog_hash=str(
                            strategy_catalog["candidate_strategy_catalog_hash"]
                        ),
                        monitor_decision=decision,
                        monitor_score=score,
                        call_sequence_hash=calls_hash(plan.calls),
                        monitor_input_hashes=inputs,
                        seed_schedule=seeds,
                        call_provenance=[
                            item.to_trusted_dict() for item in evaluations
                        ],
                        generator_source_tree_hash=generator_source_tree_hash,
                        action_eligibility_manifest_hash=eligibility_hash,
                        action_validation_hash=str(
                            action_validation["action_validation_hash"]
                        ),
                    )
                )
    rows.sort(
        key=lambda row: (
            row["suite"],
            row["structural_group_id"],
            row["scenario_id"],
            row["profile_id"],
            row["strategy_id"],
        )
    )
    manifest = make_observation_set_manifest(
        observations=rows,
        dataset_split=dataset_split,
        catalog_hash=str(catalog["catalog_hash"]),
        split_manifest_hash=str(split_manifest["split_manifest_hash"]),
        candidate_strategy_catalog_hash=str(
            strategy_catalog["candidate_strategy_catalog_hash"]
        ),
        generator_source_tree_hash=generator_source_tree_hash,
        compatibility=report.to_dict(),
        scientific_evidence_eligible=all(
            profile.get("family") != "deterministic_task_policy"
            for profile in profiles
        ),
        learned_runtime=learned_runtime,
        action_eligibility_manifest_hash=eligibility_hash,
        eligible_scenario_ids=eligible_scenario_ids,
        action_validations=action_validations,
    )
    # A digest over the ordered row set is already in the manifest.  This
    # redundant local assertion catches accidental mutation before publication.
    if manifest["observations_hash"] != stable_digest(rows):
        raise AssertionError("observation rows changed during manifest construction")
    validate_observation_set_manifest(
        manifest,
        observations=rows,
        dataset_split=dataset_split,
        catalog_hash=str(catalog["catalog_hash"]),
        split_manifest_hash=str(split_manifest["split_manifest_hash"]),
        candidate_strategy_catalog_hash=str(
            strategy_catalog["candidate_strategy_catalog_hash"]
        ),
        expected_runtime_fingerprints=runtime_fingerprints,
        action_eligibility_manifest_hash=eligibility_hash,
        eligible_scenario_ids=eligible_scenario_ids,
        strategy_ids=tuple(str(row["strategy_id"]) for row in strategies),
    )
    return rows, manifest


__all__ = ["generate_pair_observation_set"]
