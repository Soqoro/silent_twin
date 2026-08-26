"""Resolve frozen AgentDojo artifacts into one controlled backend episode.

Only this assembly boundary joins public catalog metadata, trusted strategy
plans, monitor profiles, and the pinned compatibility facade.  Test-set
outcomes are never consulted here: a pair registry must already contain the
suite construction frozen by :mod:`silenttwin.agentdojo.pair_mining`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from silenttwin.backends.agentdojo import AgentDojoBackend
from silenttwin.backends.base import PublicPlan, PublicScenario, TrustedPlan

from .advanced import AuthoredEffect, validate_authored_effect_graph
from .canonical import canonicalize_tool_call, canonicalize_tool_schemas
from .config import AgentDojoExperimentConfig, ModelIdentity, stable_hash
from .monitors import (
    DeterministicPolicyMonitor,
    GptOssSafeguardMonitor,
    GraniteGuardianMonitor,
    LocalActionMonitor,
    MonitorPair,
    MonitorProfile,
)
from .pair_mining import validate_candidate_strategy_catalog, validate_pair_registry
from .visibility import public_value


class AssemblyError(ValueError):
    """Frozen artifacts cannot be assembled into an executable episode."""


_PLACEHOLDERS = {
    "{{user_prompt}}": "user_prompt",
    "{{injection_goal}}": "injection_goal",
    "{{injection_vector_id}}": "injection_vector_id",
    "{{user_task_id}}": "user_task_id",
    "{{injection_task_id}}": "injection_task_id",
    "{{scenario_id}}": "scenario_id",
}


def _substitute(value: Any, bindings: Mapping[str, str]) -> Any:
    """Substitute only explicit whole-value tokens in a frozen plan.

    Partial string formatting is intentionally unsupported: it is difficult to
    audit and can silently alter structured identifiers or policy text.
    """

    if isinstance(value, str) and value in _PLACEHOLDERS:
        return bindings[_PLACEHOLDERS[value]]
    if isinstance(value, Mapping):
        return {str(key): _substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    return value


def _by_id(rows: Sequence[Mapping[str, Any]], field: str, identifier: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get(field)) == identifier]
    if len(matches) != 1:
        raise AssemblyError(f"expected one {field}={identifier!r}, found {len(matches)}")
    return matches[0]


def _scenario_plan(
    strategy: Mapping[str, Any], scenario: Mapping[str, Any]
) -> Mapping[str, Any]:
    scenario_plans = strategy.get("scenario_plans")
    if isinstance(scenario_plans, Mapping) and scenario["scenario_id"] in scenario_plans:
        plan = scenario_plans[scenario["scenario_id"]]
    else:
        suite_plans = strategy.get("suite_plan_templates")
        if isinstance(suite_plans, Mapping) and scenario["suite"] in suite_plans:
            plan = suite_plans[scenario["suite"]]
        else:
            plan = strategy.get("plan_template")
    if not isinstance(plan, Mapping):
        raise AssemblyError(
            f"strategy {strategy.get('strategy_id')!r} has no plan for scenario "
            f"{scenario['scenario_id']!r}"
        )
    return plan


def _trusted_plan(
    *,
    plan_id: str,
    strategy: Mapping[str, Any],
    scenario: Mapping[str, Any],
    user_prompt: str,
    injection_goal: str,
) -> TrustedPlan:
    raw_plan = _scenario_plan(strategy, scenario)
    bindings = {
        "user_prompt": user_prompt,
        "injection_goal": injection_goal,
        "injection_vector_id": str(scenario["injection_vector_id"]),
        "user_task_id": str(scenario["user_task_id"]),
        "injection_task_id": str(scenario["injection_task_id"]),
        "scenario_id": str(scenario["scenario_id"]),
    }
    materialized = _substitute(raw_plan, bindings)
    calls_raw = materialized.get("calls")
    if not isinstance(calls_raw, list) or not calls_raw:
        raise AssemblyError(f"controlled plan {plan_id!r} has no calls")
    calls = tuple(
        canonicalize_tool_call(call, default_id=f"{plan_id}:{index}")
        for index, call in enumerate(calls_raw)
    )
    dependencies = materialized.get("dependencies", {})
    if dependencies is None:
        dependencies = {}
    if not isinstance(dependencies, Mapping):
        raise AssemblyError(f"controlled plan {plan_id!r} dependencies must be an object")
    return TrustedPlan(
        plan_id=plan_id,
        calls=calls,
        data_dependencies=tuple(str(item) for item in dependencies.get("data", ())),
        control_dependencies=tuple(str(item) for item in dependencies.get("control", ())),
        atomic_group=(
            str(materialized["atomic_group"])
            if materialized.get("atomic_group") is not None
            else None
        ),
        ground_truth_label=str(strategy.get("ground_truth_label", "prohibited")),
        registry_revision=str(
            strategy.get("strategy_revision", "silenttwin-agentdojo-strategy-v1")
        ),
    )


def _profile_decisions(
    profile: Mapping[str, Any],
    *,
    scenario_id: str,
    label_to_strategy: Mapping[str, str],
) -> dict[str, str | bool]:
    registry = profile.get("decision_registry", profile.get("decisions", {}))
    if not isinstance(registry, Mapping):
        raise AssemblyError(f"deterministic profile {profile.get('profile_id')!r} has no registry")
    scenario_registry = registry.get(scenario_id, registry)
    if not isinstance(scenario_registry, Mapping):
        raise AssemblyError("deterministic scenario decision registry must be an object")
    decisions: dict[str, str | bool] = {}
    for label, strategy_id in label_to_strategy.items():
        value = scenario_registry.get(strategy_id, scenario_registry.get(label))
        if value is not None:
            decisions[label] = value
            decisions[f"probe_{label}"] = value
    # Exact call-digest and function keys may be frozen alongside strategy IDs.
    for key, value in scenario_registry.items():
        if str(key) not in label_to_strategy.values():
            decisions[str(key)] = value
    return decisions


def _monitor(
    profile: Mapping[str, Any],
    *,
    scenario_id: str,
    label_to_strategy: Mapping[str, str],
    model_client: Any | None,
) -> Any:
    family = str(profile.get("family", ""))
    profile_id = str(profile.get("profile_id", ""))
    if family == "deterministic_task_policy":
        return DeterministicPolicyMonitor(
            profile_id=profile_id,
            decisions=_profile_decisions(
                profile,
                scenario_id=scenario_id,
                label_to_strategy=label_to_strategy,
            ),
            default_decision=str(profile.get("default_decision", "block")),
            policy_epoch=int(profile.get("policy_epoch", 0)),
        )
    required = (
        "implementation",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "checkpoint_fingerprint",
        "prompt_hash",
        "policy_hash",
        "prompt_template",
        "policy_text",
    )
    missing = [field for field in required if not isinstance(profile.get(field), str)]
    if missing:
        raise AssemblyError(f"learned monitor profile {profile_id!r} lacks {missing}")
    monitor_profile = MonitorProfile(
        profile_id=profile_id,
        family=family,
        implementation=str(profile["implementation"]),
        model_id=str(profile["model_id"]),
        model_revision=str(profile["model_revision"]),
        tokenizer_revision=str(profile["tokenizer_revision"]),
        checkpoint_fingerprint=str(profile["checkpoint_fingerprint"]),
        runtime_fingerprint=str(profile["runtime_fingerprint"]),
        dtype=str(profile["dtype"]),
        frozen_profile_hash=str(profile["profile_hash"]),
        prompt_hash=str(profile["prompt_hash"]),
        policy_hash=str(profile["policy_hash"]),
        threshold=float(profile.get("threshold", 0.5)),
        reasoning_mode=str(profile.get("reasoning_mode", "direct")),
        decoding=dict(profile.get("decoding", {})),
        prompt_template=str(profile["prompt_template"]),
        policy_text=str(profile["policy_text"]),
    )
    if family == "granite_guardian_4_1_8b":
        return GraniteGuardianMonitor(monitor_profile, model_client)
    if family == "gpt_oss_safeguard_20b":
        return GptOssSafeguardMonitor(monitor_profile, model_client)
    if family == "local_action_monitor":
        return LocalActionMonitor(monitor_profile, model_client)
    raise AssemblyError(
        f"monitor profile {profile_id!r} is not an action-level controlled monitor"
    )


def model_client_from_identity(
    identity: ModelIdentity,
    *,
    checkpoint_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    device: str | None = None,
) -> Any:
    """Instantiate an explicitly configured local model, with no fallback."""

    if identity.implementation != "local_transformers":
        raise AssemblyError(
            f"model implementation {identity.implementation!r} is not a real local runtime"
        )
    # Importing this module is cheap; torch/transformers and the checkpoint are
    # loaded only by ensure_available/complete after scheduler preflight.
    from silenttwin.model_clients.local_transformers import (
        LocalModelConfig,
        LocalTransformersModelClient,
    )

    if checkpoint_path is None:
        raise AssemblyError(
            f"{identity.role} requires an explicit operational local checkpoint path"
        )
    resolved_checkpoint = Path(checkpoint_path).expanduser()
    if not resolved_checkpoint.is_dir():
        raise AssemblyError(
            f"{identity.role} checkpoint is not a local directory: {resolved_checkpoint}"
        )
    if not identity.checkpoint_fingerprint.startswith("sha256:"):
        raise AssemblyError(
            f"{identity.role} requires a full-tree sha256 checkpoint fingerprint"
        )
    configured_cache = cache_dir if cache_dir is not None else identity.cache_dir
    configured_device = device if device is not None else identity.device
    client = LocalTransformersModelClient(
        LocalModelConfig(
            model_id=str(resolved_checkpoint),
            model_revision=identity.model_revision,
            tokenizer_revision=identity.tokenizer_revision,
            checkpoint_fingerprint=identity.checkpoint_fingerprint,
            semantic_model_id=identity.model_id,
            model_cache_dir=Path(configured_cache) if configured_cache else None,
            dtype=identity.dtype,
            max_new_tokens=identity.max_new_tokens,
            temperature=identity.temperature,
            top_p=identity.top_p,
            decoding_seed=0,
            batch_size=1,
            device=configured_device or "cuda",
        )
    )
    client.ensure_available()
    return client


def assemble_controlled_backend(
    *,
    config: AgentDojoExperimentConfig,
    scenario: Mapping[str, Any],
    strategy_catalog: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
    monitor_clients: Mapping[str, Any] | None = None,
    compat: Any | None = None,
) -> AgentDojoBackend:
    """Build a fresh backend for exactly one frozen scenario row."""

    validate_candidate_strategy_catalog(strategy_catalog)
    validate_pair_registry(pair_registry, strategy_catalog=strategy_catalog)
    if scenario.get("scenario_id") not in config.scenario_ids:
        raise AssemblyError("scenario is outside the selected grid bundle")
    if scenario.get("suite") != config.agentdojo_suite or scenario.get(
        "dataset_split"
    ) != config.dataset_split:
        raise AssemblyError("scenario suite/split differs from selected configuration")
    if strategy_catalog.get("candidate_strategy_catalog_hash") != config.candidate_strategy_catalog_hash:
        raise AssemblyError("strategy catalog hash differs from the grid")
    if pair_registry.get("pair_registry_hash") != config.pair_registry_hash:
        raise AssemblyError("pair registry hash differs from the grid")
    adapter = compat
    if adapter is None:
        from . import compat as adapter

    suite = adapter.load_suite(
        config.agentdojo_suite,
        deployment_source_revision=config.agentdojo_source_revision,
        benchmark_version=config.agentdojo_benchmark_version,
    )
    user_task = adapter.get_user_task(suite, str(scenario["user_task_id"]))
    injection_task = adapter.get_injection_task(
        suite, str(scenario["injection_task_id"])
    )
    injection_goal = str(getattr(injection_task, "GOAL", ""))
    user_prompt = str(getattr(user_task, "PROMPT", ""))
    pair = _by_id(
        tuple(pair_registry["pairs"]), "suite", config.agentdojo_suite
    )
    strategies = tuple(strategy_catalog["strategies"])
    strategy_0 = _by_id(
        strategies, "strategy_id", str(pair["candidate_0_strategy_id"])
    )
    strategy_1 = _by_id(
        strategies, "strategy_id", str(pair["candidate_1_strategy_id"])
    )
    label_to_strategy = {
        "candidate_0": str(strategy_0["strategy_id"]),
        "candidate_1": str(strategy_1["strategy_id"]),
    }
    candidate_0 = _trusted_plan(
        plan_id="candidate_0",
        strategy=strategy_0,
        scenario=scenario,
        user_prompt=user_prompt,
        injection_goal=injection_goal,
    )
    candidate_1 = _trusted_plan(
        plan_id="candidate_1",
        strategy=strategy_1,
        scenario=scenario,
        user_prompt=user_prompt,
        injection_goal=injection_goal,
    )
    probe_0 = TrustedPlan(
        plan_id="probe_candidate_0",
        calls=candidate_0.calls,
        data_dependencies=candidate_0.data_dependencies,
        control_dependencies=candidate_0.control_dependencies,
        atomic_group=candidate_0.atomic_group,
        ground_truth_label=candidate_0.ground_truth_label,
        registry_revision=candidate_0.registry_revision,
    )
    probe_1 = TrustedPlan(
        plan_id="probe_candidate_1",
        calls=candidate_1.calls,
        data_dependencies=candidate_1.data_dependencies,
        control_dependencies=candidate_1.control_dependencies,
        atomic_group=candidate_1.atomic_group,
        ground_truth_label=candidate_1.ground_truth_label,
        registry_revision=candidate_1.registry_revision,
    )
    schemas = canonicalize_tool_schemas(adapter.canonical_tool_schemas(suite))
    public_scenario = PublicScenario(
        scenario_id=str(scenario["scenario_id"]),
        suite=config.agentdojo_suite,
        user_prompt=user_prompt,
        tool_schemas=schemas,
        candidate_plans=(
            PublicPlan(
                "candidate_0",
                "candidate_0",
                str(strategy_0.get("public_description", "candidate strategy 0")),
                "candidate",
                tuple(call.public_dict(protect_arguments=True) for call in candidate_0.calls),
            ),
            PublicPlan(
                "candidate_1",
                "candidate_1",
                str(strategy_1.get("public_description", "candidate strategy 1")),
                "candidate",
                tuple(call.public_dict(protect_arguments=True) for call in candidate_1.calls),
            ),
            PublicPlan(
                "probe_candidate_0",
                "candidate_0",
                str(strategy_0.get("public_description", "candidate strategy 0")),
                "probe",
                tuple(call.public_dict(protect_arguments=True) for call in probe_0.calls),
            ),
            PublicPlan(
                "probe_candidate_1",
                "candidate_1",
                str(strategy_1.get("public_description", "candidate strategy 1")),
                "probe",
                tuple(call.public_dict(protect_arguments=True) for call in probe_1.calls),
            ),
        ),
        structural_group_id=str(scenario["structural_group_id"]),
        dataset_split=config.dataset_split,
        public_environment=public_value(
            {
                "initial_environment_hash": scenario["initial_environment_hash"],
                "injection_vector_id": scenario["injection_vector_id"],
                "candidate_mapping_revision": stable_hash(label_to_strategy),
            }
        ),
    )
    profiles = tuple(strategy_catalog["monitor_profiles"])
    profile_0 = _by_id(profiles, "profile_id", str(pair["profile_theta0"]))
    profile_1 = _by_id(profiles, "profile_id", str(pair["profile_theta1"]))
    clients = dict(monitor_clients or {})
    monitor_pair = MonitorPair(
        theta0=_monitor(
            profile_0,
            scenario_id=str(scenario["scenario_id"]),
            label_to_strategy=label_to_strategy,
            model_client=clients.get(str(profile_0["profile_id"])),
        ),
        theta1=_monitor(
            profile_1,
            scenario_id=str(scenario["scenario_id"]),
            label_to_strategy=label_to_strategy,
            model_client=clients.get(str(profile_1["profile_id"])),
        ),
        pair_registry_hash=str(pair_registry["pair_registry_hash"]),
    )
    return AgentDojoBackend(
        suite=suite,
        user_task=user_task,
        injection_task=injection_task,
        injections={str(scenario["injection_vector_id"]): injection_goal},
        public_scenario=public_scenario,
        trusted_plans={
            plan.plan_id: plan for plan in (candidate_0, candidate_1, probe_0, probe_1)
        },
        monitor_pair=monitor_pair,
        package_version=config.agentdojo_package_version,
        source_revision=config.agentdojo_source_revision,
        benchmark_version=config.agentdojo_benchmark_version,
        catalog_hash=config.agentdojo_catalog_hash,
        expected_initial_environment_hash=str(scenario["initial_environment_hash"]),
        compat=adapter,
    )


def assemble_useful_work_backend(
    *,
    config: AgentDojoExperimentConfig,
    scenario: Mapping[str, Any],
    strategy_catalog: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
    monitor_clients: Mapping[str, Any] | None = None,
    workflow_override: str | None = None,
    compat: Any | None = None,
) -> tuple[AgentDojoBackend, tuple[AuthoredEffect, ...]]:
    """Assemble one frozen E4 mixed-branch workload.

    ``mixed_workflows`` is benchmark-authored and part of the hashed strategy
    catalog.  Model-emitted dependency or policy annotations are never read.
    """

    validate_candidate_strategy_catalog(strategy_catalog)
    validate_pair_registry(pair_registry, strategy_catalog=strategy_catalog)
    raw_workflows = strategy_catalog.get("mixed_workflows")
    if not isinstance(raw_workflows, list):
        raise AssemblyError(
            "candidate-strategy catalog lacks benchmark-authored mixed_workflows for E4"
        )
    requested_workflow = workflow_override or config.workflow
    if requested_workflow is None:
        raise AssemblyError("useful-work assembly requires a frozen workflow name")
    matches = [
        row
        for row in raw_workflows
        if isinstance(row, Mapping)
        and row.get("suite") == config.agentdojo_suite
        and row.get("workflow") == requested_workflow
        and (
            row.get("scenario_id") in {None, scenario.get("scenario_id")}
        )
    ]
    scenario_specific = [row for row in matches if row.get("scenario_id") == scenario.get("scenario_id")]
    if scenario_specific:
        matches = scenario_specific
    if len(matches) != 1:
        raise AssemblyError(
            f"expected one authored {config.agentdojo_suite}/{requested_workflow} workflow, found {len(matches)}"
        )
    workflow = matches[0]
    raw_effects = workflow.get("effects")
    if not isinstance(raw_effects, list) or not raw_effects:
        raise AssemblyError("authored E4 workflow has no effects")
    adapter = compat
    if adapter is None:
        from . import compat as adapter

    suite = adapter.load_suite(
        config.agentdojo_suite,
        deployment_source_revision=config.agentdojo_source_revision,
        benchmark_version=config.agentdojo_benchmark_version,
    )
    user_task = adapter.get_user_task(suite, str(scenario["user_task_id"]))
    injection_task = adapter.get_injection_task(suite, str(scenario["injection_task_id"]))
    injection_goal = str(getattr(injection_task, "GOAL", ""))
    user_prompt = str(getattr(user_task, "PROMPT", ""))
    plans: list[TrustedPlan] = []
    authored: list[AuthoredEffect] = []
    public_plans: list[PublicPlan] = []
    for index, effect in enumerate(raw_effects):
        if not isinstance(effect, Mapping):
            raise AssemblyError("authored E4 effects must be objects")
        effect_id = str(effect.get("effect_id", ""))
        plan_id = str(effect.get("plan_id", effect_id))
        if not effect_id or not plan_id:
            raise AssemblyError("authored E4 effect lacks stable IDs")
        strategy_like = {
            "strategy_id": plan_id,
            "strategy_revision": str(workflow.get("workflow_revision", "silenttwin-agentdojo-e4-v1")),
            "ground_truth_label": effect.get("ground_truth_label"),
            "plan_template": {
                "calls": effect.get("calls", ()),
                "dependencies": {
                    "data": effect.get("data_dependencies", ()),
                    "control": effect.get("control_dependencies", ()),
                },
                "atomic_group": effect.get("atomic_group"),
            },
        }
        plan = _trusted_plan(
            plan_id=plan_id,
            strategy=strategy_like,
            scenario=scenario,
            user_prompt=user_prompt,
            injection_goal=injection_goal,
        )
        plans.append(plan)
        authored.append(
            AuthoredEffect(
                effect_id=effect_id,
                plan_id=plan_id,
                ground_truth_label=plan.ground_truth_label,
                data_dependencies=tuple(str(item) for item in effect.get("data_dependencies", ())),
                control_dependencies=tuple(str(item) for item in effect.get("control_dependencies", ())),
                atomic_group=(str(effect["atomic_group"]) if effect.get("atomic_group") is not None else None),
            )
        )
        public_plans.append(
            PublicPlan(
                plan_id,
                str(effect.get("public_label", f"workflow_effect_{index}")),
                str(effect.get("public_description", "benchmark-authored workflow effect")),
                "workflow",
                tuple(call.public_dict(protect_arguments=True) for call in plan.calls),
            )
        )
    validate_authored_effect_graph(authored)
    schemas = canonicalize_tool_schemas(adapter.canonical_tool_schemas(suite))
    public_scenario = PublicScenario(
        scenario_id=str(scenario["scenario_id"]),
        suite=config.agentdojo_suite,
        user_prompt=user_prompt,
        tool_schemas=schemas,
        candidate_plans=tuple(public_plans),
        structural_group_id=str(scenario["structural_group_id"]),
        dataset_split=config.dataset_split,
        public_environment=public_value(
            {
                "initial_environment_hash": scenario["initial_environment_hash"],
                "workflow_id": str(workflow.get("workflow_id", requested_workflow)),
                "authored_graph_hash": validate_authored_effect_graph(authored),
            }
        ),
    )
    pair = _by_id(tuple(pair_registry["pairs"]), "suite", config.agentdojo_suite)
    profiles = tuple(strategy_catalog["monitor_profiles"])
    profile_0 = dict(_by_id(profiles, "profile_id", str(pair["profile_theta0"])))
    profile_1 = dict(_by_id(profiles, "profile_id", str(pair["profile_theta1"])))
    decisions = workflow.get("monitor_decisions", {})
    if isinstance(decisions, Mapping):
        for profile in (profile_0, profile_1):
            frozen = decisions.get(profile["profile_id"])
            if isinstance(frozen, Mapping):
                existing = profile.get("decision_registry", {})
                profile["decision_registry"] = {
                    **(dict(existing) if isinstance(existing, Mapping) else {}),
                    str(scenario["scenario_id"]): dict(frozen),
                }
    label_to_strategy = {plan.plan_id: plan.plan_id for plan in plans}
    clients = dict(monitor_clients or {})
    monitor_pair = MonitorPair(
        theta0=_monitor(
            profile_0,
            scenario_id=str(scenario["scenario_id"]),
            label_to_strategy=label_to_strategy,
            model_client=clients.get(str(profile_0["profile_id"])),
        ),
        theta1=_monitor(
            profile_1,
            scenario_id=str(scenario["scenario_id"]),
            label_to_strategy=label_to_strategy,
            model_client=clients.get(str(profile_1["profile_id"])),
        ),
        pair_registry_hash=str(pair_registry["pair_registry_hash"]),
    )
    backend = AgentDojoBackend(
        suite=suite,
        user_task=user_task,
        injection_task=injection_task,
        injections={str(scenario["injection_vector_id"]): injection_goal},
        public_scenario=public_scenario,
        trusted_plans={plan.plan_id: plan for plan in plans},
        monitor_pair=monitor_pair,
        package_version=config.agentdojo_package_version,
        source_revision=config.agentdojo_source_revision,
        benchmark_version=config.agentdojo_benchmark_version,
        catalog_hash=config.agentdojo_catalog_hash,
        expected_initial_environment_hash=str(scenario["initial_environment_hash"]),
        compat=adapter,
    )
    return backend, tuple(authored)


__all__ = [
    "AssemblyError",
    "assemble_controlled_backend",
    "assemble_useful_work_backend",
    "model_client_from_identity",
]
