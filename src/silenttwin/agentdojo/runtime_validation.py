"""Model-free validation of frozen artifacts selected by a run-stage grid.

This module deliberately imports neither AgentDojo nor model runtimes.  It is
the common fail-closed boundary that executes before compatibility discovery
or local checkpoint construction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import validate_catalog
from .config import AGENTDOJO_SUITES, stable_hash
from .grid import FrozenInputs, load_frozen_inputs
from .pair_mining import (
    monitor_pair_binding,
    validate_candidate_strategy_catalog,
    validate_pair_registry,
)
from .splits import validate_split_manifest
from .runtime_integrity import (
    RuntimeIntegrityError,
    derive_learned_runtime_fingerprint,
    make_learned_runtime_provenance,
    not_applicable_learned_runtime_provenance,
    validate_locked_distributions,
    verify_agentdojo_distribution,
)


class RuntimeArtifactError(ValueError):
    """Run-stage files or grid bindings are inconsistent."""


def validate_environment_integrity(
    *,
    dependency_lock_path: Path | str,
    fixture_mode: bool,
    runtime_fingerprints: set[str],
) -> dict[str, Any]:
    """Validate package bytes/versions and bind learned identities to reality."""

    if fixture_mode:
        return not_applicable_learned_runtime_provenance()
    wheel_artifact = os.environ.get("AGENTDOJO_WHEEL_ARTIFACT")
    try:
        verify_agentdojo_distribution(wheel_artifact=wheel_artifact)
        validate_locked_distributions(dependency_lock_path)
    except RuntimeIntegrityError as exc:
        raise RuntimeArtifactError(
            f"AgentDojo package/runtime integrity preflight failed: {exc}"
        ) from exc
    if not runtime_fingerprints:
        return not_applicable_learned_runtime_provenance()
    if len(runtime_fingerprints) != 1:
        raise RuntimeArtifactError(
            "one worker requires exactly one frozen learned-runtime fingerprint"
        )
    expected_runtime = next(iter(runtime_fingerprints))
    try:
        learned_report = derive_learned_runtime_fingerprint(
            dependency_lock_path, require_learned_stack=True
        )
        derived_runtime = learned_report.fingerprint
    except RuntimeIntegrityError as exc:
        raise RuntimeArtifactError(
            f"cannot derive the active learned-runtime fingerprint: {exc}"
        ) from exc
    if derived_runtime != expected_runtime:
        raise RuntimeArtifactError(
            "active learned-runtime fingerprint does not match the scientific "
            "model identities"
        )
    # Retain the explicit scheduler binding as a second assertion, never as
    # the source of truth: arbitrary environment text cannot replace the
    # fingerprint derived above from the active Python/package manifest.
    if os.environ.get("AGENTDOJO_RUNTIME_FINGERPRINT") != derived_runtime:
        raise RuntimeArtifactError(
            "AGENTDOJO_RUNTIME_FINGERPRINT does not match the derived active runtime"
        )
    try:
        return make_learned_runtime_provenance(learned_report)
    except RuntimeIntegrityError as exc:
        raise RuntimeArtifactError(
            f"cannot retain the active learned-runtime manifest: {exc}"
        ) from exc


def validate_persistent_runtime_paths(
    *,
    path_variables: Sequence[str],
    required_directory_variables: Sequence[str] = (),
) -> None:
    """Reject authoritative runtime paths in scheduler scratch.

    ``path_variables`` may include optional cache paths.  Variables listed in
    ``required_directory_variables`` additionally have to resolve to existing
    local directories.  The check is deliberately model-free so callers can
    run it before constructing a Transformers client.
    """

    scratch_variables = ["SLURM_TMPDIR"]
    if os.environ.get("PBS_JOBID"):
        # PBS_JOBDIR is the staging/execution directory; TMPDIR is PBS-assigned
        # job scratch when present.  Both are non-authoritative runtime paths.
        scratch_variables.extend(("PBS_JOBDIR", "TMPDIR"))
    scratch_roots = tuple(
        (variable, Path(value).expanduser().resolve())
        for variable in scratch_variables
        if (value := os.environ.get(variable))
    )
    required = set(required_directory_variables)
    for variable in sorted(set(path_variables) | required):
        value = os.environ.get(variable)
        path = Path(value).expanduser() if value else None
        if variable in required and (path is None or not path.is_dir()):
            raise RuntimeArtifactError(
                f"{variable} must identify a persistent local checkpoint directory"
            )
        for scratch_variable, scratch in scratch_roots:
            if path is None:
                break
            try:
                path.resolve().relative_to(scratch)
            except ValueError:
                pass
            else:
                raise RuntimeArtifactError(
                    f"{variable} cannot point inside ephemeral scheduler scratch "
                    f"{scratch_variable}"
                )


def _validate_checkpoint_paths(required_checkpoint_roles: set[str]) -> None:
    """Require persistent role checkpoints and reject scheduler scratch."""

    variables = tuple(
        f"AGENTDOJO_{role.upper()}_CHECKPOINT"
        for role in sorted(required_checkpoint_roles)
    )
    validate_persistent_runtime_paths(
        path_variables=variables,
        required_directory_variables=variables,
    )


def _configuration(member: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    value = member.get("configuration")
    if not isinstance(value, Mapping):
        raise RuntimeArtifactError(f"selected grid member {index} lacks configuration")
    return value


def _validate_monitor_client_binding(
    *,
    config: Mapping[str, Any],
    strategy_catalog: Mapping[str, Any],
    member_index: int,
) -> None:
    """Bind both private theta adapters to the client the runner will load.

    Prompt text, policy text, and threshold are the deliberately varying
    private theta fields.  The transport, checkpoint, tokenizer, learned
    runtime, reasoning mode, dtype, and decoding parameters must be identical
    because the runner intentionally shares one inference client.
    """

    family = str(config.get("monitor_family", ""))
    if family in {"deterministic_task_policy", "transformers_pi_detector"}:
        return
    raw_models = config.get("models")
    assert isinstance(raw_models, (list, tuple))
    monitor_models = [
        model
        for model in raw_models
        if isinstance(model, Mapping) and model.get("role") == "monitor"
    ]
    if len(monitor_models) != 1:
        raise RuntimeArtifactError(
            f"selected grid member {member_index} requires exactly one monitor client"
        )
    identity = monitor_models[0]
    raw_profiles = strategy_catalog.get("monitor_profiles")
    if not isinstance(raw_profiles, list):  # guarded by the catalog validator
        raise RuntimeArtifactError("candidate-strategy catalog lacks monitor profiles")
    profiles = {
        str(profile.get("profile_id")): profile
        for profile in raw_profiles
        if isinstance(profile, Mapping)
    }
    transport_fields = (
        "implementation",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "checkpoint_fingerprint",
        "runtime_fingerprint",
        "reasoning_mode",
        "dtype",
    )
    expected_decoding = {
        "temperature": identity.get("temperature"),
        "top_p": identity.get("top_p"),
        "max_new_tokens": identity.get("max_new_tokens"),
    }
    selected_profiles: list[Mapping[str, Any]] = []
    for theta_field in ("profile_theta0", "profile_theta1"):
        profile_id = str(config.get(theta_field, ""))
        profile = profiles.get(profile_id)
        if profile is None:
            raise RuntimeArtifactError(
                f"selected grid member {member_index} references unknown {theta_field}"
            )
        selected_profiles.append(profile)
        if profile.get("family") != family:
            raise RuntimeArtifactError(
                f"selected grid member {member_index} monitor family differs from {theta_field}"
            )
        mismatched = [
            field for field in transport_fields if profile.get(field) != identity.get(field)
        ]
        if mismatched or profile.get("decoding") != expected_decoding:
            detail = mismatched or ["decoding"]
            raise RuntimeArtifactError(
                f"selected grid member {member_index} monitor client differs from "
                f"{theta_field} on {detail}"
            )
    expected_prompt_binding = stable_hash(
        {
            "theta0": selected_profiles[0].get("prompt_hash"),
            "theta1": selected_profiles[1].get("prompt_hash"),
        }
    )
    expected_policy_binding = stable_hash(
        {
            "theta0": selected_profiles[0].get("policy_hash"),
            "theta1": selected_profiles[1].get("policy_hash"),
        }
    )
    if (
        identity.get("prompt_hash") != expected_prompt_binding
        or identity.get("policy_hash") != expected_policy_binding
        or identity.get("threshold") is not None
    ):
        raise RuntimeArtifactError(
            f"selected grid member {member_index} monitor adapter binding is invalid"
        )


def validate_runtime_artifacts(
    *,
    catalog_path: Path | str,
    splits_path: Path | str,
    strategy_catalog_path: Path | str,
    pair_registry_path: Path | str,
    analysis_plan_path: Path | str,
    dependency_lock_path: Path | str,
    grid_metadata: Mapping[str, Any],
    selected_members: Sequence[Mapping[str, Any]],
) -> FrozenInputs:
    """Validate the complete upstream chain and its selected grid bindings."""

    try:
        inputs = load_frozen_inputs(
            catalog_path=catalog_path,
            splits_path=splits_path,
            strategy_catalog_path=strategy_catalog_path,
            pair_registry_path=pair_registry_path,
            analysis_plan_path=analysis_plan_path,
            dependency_lock_path=dependency_lock_path,
        )
        # ``load_frozen_inputs`` validates the model-free grid contract.  The
        # domain validators add the full census, split, profile, and held-out
        # non-contamination invariants needed at execution time.
        validate_catalog(inputs.catalog)
        validate_split_manifest(inputs.splits, catalog=inputs.catalog)
        validate_candidate_strategy_catalog(inputs.strategy_catalog)
        validate_pair_registry(
            inputs.pair_registry,
            catalog=inputs.catalog,
            split_manifest=inputs.splits,
            strategy_catalog=inputs.strategy_catalog,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeArtifactError(f"invalid frozen AgentDojo artifact chain: {exc}") from exc

    if not selected_members:
        raise RuntimeArtifactError("run-stage task selected no grid members")
    if grid_metadata.get("upstream_binding_hash") != inputs.upstream.binding_hash:
        raise RuntimeArtifactError("grid metadata is bound to another upstream chain")

    expected_bindings = {
        "agentdojo_catalog_hash": inputs.upstream.catalog_hash,
        "scenario_registry_revision": inputs.upstream.scenario_registry_revision,
        "scenario_registry_hash": inputs.upstream.scenario_registry_hash,
        "split_manifest_hash": inputs.upstream.split_manifest_hash,
        "candidate_strategy_catalog_hash": (
            inputs.upstream.candidate_strategy_catalog_hash
        ),
        "pair_registry_hash": inputs.upstream.pair_registry_hash,
        "analysis_plan_hash": inputs.upstream.analysis_plan_hash,
        "dependency_lock_hash": inputs.upstream.dependency_lock_hash,
        "agentdojo_package_version": inputs.upstream.package_version,
        "agentdojo_source_revision": inputs.upstream.source_revision,
        "agentdojo_benchmark_version": inputs.upstream.benchmark_version,
    }
    scenario_index = {
        str(row["scenario_id"]): row for row in inputs.scenarios
    }
    fake_modes: set[bool] = set()
    runtime_fingerprints: set[str] = set()
    required_checkpoint_roles: set[str] = set()
    for index, member in enumerate(selected_members):
        config = _configuration(member, index)
        for field, expected in expected_bindings.items():
            if config.get(field) != expected:
                raise RuntimeArtifactError(
                    f"selected grid member {index} has a mismatched {field}"
                )
        suite = str(config.get("agentdojo_suite", ""))
        split = str(config.get("dataset_split", ""))
        if suite not in AGENTDOJO_SUITES:
            raise RuntimeArtifactError(f"selected grid member {index} has an unknown suite")
        raw_ids = config.get("scenario_ids")
        raw_groups = config.get("structural_group_ids")
        if not isinstance(raw_ids, (list, tuple)) or not isinstance(
            raw_groups, (list, tuple)
        ):
            raise RuntimeArtifactError(
                f"selected grid member {index} lacks its scenario bundle"
            )
        ids = tuple(str(item) for item in raw_ids)
        groups = tuple(str(item) for item in raw_groups)
        try:
            rows = tuple(scenario_index[item] for item in ids)
        except KeyError as exc:
            raise RuntimeArtifactError(
                f"selected grid member {index} references an unknown scenario"
            ) from exc
        if any(row.get("suite") != suite or row.get("dataset_split") != split for row in rows):
            raise RuntimeArtifactError(
                f"selected grid member {index} crosses suite/split boundaries"
            )
        observed_groups = {str(row["structural_group_id"]) for row in rows}
        if observed_groups != set(groups):
            raise RuntimeArtifactError(
                f"selected grid member {index} has a mismatched structural bundle"
            )
        expected_bundle = stable_hash(
            {
                "suite": suite,
                "dataset_split": split,
                "scenario_ids": list(ids),
                "structural_group_ids": list(groups),
            }
        )
        if config.get("scenario_bundle_hash") != expected_bundle:
            raise RuntimeArtifactError(
                f"selected grid member {index} has an invalid scenario bundle hash"
            )
        monitor_binding = monitor_pair_binding(
            inputs.strategy_catalog, inputs.pair_registry, suite=suite
        )
        if config.get("tier2_track") == "ecological":
            monitor_binding.pop("monitor_family")
        if any(config.get(name) != value for name, value in monitor_binding.items()):
            raise RuntimeArtifactError(
                f"selected grid member {index} misstates its monitor-pair identity"
            )
        fake_modes.add(bool(config.get("fixture_mode")))
        models = config.get("models")
        if not isinstance(models, (list, tuple)):
            raise RuntimeArtifactError(
                f"selected grid member {index} lacks model identities"
            )
        for model in models:
            if not isinstance(model, Mapping):
                raise RuntimeArtifactError(
                    f"selected grid member {index} has an invalid model identity"
                )
            implementation = str(model.get("implementation", ""))
            if implementation in {"local_transformers", "transformers_pi_detector"}:
                runtime_fingerprints.add(str(model.get("runtime_fingerprint", "")))
                required_checkpoint_roles.add(str(model.get("role", "")))
        _validate_monitor_client_binding(
            config=config,
            strategy_catalog=inputs.strategy_catalog,
            member_index=index,
        )

    if len(fake_modes) != 1:
        raise RuntimeArtifactError("one run-stage task cannot mix fixture and evidence rows")
    fixture_mode = next(iter(fake_modes))
    artifact_fixture_mode = (
        inputs.strategy_catalog.get("artifact_class")
        == "deterministic_fake_smoke_fixture"
        and inputs.pair_registry.get("artifact_class")
        == "deterministic_fake_smoke_fixture"
    )
    if fixture_mode != artifact_fixture_mode:
        raise RuntimeArtifactError(
            "fixture_mode does not match the frozen strategy/pair evidence class"
        )
    observed_fake_flag = os.environ.get("AGENTDOJO_FAKE_MODEL", "0")
    if observed_fake_flag not in {"0", "1"}:
        raise RuntimeArtifactError("AGENTDOJO_FAKE_MODEL must be exactly 0 or 1")
    if fixture_mode != (observed_fake_flag == "1"):
        raise RuntimeArtifactError(
            "AGENTDOJO_FAKE_MODEL must explicitly match the frozen fixture_mode"
        )
    requires_gpu = os.environ.get(
        "AGENTDOJO_REQUIRES_GPU", "0" if fixture_mode else "1"
    )
    if requires_gpu not in {"0", "1"}:
        raise RuntimeArtifactError("AGENTDOJO_REQUIRES_GPU must be exactly 0 or 1")
    if fixture_mode and requires_gpu != "0":
        raise RuntimeArtifactError("deterministic fixture execution cannot claim a GPU requirement")
    if not fixture_mode and not runtime_fingerprints and requires_gpu != "0":
        raise RuntimeArtifactError(
            "a model-free nonfixture task must declare AGENTDOJO_REQUIRES_GPU=0"
        )
    # Evidence execution validates the installed core before any checkpoint is
    # constructed.  An optional wheel path strengthens the check by validating
    # the exact published archive and comparing its payload with site-packages;
    # without it, the independently frozen canonical wheel-payload manifest is
    # authoritative.  No resolver or network operation occurs here.
    validate_environment_integrity(
        dependency_lock_path=dependency_lock_path,
        fixture_mode=fixture_mode,
        runtime_fingerprints=runtime_fingerprints,
    )
    if runtime_fingerprints and not fixture_mode:
        _validate_checkpoint_paths(required_checkpoint_roles)
    return inputs


__all__ = [
    "RuntimeArtifactError",
    "validate_environment_integrity",
    "validate_persistent_runtime_paths",
    "validate_runtime_artifacts",
]
