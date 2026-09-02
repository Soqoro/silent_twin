from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments/silenttwin"
ENTRYPOINTS = (
    "run_agentdojo_catalog.sh",
    "run_agentdojo_pair_mining_tier2.sh",
    "run_experiment_1_feedback_leakage_agentdojo_tier2.sh",
    "run_experiment_2_feedback_assisted_bypass_agentdojo_tier2.sh",
    "run_experiment_3_channel_closure_agentdojo_tier2.sh",
    "run_experiment_4_useful_work_agentdojo_tier2.sh",
    "run_experiment_5_assumption_ablations_agentdojo_tier2.sh",
    "run_agentdojo_ecological_tier2.sh",
    "run_agentdojo_recipient_separation_train_tier2.sh",
    "run_agentdojo_interface_realization_train_tier2.sh",
    "run_agentdojo_forced_choice_readout_train_tier2.sh",
    "run_agentdojo_clean_repair_train_tier2.sh",
    "run_agentdojo_native_tool_interface_train_tier2.sh",
    "run_agentdojo_checkpoint_conformance_tier2.sh",
)
SCHEDULER_ENVIRONMENT_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_TMPDIR",
    "PBS_JOBID",
    "PBS_ARRAY_ID",
    "PBS_ARRAY_INDEX",
    "PBS_ENVIRONMENT",
    "PBS_JOBDIR",
    "PBS_O_HOME",
    "TMPDIR",
)


def _clean_scheduler_environment() -> dict[str, str]:
    values = os.environ.copy()
    for name in SCHEDULER_ENVIRONMENT_VARIABLES:
        values.pop(name, None)
    return values


def _manifest(
    path: Path,
    total_tasks: int = 2,
    *,
    fixture_mode: bool = True,
    learned_model: bool = False,
) -> None:
    metadata = {
        "record_type": "grid_metadata",
        "schema_version": "silenttwin.agentdojo.grid.v1",
        "environment_backend": "agentdojo",
        "model_free": True,
        "total_tasks": total_tasks,
    }
    records = [metadata]
    for task_id in range(total_tasks):
        records.append(
            {
                "record_type": "grid_member",
                "schema_version": "silenttwin.agentdojo.grid.v1",
                "task_id": task_id,
                "batch_offset": 0,
                "cell_index": task_id,
                "configuration": {
                    "fixture_mode": fixture_mode,
                    "models": (
                        [{"implementation": "local_transformers", "role": "attacker"}]
                        if learned_model
                        else []
                    ),
                },
                "configuration_hash": "a" * 64,
                "shard_id": "b" * 64,
            }
        )
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _run_e1(manifest: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    values = _clean_scheduler_environment()
    values.pop("E1_STAGE", None)
    values.update(
        {
            "STAGE": "run",
            "GRID_MANIFEST": str(manifest),
            "PYTHON_BIN": "/definitely/not/a/python",
            "AGENTDOJO_FAKE_MODEL": "1",
            "AGENTDOJO_REQUIRES_GPU": "0",
            **environment,
        }
    )
    return subprocess.run(
        ["bash", str(EXPERIMENT_DIR / ENTRYPOINTS[2])],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_pair_observe(**environment: str) -> subprocess.CompletedProcess[str]:
    values = _clean_scheduler_environment()
    for name in (
        "AGENTDOJO_MONITOR_CHECKPOINT",
        "AGENTDOJO_MONITOR_CHECKPOINT_MONITOR_A",
        "AGENTDOJO_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        values.pop(name, None)
    values.update(
        {
            "STAGE": "run",
            "PAIR_MINING_ACTION": "observe",
            "OBSERVATION_SPLIT": "train",
            "PYTHON_BIN": "/definitely/not/a/python",
            "AGENTDOJO_REQUIRES_GPU": "0",
            **environment,
        }
    )
    return subprocess.run(
        ["bash", str(EXPERIMENT_DIR / ENTRYPOINTS[1])],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_explicit_agentdojo_entrypoints_are_shell_valid() -> None:
    discovered = {
        path.name
        for path in EXPERIMENT_DIR.glob("*agentdojo*.sh")
        if path.name != "_agentdojo_common.sh"
    }
    assert discovered == set(ENTRYPOINTS)
    for name in ("_agentdojo_common.sh", *ENTRYPOINTS):
        result = subprocess.run(
            ["bash", "-n", str(EXPERIMENT_DIR / name)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_interface_realization_entrypoint_rejects_nontrain_split() -> None:
    values = _clean_scheduler_environment()
    values.update(
        {
            "PYTHON_BIN": "/definitely/not/a/python",
            "INTERFACE_REALIZATION_PROTOCOL": "/frozen/protocol.json",
            "INTERFACE_REALIZATION_INPUTS": "/frozen/inputs.jsonl",
            "INTERFACE_REALIZATION_OUTPUT": "/persistent/output",
            "AGENTDOJO_DEPENDENCY_LOCK": "/frozen/lock",
            "AGENTDOJO_MODEL_CACHE": "/persistent/cache",
            "AGENTDOJO_ATTACKER_CHECKPOINT": "/persistent/checkpoint",
            "AGENTDOJO_DATASET_SPLIT": "development",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(
                EXPERIMENT_DIR
                / "run_agentdojo_interface_realization_train_tier2.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "interface-realization replay is train-only" in result.stderr


def test_interface_realization_authorized_preflight_has_python_pin(tmp_path: Path) -> None:
    values = _clean_scheduler_environment()
    values.update(
        {
            "PBS_JOBID": "fixture.gaas",
            "PBS_ENVIRONMENT": "PBS_BATCH",
            "PBS_O_HOME": str(tmp_path),
            "PBS_JOBDIR": str(tmp_path),
            "PYTHON_BIN": sys.executable,
            "INTERFACE_REALIZATION_PROTOCOL": str(tmp_path / "missing-protocol.json"),
            "INTERFACE_REALIZATION_INPUTS": str(tmp_path / "missing-inputs.jsonl"),
            "INTERFACE_REALIZATION_OUTPUT": str(tmp_path / "output"),
            "AGENTDOJO_DEPENDENCY_LOCK": str(tmp_path / "lock"),
            "AGENTDOJO_MODEL_CACHE": str(tmp_path / "cache"),
            "AGENTDOJO_ATTACKER_CHECKPOINT": str(tmp_path / "checkpoint"),
            "AGENTDOJO_DATASET_SPLIT": "train",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(
                EXPERIMENT_DIR
                / "run_agentdojo_interface_realization_train_tier2.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "missing frozen interface-realization protocol" in result.stderr
    assert "unbound variable" not in result.stderr


def test_native_tool_interface_entrypoint_rejects_nontrain_split() -> None:
    values = _clean_scheduler_environment()
    values.update(
        {
            "PYTHON_BIN": "/definitely/not/a/python",
            "NATIVE_TOOL_PROTOCOL": "/frozen/protocol.json",
            "NATIVE_TOOL_INPUTS": "/frozen/inputs.jsonl",
            "NATIVE_TOOL_OUTPUT": "/persistent/output",
            "AGENTDOJO_DEPENDENCY_LOCK": "/frozen/lock",
            "AGENTDOJO_MODEL_CACHE": "/persistent/cache",
            "AGENTDOJO_VICTIM_CHECKPOINT": "/persistent/checkpoint",
            "AGENTDOJO_DATASET_SPLIT": "development",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(
                EXPERIMENT_DIR
                / "run_agentdojo_native_tool_interface_train_tier2.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "native tool-interface qualification is train-only" in result.stderr


def test_forced_choice_entrypoint_rejects_nontrain_split() -> None:
    values = _clean_scheduler_environment()
    values.update(
        {
            "PYTHON_BIN": "/definitely/not/a/python",
            "FORCED_CHOICE_PROTOCOL": "/frozen/protocol.json",
            "FORCED_CHOICE_INPUTS": "/frozen/inputs.jsonl",
            "FORCED_CHOICE_OUTPUT": "/persistent/output",
            "AGENTDOJO_DEPENDENCY_LOCK": "/frozen/lock",
            "AGENTDOJO_MODEL_CACHE": "/persistent/cache",
            "AGENTDOJO_ATTACKER_CHECKPOINT": "/persistent/checkpoint",
            "AGENTDOJO_DATASET_SPLIT": "development",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(
                EXPERIMENT_DIR
                / "run_agentdojo_forced_choice_readout_train_tier2.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "forced-choice readout is train-only" in result.stderr


def test_forced_choice_authorized_preflight_has_python_pin(tmp_path: Path) -> None:
    values = _clean_scheduler_environment()
    values.update(
        {
            "PBS_JOBID": "fixture.gaas",
            "PBS_ENVIRONMENT": "PBS_BATCH",
            "PBS_O_HOME": str(tmp_path),
            "PBS_JOBDIR": str(tmp_path),
            "PYTHON_BIN": sys.executable,
            "FORCED_CHOICE_PROTOCOL": str(tmp_path / "missing-protocol.json"),
            "FORCED_CHOICE_INPUTS": str(tmp_path / "missing-inputs.jsonl"),
            "FORCED_CHOICE_OUTPUT": str(tmp_path / "output"),
            "AGENTDOJO_DEPENDENCY_LOCK": str(tmp_path / "lock"),
            "AGENTDOJO_MODEL_CACHE": str(tmp_path / "cache"),
            "AGENTDOJO_ATTACKER_CHECKPOINT": str(tmp_path / "checkpoint"),
            "AGENTDOJO_DATASET_SPLIT": "train",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(
                EXPERIMENT_DIR
                / "run_agentdojo_forced_choice_readout_train_tier2.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "missing frozen forced-choice protocol" in result.stderr
    assert "unbound variable" not in result.stderr


def test_recipient_separation_entrypoint_requires_frozen_inputs() -> None:
    values = _clean_scheduler_environment()
    for name in (
        "AGENTDOJO_STRATEGY_CATALOG",
        "AGENTDOJO_PAIR_REGISTRY",
        "AGENTDOJO_GRID_PLAN",
    ):
        values.pop(name, None)
    values.update({"STAGE": "grid", "RECIPIENT_EXPERIMENT": "e1"})

    result = subprocess.run(
        [
            "bash",
            str(
                EXPERIMENT_DIR
                / "run_agentdojo_recipient_separation_train_tier2.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "scientific-v6 recipient-separation candidate catalog" in result.stderr


def test_recipient_separation_entrypoint_rejects_nontrain_split() -> None:
    values = _clean_scheduler_environment()
    values.update(
        {
            "STAGE": "grid",
            "RECIPIENT_EXPERIMENT": "e2",
            "AGENTDOJO_STRATEGY_CATALOG": "/frozen/strategies.json",
            "AGENTDOJO_PAIR_REGISTRY": "/frozen/pairs.json",
            "AGENTDOJO_GRID_PLAN": "/frozen/plan.json",
            "AGENTDOJO_DATASET_SPLIT": "development",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(
                EXPERIMENT_DIR
                / "run_agentdojo_recipient_separation_train_tier2.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "recipient separation is train-only" in result.stderr


def test_spooled_entrypoint_uses_explicit_repository_root(tmp_path: Path) -> None:
    source = EXPERIMENT_DIR / ENTRYPOINTS[2]
    spooled = tmp_path / "pbs-spooled-script.sh"
    spooled.write_bytes(source.read_bytes())
    values = _clean_scheduler_environment()
    values.update(
        {
            "AGENTDOJO_REPO_ROOT": str(REPO_ROOT),
            "STAGE": "run",
            "GRID_MANIFEST": str(tmp_path / "unread-before-authorization.jsonl"),
            "PYTHON_BIN": "/definitely/not/a/python",
            "AGENTDOJO_FAKE_MODEL": "1",
            "AGENTDOJO_REQUIRES_GPU": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(spooled)],
        cwd=tmp_path,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "outside an authorized Slurm or PBS job" in result.stderr
    assert "_agentdojo_common.sh" not in result.stderr


def test_checkpoint_conformance_rejects_array_job_before_python() -> None:
    values = _clean_scheduler_environment()
    values.update(
        {
            "STAGE": "run",
            "PBS_JOBID": "123[0].gaas",
            "PBS_ARRAY_INDEX": "0",
            "PBS_ENVIRONMENT": "PBS_BATCH",
            "PYTHON_BIN": "/definitely/not/a/python",
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / ENTRYPOINTS[-1])],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "must not be an array job" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_checkpoint_conformance_reports_missing_runtime_fingerprint() -> None:
    values = _clean_scheduler_environment()
    values.pop("AGENTDOJO_RUNTIME_FINGERPRINT", None)
    values.update(
        {
            "STAGE": "run",
            "PBS_JOBID": "123.gaas",
            "PBS_ENVIRONMENT": "PBS_BATCH",
            "PYTHON_BIN": "/definitely/not/a/python",
            "AGENTDOJO_DATASET_SPLIT": "development",
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / ENTRYPOINTS[-1])],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "frozen sha256 runtime fingerprint is required" in result.stderr
    assert "unbound variable" not in result.stderr


def test_scripts_contain_no_submission_or_guessed_site_gpu_flags() -> None:
    contents = "\n".join(
        (EXPERIMENT_DIR / name).read_text(encoding="utf-8")
        for name in ("_agentdojo_common.sh", *ENTRYPOINTS)
    )
    assert "sbatch" not in contents
    assert "qsub" not in contents
    assert "#SBATCH" not in contents
    assert "#PBS" not in contents
    for flag in ("--account", "--partition", "--gres", "--gpus"):
        assert flag not in contents


def test_checkpoint_conformance_rejects_dirty_source_before_runtime_or_gpu_load() -> None:
    contents = (EXPERIMENT_DIR / ENTRYPOINTS[-1]).read_text(encoding="utf-8")

    dirty_check = contents.index("git status --porcelain --untracked-files=all")
    runtime_check = contents.index("observed_runtime_fingerprint=")
    gpu_check = contents.index("nvidia-smi -L")

    assert dirty_check < runtime_check < gpu_check


def test_out_of_range_array_fails_before_python_is_inspected(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest, total_tasks=2)
    result = _run_e1(
        manifest,
        SLURM_JOB_ID="123",
        SLURM_ARRAY_TASK_ID="2",
    )
    assert result.returncode == 2
    assert "out of range" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_pbs_array_index_is_accepted_before_python_is_inspected(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest, total_tasks=2)
    result = _run_e1(
        manifest,
        PBS_JOBID="123[1].gaas",
        PBS_ARRAY_ID="123[].gaas",
        PBS_ARRAY_INDEX="1",
        PBS_ENVIRONMENT="PBS_BATCH",
    )
    assert result.returncode == 2
    assert "PYTHON_BIN is unavailable" in result.stderr
    assert "PBS_ARRAY_INDEX" not in result.stderr


def test_pbs_out_of_range_array_fails_before_python_is_inspected(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest, total_tasks=2)
    result = _run_e1(
        manifest,
        PBS_JOBID="123[2].gaas",
        PBS_ARRAY_ID="123[].gaas",
        PBS_ARRAY_INDEX="2",
        PBS_ENVIRONMENT="PBS_BATCH",
    )
    assert result.returncode == 2
    assert "PBS_ARRAY_INDEX=2 is out of range" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_pbs_array_index_must_be_present_and_canonical(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest, total_tasks=2)
    for raw_index, expected in (
        (None, "PBS_ARRAY_INDEX is required"),
        ("01", "must be a non-negative base-10 integer"),
        ("-1", "must be a non-negative base-10 integer"),
        ("task-1", "must be a non-negative base-10 integer"),
    ):
        environment = {
            "PBS_JOBID": "123[].gaas",
            "PBS_ARRAY_ID": "123[].gaas",
            "PBS_ENVIRONMENT": "PBS_BATCH",
        }
        if raw_index is not None:
            environment["PBS_ARRAY_INDEX"] = raw_index
        result = _run_e1(manifest, **environment)
        assert result.returncode == 2
        assert expected in result.stderr
        assert "PYTHON_BIN is unavailable" not in result.stderr


def test_pbs_job_requires_batch_environment_before_python(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest)
    result = _run_e1(
        manifest,
        PBS_JOBID="123[0].gaas",
        PBS_ARRAY_INDEX="0",
        PBS_ENVIRONMENT="PBS_INTERACTIVE",
    )
    assert result.returncode == 2
    assert "requires PBS_ENVIRONMENT=PBS_BATCH" in result.stderr
    assert "PYTHON_BIN" not in result.stderr


def test_ambiguous_scheduler_context_is_rejected_before_python(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest)
    result = _run_e1(
        manifest,
        SLURM_JOB_ID="slurm-123",
        SLURM_ARRAY_TASK_ID="0",
        PBS_JOBID="pbs-123[0].gaas",
        PBS_ARRAY_INDEX="0",
        PBS_ENVIRONMENT="PBS_BATCH",
    )
    assert result.returncode == 2
    assert "ambiguous scheduler context" in result.stderr
    assert "PYTHON_BIN" not in result.stderr


def test_run_fails_outside_scheduler_before_python_or_model_validation(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest)
    result = _run_e1(manifest, SLURM_ARRAY_TASK_ID="0")
    assert result.returncode == 2
    assert "outside an authorized Slurm or PBS job" in result.stderr
    assert "PYTHON_BIN" not in result.stderr


def test_ephemeral_authoritative_cache_is_rejected_before_python(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest)
    scratch = tmp_path / "slurm-scratch"
    scratch.mkdir()
    result = _run_e1(
        manifest,
        SLURM_JOB_ID="123",
        SLURM_ARRAY_TASK_ID="0",
        SLURM_TMPDIR=str(scratch),
        AGENTDOJO_FAKE_MODEL="0",
        AGENTDOJO_REQUIRES_GPU="0",
        AGENTDOJO_MODEL_CACHE=str(scratch / "models"),
    )
    assert result.returncode == 2
    assert "must be persistent" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_pbs_ephemeral_authoritative_cache_is_rejected_before_python(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest)
    for scratch_variable in ("PBS_JOBDIR", "TMPDIR"):
        scratch = tmp_path / scratch_variable.lower()
        scratch.mkdir()
        result = _run_e1(
            manifest,
            PBS_JOBID="123[0].gaas",
            PBS_ARRAY_ID="123[].gaas",
            PBS_ARRAY_INDEX="0",
            PBS_ENVIRONMENT="PBS_BATCH",
            AGENTDOJO_FAKE_MODEL="0",
            AGENTDOJO_REQUIRES_GPU="0",
            AGENTDOJO_MODEL_CACHE=str(scratch / "models"),
            **{scratch_variable: str(scratch)},
        )
        assert result.returncode == 2
        assert scratch_variable in result.stderr
        assert "must be persistent" in result.stderr
        assert "PYTHON_BIN is unavailable" not in result.stderr


def test_pbs_home_jobdir_is_not_misclassified_as_scratch_before_python(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    output_root = home / "persistent-results"
    home.mkdir()
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest)
    result = _run_e1(
        manifest,
        PBS_JOBID="123[0].gaas",
        PBS_ARRAY_ID="123[].gaas",
        PBS_ARRAY_INDEX="0",
        PBS_ENVIRONMENT="PBS_BATCH",
        PBS_JOBDIR=str(home),
        PBS_O_HOME=str(home),
        OUT_ROOT=str(output_root),
    )
    assert result.returncode == 2
    assert "PYTHON_BIN is unavailable" in result.stderr
    assert "scheduler scratch PBS_JOBDIR" not in result.stderr


def test_pair_observation_rejects_generic_profile_and_cache_scratch_paths_before_python(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "slurm-scratch"
    scratch.mkdir()
    for variable in (
        "AGENTDOJO_MONITOR_CHECKPOINT",
        "AGENTDOJO_MONITOR_CHECKPOINT_MONITOR_A",
        "AGENTDOJO_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        result = _run_pair_observe(
            SLURM_JOB_ID="123",
            SLURM_TMPDIR=str(scratch),
            **{variable: str(scratch / variable.lower())},
        )
        assert result.returncode == 2
        assert variable in result.stderr
        assert "must be persistent" in result.stderr
        assert "PYTHON_BIN is unavailable" not in result.stderr


def test_pair_observation_requires_persistent_model_cache_before_python() -> None:
    result = _run_pair_observe(
        SLURM_JOB_ID="123",
        AGENTDOJO_REQUIRES_GPU="0",
        AGENTDOJO_MODEL_CACHE="",
    )
    assert result.returncode == 2
    assert "learned monitors require AGENTDOJO_MODEL_CACHE" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_pbs_pair_observation_authorization_precedes_model_cache_check() -> None:
    result = _run_pair_observe(
        PBS_JOBID="123.gaas",
        PBS_ENVIRONMENT="PBS_BATCH",
        AGENTDOJO_REQUIRES_GPU="0",
        AGENTDOJO_MODEL_CACHE="",
    )
    assert result.returncode == 2
    assert "learned monitors require AGENTDOJO_MODEL_CACHE" in result.stderr
    assert "authorized Slurm or PBS job" not in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_experiment_alias_can_override_generic_stage(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest)
    values = _clean_scheduler_environment()
    values.update(
        {
            "STAGE": "grid",
            "E1_STAGE": "run",
            "GRID_MANIFEST": str(manifest),
            "PYTHON_BIN": "/definitely/not/a/python",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTDOJO_FAKE_MODEL": "1",
            "AGENTDOJO_REQUIRES_GPU": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / ENTRYPOINTS[2])],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "outside an authorized Slurm or PBS job" in result.stderr


def test_run_stage_invokes_the_concrete_checkpointed_grid_runner() -> None:
    common = (EXPERIMENT_DIR / "_agentdojo_common.sh").read_text(encoding="utf-8")
    assert "silenttwin.agentdojo.runner run-grid-task" in common
    assert '--grid-manifest "$GRID_MANIFEST" --task-id "$AGENTDOJO_TASK_ID"' in common


def test_checked_smoke_plan_uses_fixtures_but_production_defaults_stay_operator_owned() -> None:
    script = f"""
source {EXPERIMENT_DIR / '_agentdojo_common.sh'}
agentdojo_init e1 E1 controlled
printf '%s\\n%s\\n' "$AGENTDOJO_STRATEGY_CATALOG" "$AGENTDOJO_PAIR_REGISTRY"
"""
    smoke = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert smoke.stdout.splitlines() == [
        str(
            REPO_ROOT
            / "configs/silenttwin/agentdojo/fixtures/deterministic-fake-smoke-candidate-strategies-v1.json"
        ),
        str(
            REPO_ROOT
            / "configs/silenttwin/agentdojo/fixtures/deterministic-fake-smoke-pair-registry-v1.json"
        ),
    ]

    production = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "AGENTDOJO_GRID_PLAN": "/persistent/operator/controlled-local-v1.json",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert production.returncode == 0, production.stderr
    assert production.stdout.splitlines() == [
        str(REPO_ROOT / "configs/silenttwin/agentdojo/candidate-strategies-v1.json"),
        str(REPO_ROOT / "configs/silenttwin/agentdojo/pair-registry-v1.json"),
    ]

    pair_mining_script = script.replace(
        "agentdojo_init e1 E1 controlled",
        "agentdojo_init pair_mining PAIR_MINING controlled",
    )
    pair_mining = subprocess.run(
        ["bash", "-c", pair_mining_script],
        cwd=REPO_ROOT,
        env={**os.environ, "STAGE": "grid"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert pair_mining.returncode == 0, pair_mining.stderr
    assert pair_mining.stdout.splitlines() == production.stdout.splitlines()


def test_fake_model_execution_is_explicit_not_inferred_from_plan_filename() -> None:
    common = (EXPERIMENT_DIR / "_agentdojo_common.sh").read_text(encoding="utf-8")
    assert 'local fake_model="${AGENTDOJO_FAKE_MODEL:-0}"' in common
    assert 'basename -- "$AGENTDOJO_GRID_PLAN"' not in common
    assert "*fake*" not in common


def test_checked_smoke_filename_does_not_enable_fake_execution(tmp_path: Path) -> None:
    manifest = tmp_path / "grid.jsonl"
    _manifest(manifest, total_tasks=1)
    result = _run_e1(
        manifest,
        SLURM_JOB_ID="123",
        SLURM_ARRAY_TASK_ID="0",
        AGENTDOJO_FAKE_MODEL="",
        AGENTDOJO_REQUIRES_GPU="0",
    )
    assert result.returncode == 2
    assert "AGENTDOJO_FAKE_MODEL disagrees" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_selected_learned_role_requires_cache_before_python(tmp_path: Path) -> None:
    manifest = tmp_path / "learned-grid.jsonl"
    _manifest(
        manifest,
        total_tasks=1,
        fixture_mode=False,
        learned_model=True,
    )
    result = _run_e1(
        manifest,
        SLURM_JOB_ID="123",
        SLURM_ARRAY_TASK_ID="0",
        AGENTDOJO_FAKE_MODEL="0",
        AGENTDOJO_REQUIRES_GPU="0",
    )
    assert result.returncode == 2
    assert "selected learned models require AGENTDOJO_MODEL_CACHE" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr


def test_selected_model_free_production_task_needs_no_cache_or_gpu(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "model-free-grid.jsonl"
    _manifest(manifest, total_tasks=1, fixture_mode=False, learned_model=False)
    result = _run_e1(
        manifest,
        SLURM_JOB_ID="123",
        SLURM_ARRAY_TASK_ID="0",
        AGENTDOJO_FAKE_MODEL="0",
        AGENTDOJO_REQUIRES_GPU="0",
        AGENTDOJO_MODEL_CACHE="",
    )
    assert result.returncode == 2
    assert "PYTHON_BIN is unavailable" in result.stderr
    assert "AGENTDOJO_MODEL_CACHE" not in result.stderr
    assert "visible GPU" not in result.stderr


def test_selected_model_free_gpu_decision_is_exported_to_python_validator(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "model-free-grid.jsonl"
    _manifest(manifest, total_tasks=1, fixture_mode=False, learned_model=False)
    script = f"""
source {EXPERIMENT_DIR / '_agentdojo_common.sh'}
agentdojo_init e4 E4 controlled
agentdojo_run_preflight_before_python
printf '%s\n' "$AGENTDOJO_REQUIRES_GPU"
"""
    values = _clean_scheduler_environment()
    values.pop("AGENTDOJO_REQUIRES_GPU", None)
    values.update(
        {
            "STAGE": "run",
            "GRID_MANIFEST": str(manifest),
            "SLURM_JOB_ID": "123",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTDOJO_FAKE_MODEL": "0",
        }
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0\n"


def test_cpu_learned_override_sets_role_devices_before_python(tmp_path: Path) -> None:
    manifest = tmp_path / "learned-grid.jsonl"
    _manifest(manifest, total_tasks=1, fixture_mode=False, learned_model=True)
    script = f"""
source {EXPERIMENT_DIR / '_agentdojo_common.sh'}
agentdojo_init e1 E1 controlled
agentdojo_run_preflight_before_python
printf '%s %s %s %s\n' "$AGENTDOJO_REQUIRES_GPU" "$ATTACKER_DEVICE" "$VICTIM_DEVICE" "$MONITOR_DEVICE"
"""
    values = _clean_scheduler_environment()
    for name in ("ATTACKER_DEVICE", "VICTIM_DEVICE", "MONITOR_DEVICE"):
        values.pop(name, None)
    values.update(
        {
            "STAGE": "run",
            "GRID_MANIFEST": str(manifest),
            "SLURM_JOB_ID": "123",
            "SLURM_ARRAY_TASK_ID": "0",
            "AGENTDOJO_FAKE_MODEL": "0",
            "AGENTDOJO_REQUIRES_GPU": "0",
            "AGENTDOJO_MODEL_CACHE": "/persistent/model-cache",
        }
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=values,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0 cpu cpu cpu\n"


def test_cpu_override_rejects_cuda_role_device_before_python(tmp_path: Path) -> None:
    manifest = tmp_path / "learned-grid.jsonl"
    _manifest(manifest, total_tasks=1, fixture_mode=False, learned_model=True)
    result = _run_e1(
        manifest,
        SLURM_JOB_ID="123",
        SLURM_ARRAY_TASK_ID="0",
        AGENTDOJO_FAKE_MODEL="0",
        AGENTDOJO_REQUIRES_GPU="0",
        AGENTDOJO_MODEL_CACHE="/persistent/model-cache",
        ATTACKER_DEVICE="cuda:0",
    )
    assert result.returncode == 2
    assert "ATTACKER_DEVICE requests CUDA" in result.stderr
    assert "PYTHON_BIN is unavailable" not in result.stderr
