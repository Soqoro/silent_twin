from __future__ import annotations

import pytest

from silenttwin.io.provenance import _scheduler_metadata


SCHEDULER_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_GPUS",
    "PBS_JOBID",
    "PBS_ARRAY_ID",
    "PBS_ARRAY_INDEX",
    "PBS_QUEUE",
    "PBS_NODEFILE",
    "NCPUS",
)


def _clear_scheduler_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in SCHEDULER_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_pbs_scheduler_metadata_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_scheduler_environment(monkeypatch)
    monkeypatch.setenv("PBS_JOBID", "123[4].gaas")
    monkeypatch.setenv("PBS_ARRAY_ID", "123[].gaas")
    monkeypatch.setenv("PBS_ARRAY_INDEX", "4")
    monkeypatch.setenv("PBS_QUEUE", "operator-approved-queue")
    monkeypatch.setenv("PBS_NODEFILE", "/var/spool/pbs/aux/123.gaas")
    monkeypatch.setenv("NCPUS", "8")

    scheduler = _scheduler_metadata()

    assert scheduler["kind"] == "pbs"
    assert scheduler["job_id"] == "123[4].gaas"
    assert scheduler["array_job_id"] == "123[].gaas"
    assert scheduler["array_task_id"] == "4"
    assert scheduler["queue"] == "operator-approved-queue"
    assert scheduler["node_file"] == "/var/spool/pbs/aux/123.gaas"
    assert scheduler["cpus_per_task"] == "8"
    assert scheduler["partition"] is None
    assert scheduler["pbs_job_id"] == "123[4].gaas"
    assert scheduler["slurm_job_id"] is None


def test_slurm_scheduler_metadata_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_scheduler_environment(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "operator-approved-partition")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[01-02]")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")

    scheduler = _scheduler_metadata()

    assert scheduler["kind"] == "slurm"
    assert scheduler["job_id"] == "456"
    assert scheduler["array_task_id"] == "2"
    assert scheduler["partition"] == "operator-approved-partition"
    assert scheduler["queue"] is None
    assert scheduler["node_list"] == "node[01-02]"
    assert scheduler["job_gpus"] == "0"


def test_mixed_scheduler_metadata_is_explicitly_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_scheduler_environment(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    monkeypatch.setenv("PBS_JOBID", "123.gaas")

    scheduler = _scheduler_metadata()

    assert scheduler["kind"] == "ambiguous"
    assert scheduler["job_id"] is None
    assert scheduler["slurm_job_id"] == "456"
    assert scheduler["pbs_job_id"] == "123.gaas"
