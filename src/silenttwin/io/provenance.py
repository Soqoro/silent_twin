"""Reproducibility metadata without network access or credentials."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


def _repository_root() -> Path:
    # .../repo/src/silenttwin/io/provenance.py
    return Path(__file__).resolve().parents[3]


def _git_output(*arguments: str) -> str | None:
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=_repository_root(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return process.stdout.strip()


def source_tree_hash() -> str:
    """Hash scientific sources, including relevant uncommitted new files.

    The launchers determine array-to-configuration selection, so they are part
    of executable provenance even though they are not Python modules.  The
    declarative Tier-1 reference and packaging metadata are included for the
    same reason.  Generated outputs, tests, and prose documentation are
    intentionally excluded.
    """

    repository_root = _repository_root()
    source_patterns = (
        "src/silenttwin/**/*.py",
        "experiments/silenttwin/*.sh",
        "configs/silenttwin/**/*",
        "pyproject.toml",
        "requirements-dev.lock",
        "requirements-tier2-agentdojo.lock",
    )
    paths = {
        path
        for pattern in source_patterns
        for path in repository_root.glob(pattern)
        if path.is_file()
    }
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def collect_provenance() -> dict[str, Any]:
    try:
        package_version = version("silenttwin")
    except PackageNotFoundError:
        package_version = "0.1.0+source"
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    scheduler = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }
    return {
        "code_revision": _git_output("rev-parse", "HEAD") or "unknown",
        "code_dirty": bool(status),
        "source_tree_hash": source_tree_hash(),
        "package_version": package_version,
        "python_implementation": platform.python_implementation(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "scheduler": scheduler,
        "gpu_environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
    }


_COMPATIBILITY_KEYS = (
    "code_revision",
    "code_dirty",
    "source_tree_hash",
    "package_version",
    "python_implementation",
    "python_version",
    "platform",
)


def provenance_compatible(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    return all(recorded.get(key) == current.get(key) for key in _COMPATIBILITY_KEYS)


def provenance_mismatches(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    return {
        key: (recorded.get(key), current.get(key))
        for key in _COMPATIBILITY_KEYS
        if recorded.get(key) != current.get(key)
    }
