"""Durable, configuration-bound per-episode checkpoints.

Each completed episode is its own atomically published JSON object.  A crash
can therefore lose at most the episode currently executing, and resume never
needs to trust a partially appended result stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from silenttwin.config import ExperimentConfig, stable_hash
from silenttwin.io.jsonl import ResultValidationError, atomic_write_json


CHECKPOINT_SCHEMA_VERSION = "silenttwin.checkpoint.v1"
CHECKPOINT_DIRECTORY = "checkpoints"
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"


def episode_id(config: ExperimentConfig, sample_index: int) -> str:
    """Return a stable ID for one episode within an exact scientific shard."""

    return stable_hash(
        {
            "configuration_hash": config.configuration_hash,
            "sample_index": int(sample_index),
            "dataset_revision": config.dataset_revision,
            "dataset_split": config.dataset_split,
        }
    )


@dataclass(slots=True)
class CheckpointStore:
    output_dir: Path
    config: ExperimentConfig
    sample_indices: tuple[int, ...]
    provenance_hash: str | None

    def __init__(
        self,
        output_dir: Path | str,
        config: ExperimentConfig,
        sample_indices: Iterable[int],
        provenance_hash: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.sample_indices = tuple(int(index) for index in sample_indices)
        self.provenance_hash = provenance_hash
        if len(set(self.sample_indices)) != len(self.sample_indices):
            raise ValueError("checkpoint sample indices must be unique")

    @property
    def directory(self) -> Path:
        return self.output_dir / CHECKPOINT_DIRECTORY

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / CHECKPOINT_MANIFEST

    @property
    def expected(self) -> dict[str, int]:
        return {episode_id(self.config, index): index for index in self.sample_indices}

    def _manifest(self, *, status: str, completed_ids: Iterable[str]) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": status,
            "configuration_hash": self.config.configuration_hash,
            "source_tree_hash": self.provenance_hash,
            "sample_start": self.config.sample_start,
            "expected_sample_count": self.config.num_samples,
            "expected_episode_ids": sorted(self.expected),
            "completed_episode_ids": sorted(completed_ids),
        }

    def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self._validate_manifest()
        else:
            atomic_write_json(
                self.manifest_path,
                self._manifest(status="running", completed_ids=()),
            )

    def _read_manifest(self) -> dict[str, Any]:
        import json

        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ResultValidationError("checkpoint manifest is missing") from error
        except (OSError, ValueError) as error:
            raise ResultValidationError(f"invalid checkpoint manifest: {error}") from error
        if not isinstance(value, dict):
            raise ResultValidationError("checkpoint manifest must be a JSON object")
        return value

    def _validate_manifest(self) -> dict[str, Any]:
        manifest = self._read_manifest()
        if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ResultValidationError("incompatible checkpoint schema version")
        if manifest.get("configuration_hash") != self.config.configuration_hash:
            raise ResultValidationError("checkpoint belongs to an incompatible configuration")
        if manifest.get("source_tree_hash") != self.provenance_hash:
            raise ResultValidationError("checkpoint belongs to an incompatible source tree")
        if manifest.get("expected_episode_ids") != sorted(self.expected):
            raise ResultValidationError("checkpoint episode set does not match this shard")
        return manifest

    def _path(self, identifier: str) -> Path:
        if identifier not in self.expected:
            raise ResultValidationError(f"unexpected checkpoint episode ID {identifier!r}")
        return self.directory / f"{identifier}.json"

    def load(self) -> dict[int, dict[str, Any]]:
        initial_manifest = self._validate_manifest()
        import json

        completed: dict[int, dict[str, Any]] = {}
        identifiers: set[str] = set()
        for path in sorted(self.directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ResultValidationError(f"invalid checkpoint {path}: {error}") from error
            if not isinstance(record, dict):
                raise ResultValidationError(f"checkpoint {path} is not a JSON object")
            identifier = str(record.get("episode_id", ""))
            if path.name != f"{identifier}.json" or identifier not in self.expected:
                raise ResultValidationError(f"checkpoint {path} has an unexpected episode ID")
            if record.get("configuration_hash") != self.config.configuration_hash:
                raise ResultValidationError(f"checkpoint {path} has an incompatible hash")
            sample = record.get("sample")
            if not isinstance(sample, dict) or sample.get("record_type") != "sample":
                raise ResultValidationError(f"checkpoint {path} lacks a sample record")
            index = int(record.get("sample_index", -1))
            if self.expected[identifier] != index or index in completed:
                raise ResultValidationError(f"checkpoint {path} has a duplicate or wrong index")
            identifiers.add(identifier)
            completed[index] = sample
        manifest = self._validate_manifest()
        declared = set(manifest.get("completed_episode_ids", []))
        if declared != identifiers:
            # ``save`` publishes the durable episode before updating this
            # advisory index. A process kill between those two atomic renames
            # therefore leaves a strict, recoverable state: every declared
            # episode still exists, and one or more fully validated expected
            # episode files may be newer than the running manifest. Treat the
            # episode files as authoritative and repair only that one-way
            # crash window. Missing declared files, unknown files, and any
            # disagreement after completion remain hard failures.
            if (
                initial_manifest.get("status") == "running"
                and declared < identifiers
            ):
                atomic_write_json(
                    self.manifest_path,
                    self._manifest(status="running", completed_ids=identifiers),
                )
            else:
                raise ResultValidationError(
                    "checkpoint manifest and completed episode files disagree"
                )
        # Callers iterate this mapping to reconstruct result rows.  Episode
        # filenames are content hashes, so filesystem/name order is unrelated
        # to the preregistered sample order.
        return {index: completed[index] for index in sorted(completed)}

    def save(self, sample_index: int, sample: Mapping[str, Any]) -> None:
        identifier = episode_id(self.config, sample_index)
        path = self._path(identifier)
        if path.exists():
            raise ResultValidationError(f"checkpoint already exists for episode {identifier}")
        atomic_write_json(
            path,
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "configuration_hash": self.config.configuration_hash,
                "episode_id": identifier,
                "sample_index": sample_index,
                "sample": dict(sample),
            },
        )
        completed = {
            existing.stem for existing in self.directory.glob("*.json") if existing.is_file()
        }
        atomic_write_json(
            self.manifest_path,
            self._manifest(status="running", completed_ids=completed),
        )

    def mark_complete(self) -> None:
        completed = set(self.expected)
        actual = {path.stem for path in self.directory.glob("*.json") if path.is_file()}
        if actual != completed:
            raise ResultValidationError("cannot complete a checkpoint with missing episodes")
        atomic_write_json(
            self.manifest_path,
            self._manifest(status="complete", completed_ids=completed),
        )


__all__ = [
    "CHECKPOINT_DIRECTORY",
    "CHECKPOINT_MANIFEST",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointStore",
    "episode_id",
]
