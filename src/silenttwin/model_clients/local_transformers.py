"""Local-files-only Hugging Face model adapter for scheduled GPU jobs.

Imports and model loading are lazy.  This module never downloads a checkpoint,
contacts an API, or makes CPU-only CI depend on heavyweight GPU packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from silenttwin.attackers.llm_attacker import ModelResponse, ModelUsage
from silenttwin.config import is_immutable_model_revision, stable_hash
from silenttwin.io.jsonl import atomic_write_json
from silenttwin.schemas import canonical_json


class LocalModelUnavailableError(RuntimeError):
    pass


CHECKPOINT_FINGERPRINT_SCHEMA = "silenttwin.checkpoint-fingerprint.v1"


def _build_checkpoint_fingerprint_manifest(model_dir: Path | str) -> dict[str, Any]:
    """Hash every regular file and return an auditable tree manifest.

    The digest binds filenames, sizes, and bytes. It is intentionally stricter
    than a directory name or modification time, either of which can silently
    identify different weights under the same requested revision.
    """

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"local checkpoint is not a directory: {root}")
    files = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files:
        raise ValueError(f"local checkpoint directory contains no files: {root}")
    tree = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                size += len(chunk)
                content.update(chunk)
        tree.update(len(relative).to_bytes(8, "big"))
        tree.update(relative)
        tree.update(size.to_bytes(16, "big"))
        tree.update(content.digest())
        stat = path.stat()
        records.append(
            {
                "path": relative.decode("utf-8"),
                "size": size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "sha256": content.hexdigest(),
            }
        )
    return {
        "schema_version": CHECKPOINT_FINGERPRINT_SCHEMA,
        "model_dir": str(root),
        "model_revision": f"sha256:{tree.hexdigest()}",
        "files": records,
    }


def fingerprint_local_checkpoint(model_dir: Path | str) -> str:
    """Return an exact full-byte fingerprint for one local checkpoint."""

    return str(_build_checkpoint_fingerprint_manifest(model_dir)["model_revision"])


def checkpoint_fingerprint_manifest_path(
    model_dir: Path | str, cache_dir: Path | str
) -> Path:
    root = Path(model_dir).expanduser().resolve()
    identity = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return (
        Path(cache_dir).expanduser()
        / "silenttwin-checkpoint-fingerprints"
        / f"{identity}.json"
    )


def prepare_local_checkpoint_fingerprint(
    model_dir: Path | str, cache_dir: Path | str
) -> dict[str, Any]:
    """Perform one full audit and atomically cache its immutable manifest."""

    manifest = _build_checkpoint_fingerprint_manifest(model_dir)
    path = checkpoint_fingerprint_manifest_path(model_dir, cache_dir)
    atomic_write_json(path, manifest)
    return {
        "model_revision": manifest["model_revision"],
        "manifest_path": str(path),
        "manifest_hash": stable_hash(manifest),
    }


@dataclass(frozen=True, slots=True)
class LocalModelConfig:
    model_id: str
    model_revision: str
    tokenizer_revision: str | None = None
    checkpoint_fingerprint: str | None = None
    semantic_model_id: str | None = None
    model_cache_dir: Path | None = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    decoding_seed: int = 0
    batch_size: int = 1
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("MODEL_ID must be configured")
        if not self.model_revision:
            raise ValueError("MODEL_REVISION must be configured")
        if not is_immutable_model_revision(self.model_revision):
            raise ValueError(
                "MODEL_REVISION must be an exact 40-64 hex commit or "
                "sha256:<64-hex> local-checkpoint fingerprint"
            )
        if self.model_cache_dir is not None:
            object.__setattr__(self, "model_cache_dir", Path(self.model_cache_dir))
        if self.checkpoint_fingerprint is not None and (
            not self.checkpoint_fingerprint.startswith("sha256:")
            or not is_immutable_model_revision(self.checkpoint_fingerprint)
        ):
            raise ValueError(
                "CHECKPOINT_FINGERPRINT must be sha256:<64-hex> when provided"
            )
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("DTYPE must be float32, float16, or bfloat16")
        if self.max_new_tokens <= 0 or self.batch_size <= 0:
            raise ValueError("MAX_NEW_TOKENS and BATCH_SIZE must be positive")
        if self.batch_size != 1:
            raise ValueError(
                "the single-prompt local adapter supports BATCH_SIZE=1 only"
            )
        if not 0.0 <= self.temperature:
            raise ValueError("TEMPERATURE must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("TOP_P must lie in (0, 1]")
        if self.decoding_seed < 0:
            raise ValueError("DECODING_SEED must be non-negative")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "LocalModelConfig":
        values = os.environ if environment is None else environment
        cache = values.get("MODEL_CACHE_DIR")
        return cls(
            model_id=values.get("MODEL_ID", ""),
            model_revision=values.get("MODEL_REVISION", ""),
            tokenizer_revision=values.get("TOKENIZER_REVISION") or None,
            model_cache_dir=Path(cache) if cache else None,
            dtype=values.get("DTYPE", "bfloat16"),
            max_new_tokens=int(values.get("MAX_NEW_TOKENS", "128")),
            temperature=float(values.get("TEMPERATURE", "0")),
            top_p=float(values.get("TOP_P", "1")),
            decoding_seed=int(values.get("DECODING_SEED", "0")),
            batch_size=int(values.get("BATCH_SIZE", "1")),
            device=values.get("MODEL_DEVICE", "cuda"),
        )


class LocalTransformersModelClient:
    """Lazy local checkpoint client implementing ``ModelClient``."""

    def __init__(self, config: LocalModelConfig) -> None:
        if not isinstance(config, LocalModelConfig):
            raise TypeError("config must be LocalModelConfig")
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._transformers_version: str | None = None
        self._resolved_model_revision: str | None = None
        self._resolved_tokenizer_revision: str | None = None
        self._resolved_checkpoint_fingerprint: str | None = None
        self._local_verification_mode: str | None = None
        self._local_fingerprint_manifest_hash: str | None = None

    @staticmethod
    def _read_fingerprint_manifest(path: Path) -> dict[str, Any]:
        import json

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LocalModelUnavailableError(
                f"invalid checkpoint fingerprint manifest {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise LocalModelUnavailableError(
                f"checkpoint fingerprint manifest is not an object: {path}"
            )
        return value

    @staticmethod
    def _validate_cached_fingerprint(
        root: Path,
        manifest: Mapping[str, Any],
        requested_revision: str,
    ) -> None:
        if manifest.get("schema_version") != CHECKPOINT_FINGERPRINT_SCHEMA:
            raise LocalModelUnavailableError("unsupported checkpoint fingerprint schema")
        if manifest.get("model_dir") != str(root):
            raise LocalModelUnavailableError(
                "checkpoint fingerprint manifest belongs to a different directory"
            )
        if str(manifest.get("model_revision", "")).lower() != requested_revision.lower():
            raise LocalModelUnavailableError(
                "checkpoint fingerprint manifest does not match requested revision"
            )
        rows = manifest.get("files")
        if not isinstance(rows, list) or not rows:
            raise LocalModelUnavailableError("checkpoint fingerprint manifest has no files")
        expected: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                raise LocalModelUnavailableError("invalid checkpoint fingerprint file row")
            relative = str(row["path"])
            if relative in expected:
                raise LocalModelUnavailableError("duplicate checkpoint fingerprint path")
            expected[relative] = row
        actual_paths = {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*")
            if path.is_file()
        }
        if set(actual_paths) != set(expected):
            raise LocalModelUnavailableError(
                "local checkpoint file set changed after fingerprinting"
            )
        for relative, path in actual_paths.items():
            stat = path.stat()
            row = expected[relative]
            if (
                row.get("size") != stat.st_size
                or row.get("mtime_ns") != stat.st_mtime_ns
                or row.get("ctime_ns") != stat.st_ctime_ns
            ):
                raise LocalModelUnavailableError(
                    f"local checkpoint file changed after fingerprinting: {relative}"
                )

    def _verify_local_identity(self) -> None:
        model_path = Path(self.config.model_id).expanduser()
        if not model_path.is_dir():
            return
        root = model_path.resolve()
        manifest_path = (
            checkpoint_fingerprint_manifest_path(root, self.config.model_cache_dir)
            if self.config.model_cache_dir is not None
            else None
        )
        manifest: dict[str, Any]
        if manifest_path is not None and manifest_path.is_file():
            manifest = self._read_fingerprint_manifest(manifest_path)
            requested_fingerprint = (
                self.config.checkpoint_fingerprint or self.config.model_revision
            )
            self._validate_cached_fingerprint(root, manifest, requested_fingerprint)
            # Metadata is only an early corruption check.  The cache and its
            # manifest may both be operator-writable, so publication runs
            # always recompute the complete byte-tree digest before loading.
            audited = _build_checkpoint_fingerprint_manifest(root)
            if audited["model_revision"] != manifest["model_revision"]:
                raise LocalModelUnavailableError(
                    "full checkpoint rehash disagrees with cached fingerprint"
                )
            manifest = audited
            self._local_verification_mode = "full_tree_sha256_audit"
        else:
            manifest = _build_checkpoint_fingerprint_manifest(root)
            if manifest_path is not None:
                atomic_write_json(manifest_path, manifest)
            self._local_verification_mode = "full_tree_sha256_initialization"
        resolved = str(manifest["model_revision"])
        requested_fingerprint = (
            self.config.checkpoint_fingerprint or self.config.model_revision
        )
        if resolved != requested_fingerprint.lower():
            raise LocalModelUnavailableError(
                "local checkpoint fingerprint mismatch: requested "
                f"{requested_fingerprint!r}, resolved {resolved!r}; rerun "
                "`python -m silenttwin.cli fingerprint-model` and freeze the grid"
            )
        self._resolved_checkpoint_fingerprint = resolved
        self._local_fingerprint_manifest_hash = stable_hash(manifest)

    def _load(self) -> None:
        if self._model is not None:
            return
        self._verify_local_identity()
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LocalModelUnavailableError(
                "local Tier-2 inference requires cluster-provided `torch` and `transformers`; "
                "they are intentionally absent from the CPU development environment"
            ) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise LocalModelUnavailableError(
                "CUDA was requested but is unavailable; run Tier-2 inference in an "
                "authorized GPU scheduler job"
            )
        cache_dir = str(self.config.model_cache_dir) if self.config.model_cache_dir else None
        tokenizer_revision = self.config.tokenizer_revision or self.config.model_revision
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                revision=tokenizer_revision,
                cache_dir=cache_dir,
                local_files_only=True,
                trust_remote_code=False,
            )
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[self.config.dtype]
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                revision=self.config.model_revision,
                cache_dir=cache_dir,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
            )
        except OSError as exc:
            raise LocalModelUnavailableError(
                f"model {self.config.model_id!r} revision {self.config.model_revision!r} "
                "is not present in the configured local cache; no download was attempted"
            ) from exc
        loaded_model = model.to(self.config.device)
        loaded_model.eval()
        local_checkpoint = self._resolved_checkpoint_fingerprint is not None
        if local_checkpoint:
            # A directory tree is bound byte-for-byte above.  Preserve the
            # separately frozen semantic model/tokenizer revisions as
            # deployment assertions; local ``from_pretrained`` does not
            # reliably expose a Hub commit for arbitrary exported trees.
            self._resolved_model_revision = self.config.model_revision.lower()
            self._resolved_tokenizer_revision = tokenizer_revision.lower()
        elif self._resolved_model_revision is None:
            resolved_model = self._revision(loaded_model)
            resolved_tokenizer = self._revision(tokenizer)
            if resolved_model is None or resolved_model.lower() != self.config.model_revision.lower():
                raise LocalModelUnavailableError(
                    "the locally cached model did not resolve to the exact requested commit: "
                    f"requested {self.config.model_revision!r}, resolved {resolved_model!r}"
                )
            requested_tokenizer = (
                self.config.tokenizer_revision or self.config.model_revision
            )
            if (
                resolved_tokenizer is None
                or resolved_tokenizer.lower() != requested_tokenizer.lower()
            ):
                raise LocalModelUnavailableError(
                    "the locally cached tokenizer did not resolve to the exact requested commit: "
                    f"requested {requested_tokenizer!r}, resolved {resolved_tokenizer!r}"
                )
            self._resolved_model_revision = resolved_model.lower()
            self._resolved_tokenizer_revision = resolved_tokenizer.lower()
        self._torch = torch
        self._transformers_version = transformers.__version__
        self._tokenizer = tokenizer
        self._model = loaded_model

    def ensure_available(self) -> None:
        """Preflight the local checkpoint and CUDA environment once.

        Tier-2 run construction calls this outside the attacker callback so a
        missing checkpoint/package/GPU terminates the shard clearly instead of
        producing an entire shard of invalid model outputs.
        """

        self._load()

    def failure_metadata(self) -> dict[str, Any]:
        """Return best-effort immutable provenance for a failed completion."""

        torch = self._torch
        gpu_name: str | None = None
        if torch is not None and torch.cuda.is_available():
            try:
                gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
            except Exception:
                gpu_name = None
        return {
            "client": "local_transformers",
            "model_id": self.config.semantic_model_id or self.config.model_id,
            "local_checkpoint_path": (
                self.config.model_id
                if Path(self.config.model_id).expanduser().is_dir()
                else None
            ),
            "requested_model_revision": self.config.model_revision,
            "model_revision": self._resolved_model_revision
            or self.config.model_revision,
            "requested_tokenizer_revision": (
                self.config.tokenizer_revision or self.config.model_revision
            ),
            "tokenizer_revision": self._resolved_tokenizer_revision
            or self.config.tokenizer_revision
            or self.config.model_revision,
            "local_checkpoint_verification_mode": self._local_verification_mode,
            "local_checkpoint_manifest_hash": self._local_fingerprint_manifest_hash,
            "local_checkpoint_fingerprint": self._resolved_checkpoint_fingerprint,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "batch_size": self.config.batch_size,
            "torch_version": getattr(torch, "__version__", None),
            "transformers_version": self._transformers_version,
            "cuda_version": (
                getattr(getattr(torch, "version", None), "cuda", None)
                if torch is not None
                else None
            ),
            "gpu_name": gpu_name,
            "external_api_calls": 0,
            "local_files_only": True,
        }

    @staticmethod
    def _revision(component: Any) -> str | None:
        value = (
            getattr(getattr(component, "config", None), "_commit_hash", None)
            or getattr(component, "init_kwargs", {}).get("_commit_hash")
        )
        return None if value is None else str(value)

    def _complete_rendered(
        self,
        rendered: str,
        *,
        input_prompt: str,
        seed: int,
        max_tokens: int,
        input_metadata: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        assert torch is not None and tokenizer is not None and model is not None
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        effective_seed = int(seed if seed is not None else self.config.decoding_seed)
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        do_sample = self.config.temperature > 0.0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": min(max_tokens, self.config.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        started = time.perf_counter()
        # ``manual_seed`` mutates process-global RNG streams.  Victim,
        # attacker, and monitor calls may share a worker process, so allowing
        # one role to advance another role's state would make decoding depend
        # on call ordering. ``fork_rng`` snapshots and restores every device
        # touched by this call while retaining compatibility with Transformers
        # versions whose ``generate`` path does not accept a Generator.
        fork_devices: list[int] = []
        if torch.cuda.is_available() and str(self.config.device).startswith("cuda"):
            parsed_device = torch.device(self.config.device)
            fork_devices = [
                torch.cuda.current_device()
                if parsed_device.index is None
                else int(parsed_device.index)
            ]
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.manual_seed(effective_seed)
            if fork_devices:
                torch.cuda.manual_seed_all(effective_seed)
            with torch.inference_mode():
                generated = model.generate(**encoded, **generate_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        completion_ids = generated[0, input_tokens:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        metadata = {
            "client": "local_transformers",
            "model_id": self.config.semantic_model_id or self.config.model_id,
            "local_checkpoint_path": (
                self.config.model_id
                if Path(self.config.model_id).expanduser().is_dir()
                else None
            ),
            "requested_model_revision": self.config.model_revision,
            "model_revision": self._resolved_model_revision,
            "requested_tokenizer_revision": (
                self.config.tokenizer_revision or self.config.model_revision
            ),
            "tokenizer_revision": self._resolved_tokenizer_revision,
            "local_checkpoint_fingerprint": (
                self._resolved_checkpoint_fingerprint
            ),
            "local_checkpoint_verification_mode": self._local_verification_mode,
            "local_checkpoint_manifest_hash": self._local_fingerprint_manifest_hash,
            "chat_template_hash": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
            "input_prompt_hash": hashlib.sha256(
                input_prompt.encode("utf-8")
            ).hexdigest(),
            "rendered_input_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            # Trusted provenance needs the exact tokenizer chat-template
            # material, not only a digest.  This metadata never enters the
            # agent-visible transcript.
            "rendered_input": rendered,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "decoding_seed": effective_seed,
            "batch_size": self.config.batch_size,
            "latency_ms": latency_ms,
            "retries": 0,
            "terminal_failure": None,
            "torch_version": torch.__version__,
            "transformers_version": self._transformers_version,
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_name": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
            "external_api_calls": 0,
            "local_files_only": True,
        }
        if input_metadata is not None:
            metadata.update(dict(input_metadata))
        return ModelResponse(
            text=text,
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=int(completion_ids.shape[-1]),
            ),
            metadata=metadata,
        )

    def complete(
        self,
        prompt: str,
        *,
        seed: int = 0,
        max_tokens: int = 128,
    ) -> ModelResponse:
        self._load()
        tokenizer = self._tokenizer
        assert tokenizer is not None
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = prompt
        if not isinstance(rendered, str):
            raise LocalModelUnavailableError(
                "tokenizer chat template did not render text"
            )
        return self._complete_rendered(
            rendered,
            input_prompt=prompt,
            seed=seed,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _normalize_chat_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[dict[str, str], ...]:
        if isinstance(messages, (str, bytes, bytearray)) or not isinstance(
            messages, Sequence
        ):
            raise TypeError("messages must be a sequence of role/content mappings")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise TypeError(
                    f"chat message {index} must contain exactly role and content"
                )
            role = message["role"]
            content = message["content"]
            if (
                not isinstance(role, str)
                or role not in {"system", "user", "assistant"}
                or not isinstance(content, str)
            ):
                raise TypeError(f"chat message {index} has an invalid role or content")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("messages must not be empty")
        return tuple(normalized)

    def complete_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int = 0,
        max_tokens: int = 128,
    ) -> ModelResponse:
        """Complete an exact structured chat using the checkpoint's template."""

        normalized = self._normalize_chat_messages(messages)
        self._load()
        tokenizer = self._tokenizer
        assert tokenizer is not None
        if not getattr(tokenizer, "chat_template", None):
            raise LocalModelUnavailableError(
                "structured chat completion requires a tokenizer chat template"
            )
        rendered = tokenizer.apply_chat_template(
            list(normalized),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise LocalModelUnavailableError(
                "tokenizer chat template did not render text"
            )
        canonical_messages = canonical_json(list(normalized))
        return self._complete_rendered(
            rendered,
            input_prompt=canonical_messages,
            seed=seed,
            max_tokens=max_tokens,
            input_metadata={
                "input_mode": "structured_chat",
                "input_messages": [dict(message) for message in normalized],
                "input_messages_hash": hashlib.sha256(
                    canonical_messages.encode("utf-8")
                ).hexdigest(),
            },
        )


__all__ = [
    "CHECKPOINT_FINGERPRINT_SCHEMA",
    "checkpoint_fingerprint_manifest_path",
    "fingerprint_local_checkpoint",
    "LocalModelConfig",
    "LocalModelUnavailableError",
    "LocalTransformersModelClient",
    "prepare_local_checkpoint_fingerprint",
]
