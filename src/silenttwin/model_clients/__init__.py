"""Optional, explicitly configured local model clients."""

from silenttwin.model_clients.local_transformers import (
    CHECKPOINT_FINGERPRINT_SCHEMA,
    checkpoint_fingerprint_manifest_path,
    fingerprint_local_checkpoint,
    LocalModelConfig,
    LocalModelUnavailableError,
    LocalTransformersModelClient,
    prepare_local_checkpoint_fingerprint,
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
