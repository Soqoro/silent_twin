#!/bin/bash
# One-scenario, development-only learned-checkpoint protocol conformance.
set -euo pipefail

if [[ -n "${AGENTDOJO_REPO_ROOT:-}" ]]; then
    script_dir="$AGENTDOJO_REPO_ROOT/experiments/silenttwin"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"
agentdojo_init conformance CONFORMANCE controlled

CONFORMANCE_SPEC="${CONFORMANCE_SPEC:-}"
CONFORMANCE_OUTPUT="${CONFORMANCE_OUTPUT:-}"
ATTACKER_DEVICE="${ATTACKER_DEVICE:-cuda:0}"
MONITOR_DEVICE="${MONITOR_DEVICE:-cuda:0}"
EXPECTED_SOURCE_TREE_HASH="${EXPECTED_SOURCE_TREE_HASH:-}"
AGENTDOJO_REQUIRES_GPU="${AGENTDOJO_REQUIRES_GPU:-1}"
AGENTDOJO_FAKE_MODEL="${AGENTDOJO_FAKE_MODEL:-0}"

[[ "$AGENTDOJO_STAGE" == run ]] || \
    agentdojo_die "checkpoint conformance supports STAGE=run only"
agentdojo_require_scheduler_job "checkpoint conformance"
[[ -z "${PBS_ARRAY_INDEX:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
    agentdojo_die "one-scenario checkpoint conformance must not be an array job"
[[ "$AGENTDOJO_DATASET_SPLIT" == development ]] || \
    agentdojo_die "checkpoint conformance must use the development split"
[[ "$AGENTDOJO_REQUIRES_GPU" == 1 ]] || \
    agentdojo_die "checkpoint conformance requires one GPU"
[[ "$AGENTDOJO_FAKE_MODEL" == 0 ]] || \
    agentdojo_die "checkpoint conformance forbids fake models"
[[ "$ATTACKER_DEVICE" == cuda:0 && "$MONITOR_DEVICE" == cuda:0 ]] || \
    agentdojo_die "checkpoint conformance requires attacker and monitor on cuda:0"
[[ "${AGENTDOJO_RUNTIME_FINGERPRINT:-}" =~ ^sha256:[0-9a-f]{64}$ ]] || \
    agentdojo_die "a frozen sha256 runtime fingerprint is required"
[[ "$EXPECTED_SOURCE_TREE_HASH" =~ ^[0-9a-f]{64}$ ]] || \
    agentdojo_die "a frozen source-tree hash is required"
[[ -n "$CONFORMANCE_SPEC" && -f "$CONFORMANCE_SPEC" ]] || \
    agentdojo_die "CONFORMANCE_SPEC is not a readable file"
[[ -n "$CONFORMANCE_OUTPUT" && ! -e "$CONFORMANCE_OUTPUT" ]] || \
    agentdojo_die "CONFORMANCE_OUTPUT must be a new path"
[[ -f "$AGENTDOJO_CATALOG" && -f "$AGENTDOJO_SPLITS" && \
    -f "$AGENTDOJO_STRATEGY_CATALOG" ]] || \
    agentdojo_die "catalog, split, or conformance strategy artifact is absent"
[[ -n "${AGENTDOJO_MODEL_CACHE:-}" && -d "$AGENTDOJO_MODEL_CACHE" ]] || \
    agentdojo_die "AGENTDOJO_MODEL_CACHE is not a persistent directory"
[[ -n "${AGENTDOJO_ATTACKER_CHECKPOINT:-}" && \
    -d "$AGENTDOJO_ATTACKER_CHECKPOINT" ]] || \
    agentdojo_die "Qwen attacker checkpoint is absent"
[[ -n "${AGENTDOJO_MONITOR_CHECKPOINT:-}" && \
    -d "$AGENTDOJO_MONITOR_CHECKPOINT" ]] || \
    agentdojo_die "Granite monitor checkpoint is absent"
agentdojo_path_is_within "$CONFORMANCE_OUTPUT" "$OUT_ROOT" || \
    agentdojo_die "CONFORMANCE_OUTPUT must be below persistent OUT_ROOT"

agentdojo_reject_ephemeral_runtime_paths
agentdojo_offline_environment
agentdojo_activate_and_require_python311

observed_source_tree_hash="$(
    "$PYTHON_BIN" -c \
        'from silenttwin.io.provenance import source_tree_hash; print(source_tree_hash())'
)"
[[ "$observed_source_tree_hash" == "$EXPECTED_SOURCE_TREE_HASH" ]] || \
    agentdojo_die "active source tree differs from the frozen conformance tree"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
    agentdojo_die "checkpoint conformance requires a clean committed worktree"

observed_runtime_fingerprint="$(
    "$PYTHON_BIN" -m silenttwin.agentdojo.runtime_integrity \
        --dependency-lock "$AGENTDOJO_DEPENDENCY_LOCK" \
        --fingerprint-only
)"
[[ "$observed_runtime_fingerprint" == "$AGENTDOJO_RUNTIME_FINGERPRINT" ]] || \
    agentdojo_die "active learned runtime differs from the frozen fingerprint"

command -v nvidia-smi >/dev/null 2>&1 || \
    agentdojo_die "checkpoint conformance requires nvidia-smi"
nvidia-smi -L
nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
"$PYTHON_BIN" -c '
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible CUDA device, found {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory
if "H200" not in name.upper():
    raise SystemExit(f"expected NVIDIA H200, observed {name}")
if total < 100 * 1024**3:
    raise SystemExit(f"expected at least 100 GiB device memory, observed {total}")
print(f"validated_gpu={name} total_memory_bytes={total}")
'

mkdir -p -- "$(dirname -- "$CONFORMANCE_OUTPUT")"
exec "$PYTHON_BIN" -m silenttwin.agentdojo.conformance \
    --spec "$CONFORMANCE_SPEC" \
    --catalog "$AGENTDOJO_CATALOG" \
    --splits "$AGENTDOJO_SPLITS" \
    --strategy-catalog "$AGENTDOJO_STRATEGY_CATALOG" \
    --dependency-lock "$AGENTDOJO_DEPENDENCY_LOCK" \
    --attacker-checkpoint "$AGENTDOJO_ATTACKER_CHECKPOINT" \
    --monitor-checkpoint "$AGENTDOJO_MONITOR_CHECKPOINT" \
    --model-cache "$AGENTDOJO_MODEL_CACHE" \
    --attacker-device "$ATTACKER_DEVICE" \
    --monitor-device "$MONITOR_DEVICE" \
    --output "$CONFORMANCE_OUTPUT"
