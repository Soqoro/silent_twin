#!/bin/bash
# Scientific-v6 train-only fresh clean-repair component study.

set -euo pipefail

if [[ -n "${AGENTDOJO_REPO_ROOT:-}" ]]; then
    script_dir="$AGENTDOJO_REPO_ROOT/experiments/silenttwin"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    AGENTDOJO_REPO_ROOT="$(cd -- "$script_dir/../.." && pwd -P)"
fi

# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"

: "${PYTHON_BIN:?set the clean scientific-v6 learned Python binary}"
: "${CLEAN_REPAIR_PROTOCOL:?set the frozen clean-repair protocol}"
: "${CLEAN_REPAIR_INPUTS:?set the immutable 49-task train input corpus}"
: "${CLEAN_REPAIR_OUTPUT:?set the persistent clean-repair run directory}"
: "${AGENTDOJO_DEPENDENCY_LOCK:?set the frozen AgentDojo dependency lock}"
: "${AGENTDOJO_MODEL_CACHE:?set the persistent offline model cache}"
: "${AGENTDOJO_VICTIM_CHECKPOINT:?set the frozen Qwen repair checkpoint}"
AGENTDOJO_PYTHON_PIN="${AGENTDOJO_PYTHON_PIN:-$AGENTDOJO_REPO_ROOT/configs/silenttwin/agentdojo/python-version.txt}"
export AGENTDOJO_PYTHON_PIN

if [[ "${AGENTDOJO_DATASET_SPLIT:-train}" != train ]]; then
    agentdojo_die "clean-repair component study is train-only"
fi
if [[ -n "${PBS_ARRAY_INDEX:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    agentdojo_die "clean-repair component study requires one scalar scheduler job"
fi
if [[ -n "${CLEAN_REPAIR_MAX_NEW_TASKS:-}" && \
      ! "${CLEAN_REPAIR_MAX_NEW_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    agentdojo_die "CLEAN_REPAIR_MAX_NEW_TASKS must be a positive integer"
fi
export AGENTDOJO_DATASET_SPLIT=train
export AGENTDOJO_REQUIRES_GPU=1
export OUT_ROOT="$CLEAN_REPAIR_OUTPUT"

cd -- "$AGENTDOJO_REPO_ROOT"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$AGENTDOJO_REPO_ROOT/src:$PYTHONPATH"
else
    export PYTHONPATH="$AGENTDOJO_REPO_ROOT/src"
fi

agentdojo_require_scheduler_job "clean-repair component study"
agentdojo_reject_ephemeral_runtime_paths
agentdojo_offline_environment
agentdojo_activate_and_require_python311
[[ -f "$CLEAN_REPAIR_PROTOCOL" ]] || \
    agentdojo_die "missing frozen clean-repair protocol"
[[ -f "$CLEAN_REPAIR_INPUTS" ]] || \
    agentdojo_die "missing immutable clean-repair inputs"
command -v nvidia-smi >/dev/null 2>&1 || \
    agentdojo_die "the scheduler allocation exposes no nvidia-smi"
nvidia-smi -L >/dev/null 2>&1 || \
    agentdojo_die "the scheduler allocation exposes no visible GPU"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false SILENTTWIN_OFFLINE=1
export AGENTDOJO_ALLOW_API_FALLBACK=0 AGENTDOJO_ALLOW_MOCK_FALLBACK=0

command=(
    "$PYTHON_BIN" -m silenttwin.agentdojo.clean_repair run
    --protocol "$CLEAN_REPAIR_PROTOCOL"
    --inputs "$CLEAN_REPAIR_INPUTS"
    --dependency-lock "$AGENTDOJO_DEPENDENCY_LOCK"
    --checkpoint "$AGENTDOJO_VICTIM_CHECKPOINT"
    --model-cache "$AGENTDOJO_MODEL_CACHE"
    --output-dir "$CLEAN_REPAIR_OUTPUT"
    --device "${VICTIM_DEVICE:-cuda:0}"
)
if [[ -n "${CLEAN_REPAIR_MAX_NEW_TASKS:-}" ]]; then
    command+=(--max-new-tasks "$CLEAN_REPAIR_MAX_NEW_TASKS")
fi
exec "${command[@]}"
