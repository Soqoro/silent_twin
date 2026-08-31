#!/bin/bash
# Scientific-v6 adaptive train-only forced-choice source readout.

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
: "${FORCED_CHOICE_PROTOCOL:?set the frozen forced-choice protocol}"
: "${FORCED_CHOICE_INPUTS:?set the immutable train readout inputs}"
: "${FORCED_CHOICE_OUTPUT:?set the persistent readout output directory}"
: "${AGENTDOJO_DEPENDENCY_LOCK:?set the frozen AgentDojo dependency lock}"
: "${AGENTDOJO_MODEL_CACHE:?set the persistent offline model cache}"
: "${AGENTDOJO_ATTACKER_CHECKPOINT:?set the frozen Qwen checkpoint}"
AGENTDOJO_PYTHON_PIN="${AGENTDOJO_PYTHON_PIN:-$AGENTDOJO_REPO_ROOT/configs/silenttwin/agentdojo/python-version.txt}"
export AGENTDOJO_PYTHON_PIN

if [[ "${AGENTDOJO_DATASET_SPLIT:-train}" != train ]]; then
    agentdojo_die "forced-choice readout is train-only"
fi
if [[ -n "${PBS_ARRAY_INDEX:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    agentdojo_die "forced-choice readout requires one scalar scheduler job"
fi
export AGENTDOJO_DATASET_SPLIT=train
export AGENTDOJO_REQUIRES_GPU=1
export OUT_ROOT="$FORCED_CHOICE_OUTPUT"

cd -- "$AGENTDOJO_REPO_ROOT"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$AGENTDOJO_REPO_ROOT/src:$PYTHONPATH"
else
    export PYTHONPATH="$AGENTDOJO_REPO_ROOT/src"
fi

agentdojo_require_scheduler_job "forced-choice readout"
agentdojo_reject_ephemeral_runtime_paths
agentdojo_offline_environment
agentdojo_activate_and_require_python311
[[ -f "$FORCED_CHOICE_PROTOCOL" ]] || \
    agentdojo_die "missing frozen forced-choice protocol"
[[ -f "$FORCED_CHOICE_INPUTS" ]] || \
    agentdojo_die "missing immutable forced-choice inputs"
command -v nvidia-smi >/dev/null 2>&1 || \
    agentdojo_die "the scheduler allocation exposes no nvidia-smi"
nvidia-smi -L >/dev/null 2>&1 || \
    agentdojo_die "the scheduler allocation exposes no visible GPU"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false SILENTTWIN_OFFLINE=1
export AGENTDOJO_ALLOW_API_FALLBACK=0 AGENTDOJO_ALLOW_MOCK_FALLBACK=0

exec "$PYTHON_BIN" -m silenttwin.agentdojo.forced_choice_readout run \
    --protocol "$FORCED_CHOICE_PROTOCOL" \
    --inputs "$FORCED_CHOICE_INPUTS" \
    --dependency-lock "$AGENTDOJO_DEPENDENCY_LOCK" \
    --checkpoint "$AGENTDOJO_ATTACKER_CHECKPOINT" \
    --model-cache "$AGENTDOJO_MODEL_CACHE" \
    --output-dir "$FORCED_CHOICE_OUTPUT" \
    --device "${ATTACKER_DEVICE:-cuda:0}"
