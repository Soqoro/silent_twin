#!/bin/bash
#SBATCH --job-name=st-e2-tier2
#SBATCH --output=logs/st-e2-tier2-%A_%a.out
#SBATCH --error=logs/st-e2-tier2-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Prepared-only Tier-2 entrypoint. Supply the site-approved account, GPU
# partition, and exact one-GPU allocation flag to sbatch; none is guessed here.
#
# Safe local inspection (never loads the model):
#   MODEL_ID=/persistent/model MODEL_REVISION=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
#   MODEL_CACHE_DIR=/persistent/cache DATASET_REVISION=silenttwin-tier1-v1 \
#   PILOT_PRESET=pilot_d E2_STAGE=grid \
#   bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
#
# After substituting site-approved flags and the printed range:
#   mkdir -p logs
#   sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
#     --array=0-N%4 <EXACT_EXPORT_ARGUMENTS> \
#     experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PILOT_PRESET="${PILOT_PRESET:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
DATASET_REVISION="${DATASET_REVISION:-}"
DATASET_SPLIT="${DATASET_SPLIT:-development}"
SAMPLE_SIZE_FREEZE="${SAMPLE_SIZE_FREEZE:-}"
ENV_ACTIVATE="${ENV_ACTIVATE:-}"
stage="${E2_STAGE:-run}"

path_is_within() {
    local candidate root
    candidate="$(realpath -m -- "$1")" || return 1
    root="$(realpath -m -- "$2")" || return 1
    [[ "$candidate" == "$root" || "$candidate" == "$root/"* ]]
}

case "$PILOT_PRESET" in
    pilot_c|pilot_d)
        [[ "$DATASET_SPLIT" == development && -z "$SAMPLE_SIZE_FREEZE" ]] || {
            printf '%s\n' 'silenttwin: error: Pilot C/D are development-only and cannot consume a held-out freeze' >&2
            exit 2
        }
        ;;
    "")
        [[ "$DATASET_SPLIT" == test && -n "$SAMPLE_SIZE_FREEZE" ]] || {
            printf '%s\n' 'silenttwin: error: custom Tier-2 E2 requires DATASET_SPLIT=test and SAMPLE_SIZE_FREEZE' >&2
            exit 2
        }
        EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-20}"
        export EPISODES_PER_SHARD
        ;;
    *) printf '%s\n' 'silenttwin: error: Tier-2 E2 accepts pilot_c, pilot_d, or a frozen custom test grid' >&2; exit 2 ;;
esac
for required_name in MODEL_ID MODEL_REVISION MODEL_CACHE_DIR DATASET_REVISION; do
    [[ -n "${!required_name}" ]] || {
        printf 'silenttwin: error: %s is required for an exact Tier-2 grid\n' "$required_name" >&2
        exit 2
    }
done
if [[ -n "$ENV_ACTIVATE" ]]; then
    [[ -f "$ENV_ACTIVATE" ]] || {
        printf 'silenttwin: error: ENV_ACTIVATE is not a readable file: %s\n' "$ENV_ACTIVATE" >&2
        exit 2
    }
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$MODEL_CACHE_DIR/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [[ "$stage" == run ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || {
        printf '%s\n' 'silenttwin: error: Tier-2 model inference is forbidden outside an authorized SLURM job' >&2
        exit 2
    }
    if [[ -n "${SLURM_TMPDIR:-}" ]]; then
        for persistent_name in MODEL_CACHE_DIR HF_HOME HF_HUB_CACHE; do
            if path_is_within "${!persistent_name}" "$SLURM_TMPDIR"; then
                printf 'silenttwin: error: %s must be persistent, not inside SLURM_TMPDIR\n' \
                    "$persistent_name" >&2
                exit 2
            fi
        done
        if path_is_within "${OUT_ROOT:-outputs/silenttwin}" "$SLURM_TMPDIR"; then
            printf '%s\n' 'silenttwin: error: OUT_ROOT must persist outside SLURM_TMPDIR' >&2
            exit 2
        fi
    fi
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || {
        printf '%s\n' 'silenttwin: error: the authorized SLURM job does not expose a GPU' >&2
        exit 2
    }
    mkdir -p -- "$MODEL_CACHE_DIR"
fi

export PILOT_PRESET MODEL_ID MODEL_REVISION MODEL_CACHE_DIR DATASET_REVISION DATASET_SPLIT SAMPLE_SIZE_FREEZE
export TIERS=tier2 ATTACKERS=llm SILENTTWIN_REQUIRE_SLURM=1

exec bash "$script_dir/run_experiment_2_feedback_assisted_bypass.sh" "$@"
