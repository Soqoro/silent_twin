#!/bin/bash
# Train/development-only guard-pair mining. Test observations are never accepted.
set -euo pipefail
if [[ -n "${AGENTDOJO_REPO_ROOT:-}" ]]; then
    script_dir="$AGENTDOJO_REPO_ROOT/experiments/silenttwin"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"
agentdojo_init pair_mining PAIR_MINING controlled
TRAIN_OBSERVATIONS="${TRAIN_OBSERVATIONS:-}"
DEVELOPMENT_OBSERVATIONS="${DEVELOPMENT_OBSERVATIONS:-}"
TRAIN_OBSERVATION_MANIFEST="${TRAIN_OBSERVATION_MANIFEST:-}"
DEVELOPMENT_OBSERVATION_MANIFEST="${DEVELOPMENT_OBSERVATION_MANIFEST:-}"
PAIR_MINING_ACTION="${PAIR_MINING_ACTION:-reduce}"
OBSERVATION_SPLIT="${OBSERVATION_SPLIT:-train}"
OBSERVATIONS_OUTPUT="${OBSERVATIONS_OUTPUT:-$OUT_ROOT/pair_mining/observations/$OBSERVATION_SPLIT.jsonl}"
OBSERVATION_MANIFEST_OUTPUT="${OBSERVATION_MANIFEST_OUTPUT:-$OUT_ROOT/pair_mining/observations/$OBSERVATION_SPLIT.manifest.json}"
case "$PAIR_MINING_ACTION" in
    observe|reduce) ;;
    *) agentdojo_die "PAIR_MINING_ACTION must be observe or reduce" ;;
esac
case "$OBSERVATION_SPLIT" in
    train|development) ;;
    *) agentdojo_die "OBSERVATION_SPLIT must be train or development" ;;
esac

case "$AGENTDOJO_STAGE" in
    grid)
        printf 'environment_backend=agentdojo\n'
        printf 'dataset_splits=train,development\n'
        printf 'catalog=%s\n' "$AGENTDOJO_CATALOG"
        printf 'splits=%s\n' "$AGENTDOJO_SPLITS"
        printf 'strategy_catalog=%s\n' "$AGENTDOJO_STRATEGY_CATALOG"
        printf 'pair_registry_output=%s\n' "$AGENTDOJO_PAIR_REGISTRY"
        printf '%s\n' 'PAIR_MINING_ACTION=observe generates one split with the pinned live runtime'
        printf '%s\n' 'PAIR_MINING_ACTION=reduce is CPU-only and freezes train/development evidence'
        printf 'observation_command=%s\n' "python -m silenttwin.agentdojo.runner generate-pair-observations --dataset-split $OBSERVATION_SPLIT"
        printf '%s\n' 'reduce_inputs=TRAIN_OBSERVATIONS TRAIN_OBSERVATION_MANIFEST DEVELOPMENT_OBSERVATIONS DEVELOPMENT_OBSERVATION_MANIFEST'
        ;;
    run)
        agentdojo_require_scheduler_job "pair-mining run"
        agentdojo_reject_ephemeral_runtime_paths
        requires_gpu="${AGENTDOJO_REQUIRES_GPU:-}"
        if [[ -z "$requires_gpu" ]]; then
            if [[ "$PAIR_MINING_ACTION" == observe ]]; then
                requires_gpu=1
            else
                requires_gpu=0
            fi
        fi
        [[ "$requires_gpu" == 0 || "$requires_gpu" == 1 ]] || \
            agentdojo_die "AGENTDOJO_REQUIRES_GPU must be 0 or 1"
        if [[ "$PAIR_MINING_ACTION" == observe ]]; then
            [[ -n "${AGENTDOJO_MODEL_CACHE:-}" ]] || \
                agentdojo_die "pair-observation learned monitors require AGENTDOJO_MODEL_CACHE"
            if [[ -z "${MONITOR_DEVICE:-}" ]]; then
                if [[ "$requires_gpu" == 1 ]]; then
                    MONITOR_DEVICE=cuda
                else
                    MONITOR_DEVICE=cpu
                fi
            fi
            if [[ "$requires_gpu" == 0 && "$MONITOR_DEVICE" == cuda* ]]; then
                agentdojo_die "MONITOR_DEVICE requests CUDA while AGENTDOJO_REQUIRES_GPU=0"
            fi
            export MONITOR_DEVICE
        fi
        AGENTDOJO_REQUIRES_GPU="$requires_gpu"
        export AGENTDOJO_REQUIRES_GPU
        if [[ "$requires_gpu" == 1 ]]; then
            command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || \
                agentdojo_die "pair-mining monitor inference requires a visible allocated GPU"
        fi
        agentdojo_offline_environment
        agentdojo_activate_and_require_python311
        if [[ "$PAIR_MINING_ACTION" == observe ]]; then
            mkdir -p -- "$(dirname -- "$OBSERVATIONS_OUTPUT")" \
                "$(dirname -- "$OBSERVATION_MANIFEST_OUTPUT")"
            exec "$PYTHON_BIN" -m silenttwin.agentdojo.runner generate-pair-observations \
                --catalog "$AGENTDOJO_CATALOG" \
                --splits "$AGENTDOJO_SPLITS" \
                --strategy-catalog "$AGENTDOJO_STRATEGY_CATALOG" \
                --dataset-split "$OBSERVATION_SPLIT" \
                --observations-output "$OBSERVATIONS_OUTPUT" \
                --observation-manifest-output "$OBSERVATION_MANIFEST_OUTPUT"
        fi
        [[ -n "$TRAIN_OBSERVATIONS" && -n "$DEVELOPMENT_OBSERVATIONS" && \
            -n "$TRAIN_OBSERVATION_MANIFEST" && -n "$DEVELOPMENT_OBSERVATION_MANIFEST" ]] || \
            agentdojo_die "reduce requires both observation JSONL files and manifests"
        [[ "$TRAIN_OBSERVATIONS" != *test* && "$DEVELOPMENT_OBSERVATIONS" != *test* ]] || \
            agentdojo_die "test observations are forbidden during pair mining"
        [[ -f "$TRAIN_OBSERVATIONS" && -f "$DEVELOPMENT_OBSERVATIONS" && \
            -f "$TRAIN_OBSERVATION_MANIFEST" && -f "$DEVELOPMENT_OBSERVATION_MANIFEST" ]] || \
            agentdojo_die "pair-mining observations or manifests are not readable"
        exec "$PYTHON_BIN" -m silenttwin.agentdojo.runner mine-pairs \
            --catalog "$AGENTDOJO_CATALOG" \
            --splits "$AGENTDOJO_SPLITS" \
            --strategy-catalog "$AGENTDOJO_STRATEGY_CATALOG" \
            --train-observations "$TRAIN_OBSERVATIONS" \
            --development-observations "$DEVELOPMENT_OBSERVATIONS" \
            --train-observation-manifest "$TRAIN_OBSERVATION_MANIFEST" \
            --development-observation-manifest "$DEVELOPMENT_OBSERVATION_MANIFEST" \
            --pair-registry-output "$AGENTDOJO_PAIR_REGISTRY"
        ;;
    aggregate)
        [[ -f "$AGENTDOJO_STRATEGY_CATALOG" ]] || \
            agentdojo_die "candidate-strategy catalog is absent"
        [[ -f "$AGENTDOJO_PAIR_REGISTRY" ]] || agentdojo_die "pair registry is absent"
        read -r strategy_digest _ < <(sha256sum -- "$AGENTDOJO_STRATEGY_CATALOG")
        read -r pair_digest _ < <(sha256sum -- "$AGENTDOJO_PAIR_REGISTRY")
        printf 'strategy_catalog_file_sha256=%s\n' "$strategy_digest"
        printf 'pair_registry_file_sha256=%s\n' "$pair_digest"
        ;;
esac
