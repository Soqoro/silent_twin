#!/bin/bash
# Run the complete, explicit Pilot A E1/E2 grid locally on CPU, then aggregate.
# This deterministic mock pilot never submits a job, loads a model, or uses GPU.
#
#   bash experiments/silenttwin/run_pilot_a_engineering_smoke.sh
#   PILOT_STAGE=grid bash experiments/silenttwin/run_pilot_a_engineering_smoke.sh
#   PILOT_STAGE=run OUT_ROOT=outputs/silenttwin-pilot-a bash experiments/silenttwin/run_pilot_a_engineering_smoke.sh
#   PILOT_STAGE=aggregate OUT_ROOT=outputs/silenttwin-pilot-a bash experiments/silenttwin/run_pilot_a_engineering_smoke.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
cd -- "$repo_root"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-outputs/silenttwin-pilot-a}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PILOT_STAGE="${PILOT_STAGE:-all}"

case "$PILOT_STAGE" in
    grid|run|aggregate|all) ;;
    *) printf '%s\n' 'silenttwin: error: PILOT_STAGE must be grid, run, aggregate, or all' >&2; exit 2 ;;
esac

run_experiment_stage() {
    local experiment="$1"
    local stage="$2"
    local script stage_variable
    case "$experiment" in
        e1)
            script="$script_dir/run_experiment_1_feedback_leakage.sh"
            stage_variable=E1_STAGE
            ;;
        e2)
            script="$script_dir/run_experiment_2_feedback_assisted_bypass.sh"
            stage_variable=E2_STAGE
            ;;
        *) printf 'silenttwin: error: unsupported Pilot A experiment: %s\n' "$experiment" >&2; exit 2 ;;
    esac

    if [[ "$stage" == grid || "$stage" == aggregate ]]; then
        local command=(
            env "PYTHON_BIN=$PYTHON_BIN" "OUT_ROOT=$OUT_ROOT" "OVERWRITE=$OVERWRITE"
            "EXTRA_ARGS=$EXTRA_ARGS" PILOT_PRESET=pilot_a "$stage_variable=$stage"
            bash "$script"
        )
        printf 'command='
        printf '%q ' "${command[@]}"
        printf '\n'
        "${command[@]}"
        return
    fi

    local total_tasks
    total_tasks="$(
        PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" -m silenttwin.experiments.grid count \
            --experiment "$experiment" --preset pilot_a
    )"
    local task_id
    for ((task_id = 0; task_id < total_tasks; task_id++)); do
        local command=(
            env "PYTHON_BIN=$PYTHON_BIN" "OUT_ROOT=$OUT_ROOT" "OVERWRITE=$OVERWRITE"
            "EXTRA_ARGS=$EXTRA_ARGS" PILOT_PRESET=pilot_a "$stage_variable=run"
            "SLURM_ARRAY_TASK_ID=$task_id" bash "$script"
        )
        printf 'command='
        printf '%q ' "${command[@]}"
        printf '\n'
        "${command[@]}"
    done
}

if [[ "$PILOT_STAGE" == grid ]]; then
    run_experiment_stage e1 grid
    run_experiment_stage e2 grid
    exit 0
fi
if [[ "$PILOT_STAGE" == run || "$PILOT_STAGE" == all ]]; then
    run_experiment_stage e1 run
    run_experiment_stage e2 run
fi
if [[ "$PILOT_STAGE" == aggregate || "$PILOT_STAGE" == all ]]; then
    run_experiment_stage e1 aggregate
    run_experiment_stage e2 aggregate
fi
