#!/bin/bash
# Run one two-sample deterministic mock configuration for each experiment, then
# aggregate those pilot outputs. This script never submits scheduler jobs.
#
#   bash experiments/silenttwin/run_all_pilots.sh
#   PILOT_STAGE=run OUT_ROOT=outputs/silenttwin-pilot bash experiments/silenttwin/run_all_pilots.sh
#   PILOT_STAGE=aggregate OUT_ROOT=outputs/silenttwin-pilot bash experiments/silenttwin/run_all_pilots.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
cd -- "$repo_root"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-outputs/silenttwin-pilot}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
SEEDS="${SEEDS:-42}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PILOT_STAGE="${PILOT_STAGE:-all}"

case "$PILOT_STAGE" in
    run|aggregate|all) ;;
    *) printf 'silenttwin: error: PILOT_STAGE must be run, aggregate, or all\n' >&2; exit 2 ;;
esac
[[ "$SEEDS" =~ ^(0|[1-9][0-9]*)$ ]] || {
    printf 'silenttwin: error: run_all_pilots.sh accepts exactly one non-negative seed\n' >&2
    exit 2
}

print_command() {
    local argument
    printf 'command='
    for argument in "$@"; do
        printf '%q ' "$argument"
    done
    printf '\n'
}

run_pilot() {
    local script_name="$1"
    local stage_variable="$2"
    shift 2
    local command=(
        env
        "PYTHON_BIN=$PYTHON_BIN"
        "OUT_ROOT=$OUT_ROOT"
        "NUM_SAMPLES=$NUM_SAMPLES"
        "SEEDS=$SEEDS"
        "OVERWRITE=$OVERWRITE"
        "EXTRA_ARGS=$EXTRA_ARGS"
        "$stage_variable=run"
        SLURM_ARRAY_TASK_ID=0
        "$@"
        bash "$script_dir/$script_name"
    )
    print_command "${command[@]}"
    "${command[@]}"
}

aggregate_pilot() {
    local script_name="$1"
    local stage_variable="$2"
    shift 2
    local command=(
        env
        "PYTHON_BIN=$PYTHON_BIN"
        "OUT_ROOT=$OUT_ROOT"
        "EXTRA_ARGS=$EXTRA_ARGS"
        "$stage_variable=aggregate"
        "$@"
        bash "$script_dir/$script_name"
    )
    print_command "${command[@]}"
    "${command[@]}"
}

if [[ "$PILOT_STAGE" == run || "$PILOT_STAGE" == all ]]; then
    run_pilot run_experiment_1_feedback_leakage.sh E1_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
        ATTACKERS=mock_llm QUERY_BUDGETS=4
    run_pilot run_experiment_2_feedback_assisted_bypass.sh E2_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
        ATTACKERS=mock_llm QUERY_BUDGETS=4 E2_CONDITIONS=adaptive
    run_pilot run_experiment_3_channel_closure.sh E3_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
        ATTACKERS=mock_llm QUERY_BUDGETS=4
    run_pilot run_experiment_4_useful_work.sh E4_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
        ATTACKERS=mock_llm QUERY_BUDGETS=0 E4_WORKFLOWS=independent
    run_pilot run_experiment_5_assumption_ablations.sh E5_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
        ATTACKERS=mock_llm QUERY_BUDGETS=4 E5_ABLATIONS=timing_leak
fi

if [[ "$PILOT_STAGE" == aggregate || "$PILOT_STAGE" == all ]]; then
    aggregate_pilot run_experiment_1_feedback_leakage.sh E1_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
        ATTACKERS=mock_llm QUERY_BUDGETS=4 "SEEDS=$SEEDS"
    aggregate_pilot run_experiment_2_feedback_assisted_bypass.sh E2_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
        ATTACKERS=mock_llm QUERY_BUDGETS=4 E2_CONDITIONS=adaptive "SEEDS=$SEEDS"
    aggregate_pilot run_experiment_3_channel_closure.sh E3_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
        ATTACKERS=mock_llm QUERY_BUDGETS=4 "SEEDS=$SEEDS"
    aggregate_pilot run_experiment_4_useful_work.sh E4_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
        ATTACKERS=mock_llm QUERY_BUDGETS=0 E4_WORKFLOWS=independent "SEEDS=$SEEDS"
    aggregate_pilot run_experiment_5_assumption_ablations.sh E5_STAGE \
        TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
        ATTACKERS=mock_llm QUERY_BUDGETS=4 E5_ABLATIONS=timing_leak "SEEDS=$SEEDS"
fi
