#!/bin/bash
#SBATCH --job-name=st-e4-utility
#SBATCH --output=logs/st-e4-%A_%a.out
#SBATCH --error=logs/st-e4-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Experiment 4 measures safe-work salvage, dependency closure, and atomic commit.
# Default grid: E4_STAGE=grid bash experiments/silenttwin/run_experiment_4_useful_work.sh
# Tiny local smoke:
#   TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin ATTACKERS=mock_llm QUERY_BUDGETS=0 E4_WORKFLOWS=independent SEEDS=42 NUM_SAMPLES=2 SLURM_ARRAY_TASK_ID=0 bash experiments/silenttwin/run_experiment_4_useful_work.sh
# Default SLURM array (36 jobs; create logs before submission):
#   mkdir -p logs
#   sbatch --array=0-35 experiments/silenttwin/run_experiment_4_useful_work.sh
# Aggregate: E4_STAGE=aggregate bash experiments/silenttwin/run_experiment_4_useful_work.sh

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

total_jobs() {
    printf '%s\n' "$((${#tiers[@]} * ${#world_suites[@]} * ${#runtimes[@]} * ${#attackers[@]} * ${#query_budgets[@]} * ${#workflows[@]} * ${#seeds[@]}))"
}

select_config() {
    local requested_index="$1"
    local cursor=0
    local tier world runtime attacker budget workflow seed
    for tier in "${tiers[@]}"; do
        for world in "${world_suites[@]}"; do
            for runtime in "${runtimes[@]}"; do
                for attacker in "${attackers[@]}"; do
                    for budget in "${query_budgets[@]}"; do
                        for workflow in "${workflows[@]}"; do
                            for seed in "${seeds[@]}"; do
                                if ((cursor == requested_index)); then
                                    SELECTED_TIER="$tier"
                                    SELECTED_WORLD="$world"
                                    SELECTED_RUNTIME="$runtime"
                                    SELECTED_ATTACKER="$attacker"
                                    SELECTED_BUDGET="$budget"
                                    SELECTED_WORKFLOW="$workflow"
                                    SELECTED_SEED="$seed"
                                    return 0
                                fi
                                cursor=$((cursor + 1))
                            done
                        done
                    done
                done
            done
        done
    done
    st_die "internal grid selection failure for index $requested_index"
}

print_grid() {
    local total
    local task_id
    total="$(total_jobs)"
    st_print_grid_header "$total" 'tier>world_suite>runtime>attacker>query_budget>workflow>seed (seed varies fastest)'
    for ((task_id = 0; task_id < total; task_id++)); do
        select_config "$task_id"
        printf 'task_id=%s tier=%s world_suite=%s runtime=%s attacker=%s query_budget=%s workflow=%s seed=%s\n' \
            "$task_id" "$SELECTED_TIER" "$SELECTED_WORLD" "$SELECTED_RUNTIME" \
            "$SELECTED_ATTACKER" "$SELECTED_BUDGET" "$SELECTED_WORKFLOW" "$SELECTED_SEED"
    done
}

main() {
    TIERS="${TIERS:-tier1}"
    WORLD_SUITES="${WORLD_SUITES:-email files calendar payments}"
    RUNTIMES="${RUNTIMES:-opaque_termination visible_settlement silenttwin}"
    ATTACKERS="${ATTACKERS:-mock_llm}"
    QUERY_BUDGETS="${QUERY_BUDGETS:-0}"
    E4_WORKFLOWS="${E4_WORKFLOWS:-independent rejection_dependent atomic}"
    SEEDS="${SEEDS:-42}"

    st_prepare e4 "${E4_STAGE:-run}"
    st_split_words TIERS "$TIERS" tiers
    st_split_words WORLD_SUITES "$WORLD_SUITES" world_suites
    st_split_words RUNTIMES "$RUNTIMES" runtimes
    st_split_words ATTACKERS "$ATTACKERS" attackers
    st_split_words QUERY_BUDGETS "$QUERY_BUDGETS" query_budgets
    st_split_words E4_WORKFLOWS "$E4_WORKFLOWS" workflows
    st_split_words SEEDS "$SEEDS" seeds
    st_validate_tokens TIERS "${tiers[@]}"
    st_validate_tokens WORLD_SUITES "${world_suites[@]}"
    st_validate_tokens RUNTIMES "${runtimes[@]}"
    st_validate_tokens ATTACKERS "${attackers[@]}"
    st_validate_uints QUERY_BUDGETS "${query_budgets[@]}"
    st_validate_tokens E4_WORKFLOWS "${workflows[@]}"
    st_validate_uints SEEDS "${seeds[@]}"
    st_validate_choices TIERS 'tier1 tier2' "${tiers[@]}"
    st_validate_choices WORLD_SUITES 'email files calendar payments' "${world_suites[@]}"
    st_validate_choices RUNTIMES 'detailed_refusal binary_denial generic_failure visible_settlement opaque_termination randomized_feedback arm_style silenttwin silenttwin_timing_leak silenttwin_identifier_leak silenttwin_decision_dependent_twin_mismatch' "${runtimes[@]}"
    st_validate_choices ATTACKERS 'bayesian black_box mock_llm random' "${attackers[@]}"
    st_validate_choices E4_WORKFLOWS 'independent rejection_dependent atomic' "${workflows[@]}"

    local -a grid_arguments=(--experiment e4 --num-samples "$NUM_SAMPLES")
    st_append_repeated grid_arguments --tier tiers
    st_append_repeated grid_arguments --world-suite world_suites
    st_append_repeated grid_arguments --runtime runtimes
    st_append_repeated grid_arguments --attacker attackers
    st_append_repeated grid_arguments --query-budget query_budgets
    st_append_repeated grid_arguments --workflow workflows
    st_append_repeated grid_arguments --seed seeds
    local total total_members expected_grid_hash
    total="$(st_grid_count grid_arguments)"
    total_members="$(st_grid_member_count grid_arguments)"
    expected_grid_hash="$(st_grid_hash grid_arguments)"
    case "$ST_STAGE" in
        grid)
            st_grid_print grid_arguments
            ;;
        aggregate)
            local expected_manifest="$OUT_ROOT/e4/aggregate/grid_manifest.jsonl"
            st_grid_write_manifest grid_arguments "$expected_manifest"
            printf 'total_jobs=%s\nvalid_array_range=0-%s\ngrid_hash=%s\n' \
                "$total" "$((total - 1))" "$expected_grid_hash"
            st_aggregate "$total_members" "$expected_manifest" "$expected_grid_hash"
            ;;
        run)
            st_select_index "$total"
            select_config "$ST_TASK_INDEX"
            local -a selected_values
            local -A selected_cell
            st_grid_select grid_arguments "$ST_TASK_INDEX" selected_values
            st_parse_single_selected_cell selected_values selected_cell
            local run_dir="$OUT_ROOT/e4/tier=$SELECTED_TIER/world=$SELECTED_WORLD/runtime=$SELECTED_RUNTIME/attacker=$SELECTED_ATTACKER/q=$SELECTED_BUDGET/workflow=$SELECTED_WORKFLOW/seed=$SELECTED_SEED"
            local run_command=(
                "$PYTHON_BIN" -m silenttwin.cli run
                --experiment e4
                --tier "$SELECTED_TIER"
                --world-suite "$SELECTED_WORLD"
                --runtime "$SELECTED_RUNTIME"
                --attacker "$SELECTED_ATTACKER"
                --query-budget "$SELECTED_BUDGET"
                --seed "$SELECTED_SEED"
                --num-samples "$NUM_SAMPLES"
                --output-dir "$run_dir"
                --workflow "$SELECTED_WORKFLOW"
                --grid-hash "$expected_grid_hash"
                --grid-task-id "$ST_TASK_INDEX"
                --shard-id "${selected_cell[shard_id]}"
                --expected-configuration-hash "${selected_cell[configuration_hash]}"
            )
            st_append_common_run_flags run_command
            printf 'total_jobs=%s\nvalid_array_range=0-%s\ngrid_hash=%s\nselected_task_id=%s\nconfiguration_hash=%s\noutput_dir=%s\n' \
                "$total" "$((total - 1))" "$expected_grid_hash" "$ST_TASK_INDEX" \
                "${selected_cell[configuration_hash]}" "$run_dir"
            st_execute "task_$ST_TASK_INDEX" run_command
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
