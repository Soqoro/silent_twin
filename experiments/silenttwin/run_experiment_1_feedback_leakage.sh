#!/bin/bash
#SBATCH --job-name=st-e1-leakage
#SBATCH --output=logs/st-e1-%A_%a.out
#SBATCH --error=logs/st-e1-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Experiment 1 asks whether public enforcement feedback reveals private state.
# Inspect the exact default grid:
#   E1_STAGE=grid bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
# Inspect Pilot A or the batched Tier-1 Pilot B grid:
#   PILOT_PRESET=pilot_a E1_STAGE=grid bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
#   PILOT_PRESET=pilot_b E1_STAGE=grid bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
# Never copy a documented array range after changing inputs; inspect it again.

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

append_model_grid_flags() {
    local command_name="$1"
    local -n command_ref="$command_name"
    [[ -z "$MODEL_ID" ]] || command_ref+=(--model-id "$MODEL_ID")
    [[ -z "$MODEL_REVISION" ]] || command_ref+=(--model-revision "$MODEL_REVISION")
    [[ -z "$MODEL_CACHE_DIR" ]] || command_ref+=(--model-cache-dir "$MODEL_CACHE_DIR")
    [[ -z "$DTYPE" ]] || command_ref+=(--dtype "$DTYPE")
    [[ -z "$MAX_NEW_TOKENS" ]] || command_ref+=(--max-new-tokens "$MAX_NEW_TOKENS")
    [[ -z "$TEMPERATURE" ]] || command_ref+=(--temperature "$TEMPERATURE")
    [[ -z "$TOP_P" ]] || command_ref+=(--top-p "$TOP_P")
    [[ -z "$BATCH_SIZE" ]] || command_ref+=(--batch-size "$BATCH_SIZE")
    [[ -z "$DATASET_SPLIT" ]] || command_ref+=(--dataset-split "$DATASET_SPLIT")
    [[ -z "$DATASET_REVISION" ]] || command_ref+=(--dataset-revision "$DATASET_REVISION")
    [[ -z "$SAMPLE_SIZE_FREEZE" ]] || command_ref+=(--sample-size-freeze "$SAMPLE_SIZE_FREEZE")
}

build_grid_arguments() {
    grid_arguments=(--experiment e1)
    if [[ -n "$PILOT_PRESET" ]]; then
        grid_arguments+=(--preset "$PILOT_PRESET")
        append_model_grid_flags grid_arguments
        if ((${#decoding_seeds[@]} > 0)); then
            st_append_repeated grid_arguments --decoding-seed decoding_seeds
        fi
        return
    fi

    st_append_repeated grid_arguments --tier tiers
    st_append_repeated grid_arguments --world-suite world_suites
    st_append_repeated grid_arguments --runtime runtimes
    st_append_repeated grid_arguments --attacker attackers
    st_append_repeated grid_arguments --query-budget query_budgets
    st_append_repeated grid_arguments --feedback-source feedback_sources
    if ((${#feedback_source_query_budgets[@]} > 0)); then
        st_append_repeated grid_arguments --feedback-source-query-budgets \
            feedback_source_query_budgets
    fi
    st_append_repeated grid_arguments --seed seeds
    st_append_repeated grid_arguments --pair-family pair_families
    grid_arguments+=(
        --num-samples "$NUM_SAMPLES"
        --cells-per-task "$CELLS_PER_TASK"
        --dataset-split "$DATASET_SPLIT"
        --dataset-revision "$DATASET_REVISION"
    )
    if [[ -n "$EPISODES_PER_SHARD" ]]; then
        grid_arguments+=(--episodes-per-shard "$EPISODES_PER_SHARD")
    fi
    if ((${#decoding_seeds[@]} > 0)); then
        st_append_repeated grid_arguments --decoding-seed decoding_seeds
    fi
    append_model_grid_flags grid_arguments
}

append_config_run_flags() {
    local command_name="$1"
    local cell_name="$2"
    local -n command_ref="$command_name"
    local -n cell_ref="$cell_name"
    command_ref+=(
        --sample-start "${cell_ref[sample_start]}"
        --pair-family "${cell_ref[pair_family]}"
        --dataset-split "${cell_ref[dataset_split]}"
        --dataset-revision "${cell_ref[dataset_revision]}"
        --analysis-revision "${cell_ref[analysis_revision]}"
        --feedback-source "${cell_ref[feedback_source]}"
        --confidence-threshold "${cell_ref[confidence_threshold]}"
        --dtype "${cell_ref[dtype]}"
        --max-new-tokens "${cell_ref[max_new_tokens]}"
        --temperature "${cell_ref[temperature]}"
        --top-p "${cell_ref[top_p]}"
        --decoding-seed "${cell_ref[decoding_seed]}"
        --batch-size "${cell_ref[batch_size]}"
        --grid-hash "${cell_ref[grid_hash]}"
        --grid-task-id "${cell_ref[grid_task_id]}"
        --shard-id "${cell_ref[shard_id]}"
        --expected-configuration-hash "${cell_ref[configuration_hash]}"
    )
    [[ -z "${cell_ref[template_id]:-}" ]] || command_ref+=(--template-id "${cell_ref[template_id]}")
    [[ -z "${cell_ref[model_id]:-}" ]] || command_ref+=(--model-id "${cell_ref[model_id]}")
    [[ -z "${cell_ref[model_revision]:-}" ]] || command_ref+=(--model-revision "${cell_ref[model_revision]}")
    if [[ -n "${cell_ref[model_cache_dir]:-}" ]]; then
        command_ref+=(--model-cache-dir "${cell_ref[model_cache_dir]}")
    elif [[ -n "$MODEL_CACHE_DIR" ]]; then
        command_ref+=(--model-cache-dir "$MODEL_CACHE_DIR")
    fi
    [[ -z "${cell_ref[pilot_id]:-}" ]] || command_ref+=(--pilot-id "${cell_ref[pilot_id]}")
    if [[ "${cell_ref[dataset_split]}" == test ]]; then
        [[ -n "$SAMPLE_SIZE_FREEZE" ]] || st_die "SAMPLE_SIZE_FREEZE is required for held-out execution"
        command_ref+=(--sample-size-freeze "$SAMPLE_SIZE_FREEZE")
    fi
}

run_selected_cell() {
    local cell_name="$1"
    local -n cell_ref="$cell_name"
    # The complete scientific hash includes decoding seed, pair/template,
    # split, feedback source, model identity, and any held-out freeze. Legacy
    # human paths omitted several of those factors and could collide.
    local run_dir="$OUT_ROOT/e1/pilot=${cell_ref[pilot_id]:-custom}/config=${cell_ref[configuration_hash]}"
    local run_command=(
        "$PYTHON_BIN" -m silenttwin.cli run
        --experiment e1
        --tier "${cell_ref[tier]}"
        --world-suite "${cell_ref[world_suite]}"
        --runtime "${cell_ref[runtime]}"
        --attacker "${cell_ref[attacker]}"
        --query-budget "${cell_ref[query_budget]}"
        --seed "${cell_ref[seed]}"
        --num-samples "${cell_ref[num_samples]}"
        --output-dir "$run_dir"
    )
    append_config_run_flags run_command "$cell_name"
    st_append_common_run_flags run_command
    printf 'selected_task_id=%s batch_offset=%s configuration_hash=%s shard_id=%s output_dir=%s\n' \
        "${cell_ref[grid_task_id]}" "${cell_ref[batch_offset]}" \
        "${cell_ref[configuration_hash]}" "${cell_ref[shard_id]}" "$run_dir"
    st_execute "task_${cell_ref[grid_task_id]}_cell_${cell_ref[batch_offset]}" run_command
}

main() {
    TIERS="${TIERS:-tier1}"
    WORLD_SUITES="${WORLD_SUITES:-email files calendar payments}"
    RUNTIMES="${RUNTIMES:-detailed_refusal binary_denial generic_failure visible_settlement opaque_termination randomized_feedback arm_style silenttwin}"
    ATTACKERS="${ATTACKERS:-bayesian black_box mock_llm}"
    QUERY_BUDGETS="${QUERY_BUDGETS:-0 4 16 32}"
    FEEDBACK_SOURCES="${FEEDBACK_SOURCES:-genuine}"
    FEEDBACK_SOURCE_QUERY_BUDGETS="${FEEDBACK_SOURCE_QUERY_BUDGETS:-}"
    SEEDS="${SEEDS:-42}"
    PAIR_FAMILIES="${PAIR_FAMILIES:-monitor_blind_spot}"
    DECODING_SEEDS="${DECODING_SEEDS:-${DECODING_SEED:-}}"
    DATASET_SPLIT="${DATASET_SPLIT:-development}"
    DATASET_REVISION="${DATASET_REVISION:-}"
    CELLS_PER_TASK="${CELLS_PER_TASK:-1}"
    EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-}"
    PILOT_PRESET="${PILOT_PRESET:-}"
    MODEL_ID="${MODEL_ID:-}"
    MODEL_REVISION="${MODEL_REVISION:-}"
    MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
    DTYPE="${DTYPE:-}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
    TEMPERATURE="${TEMPERATURE:-}"
    TOP_P="${TOP_P:-}"
    BATCH_SIZE="${BATCH_SIZE:-}"
    SAMPLE_SIZE_FREEZE="${SAMPLE_SIZE_FREEZE:-}"

    st_prepare e1 "${E1_STAGE:-run}"
    if [[ -z "$PILOT_PRESET" && -z "$DATASET_REVISION" ]]; then
        DATASET_REVISION="silenttwin-tier1-v1"
    fi
    st_split_words TIERS "$TIERS" tiers
    st_split_words WORLD_SUITES "$WORLD_SUITES" world_suites
    st_split_words RUNTIMES "$RUNTIMES" runtimes
    st_split_words ATTACKERS "$ATTACKERS" attackers
    st_split_words QUERY_BUDGETS "$QUERY_BUDGETS" query_budgets
    st_split_words FEEDBACK_SOURCES "$FEEDBACK_SOURCES" feedback_sources
    feedback_source_query_budgets=()
    if [[ -n "$FEEDBACK_SOURCE_QUERY_BUDGETS" ]]; then
        st_split_words FEEDBACK_SOURCE_QUERY_BUDGETS \
            "$FEEDBACK_SOURCE_QUERY_BUDGETS" feedback_source_query_budgets
    fi
    st_split_words SEEDS "$SEEDS" seeds
    st_split_words PAIR_FAMILIES "$PAIR_FAMILIES" pair_families
    decoding_seeds=()
    if [[ -n "$DECODING_SEEDS" ]]; then
        st_split_words DECODING_SEEDS "$DECODING_SEEDS" decoding_seeds
    fi

    st_validate_tokens TIERS "${tiers[@]}"
    st_validate_tokens WORLD_SUITES "${world_suites[@]}"
    st_validate_tokens RUNTIMES "${runtimes[@]}"
    st_validate_tokens ATTACKERS "${attackers[@]}"
    st_validate_uints QUERY_BUDGETS "${query_budgets[@]}"
    st_validate_tokens FEEDBACK_SOURCES "${feedback_sources[@]}"
    st_validate_uints SEEDS "${seeds[@]}"
    st_validate_tokens PAIR_FAMILIES "${pair_families[@]}"
    ((${#decoding_seeds[@]} == 0)) || st_validate_uints DECODING_SEEDS "${decoding_seeds[@]}"
    st_validate_choices TIERS 'tier1 tier2' "${tiers[@]}"
    st_validate_choices WORLD_SUITES 'email files calendar payments' "${world_suites[@]}"
    st_validate_choices RUNTIMES 'detailed_refusal binary_denial generic_failure visible_settlement opaque_termination randomized_feedback arm_style silenttwin silenttwin_timing_leak silenttwin_identifier_leak silenttwin_decision_dependent_twin_mismatch' "${runtimes[@]}"
    st_validate_choices ATTACKERS 'bayesian black_box mock_llm random majority llm' "${attackers[@]}"
    st_validate_choices FEEDBACK_SOURCES 'genuine shuffled constant' "${feedback_sources[@]}"
    st_validate_choices PAIR_FAMILIES 'authorization monitor_blind_spot blind_spot' "${pair_families[@]}"
    [[ "$CELLS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || st_die "CELLS_PER_TASK must be positive"
    [[ -z "$EPISODES_PER_SHARD" || "$EPISODES_PER_SHARD" =~ ^[1-9][0-9]*$ ]] || \
        st_die "EPISODES_PER_SHARD must be empty or positive"
    [[ -z "$PILOT_PRESET" ]] || st_validate_choices PILOT_PRESET \
        'pilot_a pilot_b pilot_c pilot_d' "$PILOT_PRESET"
    [[ -z "$SAMPLE_SIZE_FREEZE" || -f "$SAMPLE_SIZE_FREEZE" ]] || \
        st_die "SAMPLE_SIZE_FREEZE is not a readable file: $SAMPLE_SIZE_FREEZE"

    local -a grid_arguments
    build_grid_arguments
    local total_tasks total_members expected_grid_hash
    total_tasks="$(st_grid_count grid_arguments)"
    total_members="$(st_grid_member_count grid_arguments)"
    expected_grid_hash="$(st_grid_hash grid_arguments)"
    case "$ST_STAGE" in
        grid)
            st_grid_print grid_arguments
            ;;
        aggregate)
            local expected_manifest="$OUT_ROOT/e1/aggregate/grid_manifest.jsonl"
            st_grid_write_manifest grid_arguments "$expected_manifest"
            printf 'total_tasks=%s\ntotal_configurations=%s\nvalid_array_range=0-%s\ngrid_hash=%s\n' \
                "$total_tasks" "$total_members" "$((total_tasks - 1))" "$expected_grid_hash"
            st_aggregate "$total_members" "$expected_manifest" "$expected_grid_hash"
            ;;
        run)
            if [[ "${SILENTTWIN_REQUIRE_SLURM:-0}" == 1 && -z "${SLURM_JOB_ID:-}" ]]; then
                st_die "Tier-2 run stage must execute inside an authorized SLURM job; use grid locally"
            fi
            st_select_index "$total_tasks"
            local -a selected_values
            st_grid_select grid_arguments "$ST_TASK_INDEX" selected_values
            printf 'total_tasks=%s\ntotal_configurations=%s\nvalid_array_range=0-%s\ngrid_hash=%s\n' \
                "$total_tasks" "$total_members" "$((total_tasks - 1))" "$expected_grid_hash"
            st_for_each_selected_cell selected_values run_selected_cell
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
