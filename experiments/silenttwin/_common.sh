#!/bin/bash
# Shared, source-only support for the SilentTwin experiment entrypoints.

set -euo pipefail

st_die() {
    printf 'silenttwin: error: %s\n' "$*" >&2
    exit 2
}

st_split_words() {
    local variable_name="$1"
    local raw_value="$2"
    local destination_name="$3"
    local -n destination="$destination_name"

    destination=()
    if [[ -n "$raw_value" ]]; then
        read -r -a destination <<< "$raw_value"
    fi
    ((${#destination[@]} > 0)) || st_die "$variable_name must contain at least one value"
    local left_index right_index
    for ((left_index = 0; left_index < ${#destination[@]}; left_index++)); do
        for ((right_index = left_index + 1; right_index < ${#destination[@]}; right_index++)); do
            [[ "${destination[left_index]}" != "${destination[right_index]}" ]] || \
                st_die "$variable_name contains duplicate value '${destination[left_index]}'"
        done
    done
}

st_validate_tokens() {
    local variable_name="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" =~ ^[A-Za-z0-9_.-]+$ ]] || \
            st_die "$variable_name contains an unsafe value: $value"
    done
}

st_validate_choices() {
    local variable_name="$1"
    local allowed_values=" $2 "
    shift 2
    local value
    for value in "$@"; do
        [[ "$allowed_values" == *" $value "* ]] || \
            st_die "$variable_name contains unsupported value '$value'; allowed: ${allowed_values:1:${#allowed_values}-2}"
    done
}

st_validate_uints() {
    local variable_name="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || \
            st_die "$variable_name must contain non-negative base-10 integers (got: $value)"
    done
}

st_prepare() {
    ST_EXPERIMENT="$1"
    ST_STAGE="$2"
    ST_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")/../.." && pwd -P)"

    case "$ST_STAGE" in
        grid|run|aggregate) ;;
        *) st_die "stage must be one of: grid, run, aggregate (got: $ST_STAGE)" ;;
    esac

    PYTHON_BIN="${PYTHON_BIN:-python3}"
    OUT_ROOT="${OUT_ROOT:-outputs/silenttwin}"
    NUM_SAMPLES="${NUM_SAMPLES:--1}"
    OVERWRITE="${OVERWRITE:-0}"
    EXTRA_ARGS="${EXTRA_ARGS:-}"

    [[ -n "$PYTHON_BIN" ]] || st_die "PYTHON_BIN must not be empty"
    [[ -n "$OUT_ROOT" ]] || st_die "OUT_ROOT must not be empty"
    [[ "$NUM_SAMPLES" =~ ^(-1|[1-9][0-9]*)$ ]] || \
        st_die "NUM_SAMPLES must be -1 or a positive base-10 integer"
    [[ "$OVERWRITE" == 0 || "$OVERWRITE" == 1 ]] || \
        st_die "OVERWRITE must be 0 or 1"

    ST_EXTRA_ARGV=()
    if [[ -n "$EXTRA_ARGS" ]]; then
        read -r -a ST_EXTRA_ARGV <<< "$EXTRA_ARGS"
    fi

    cd -- "$ST_REPO_ROOT"
    mkdir -p -- "$ST_REPO_ROOT/logs"
    if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="$ST_REPO_ROOT/src:$PYTHONPATH"
    else
        export PYTHONPATH="$ST_REPO_ROOT/src"
    fi
}

st_print_grid_header() {
    local total="$1"
    local ordering="$2"
    printf 'experiment=%s\n' "$ST_EXPERIMENT"
    printf 'total_jobs=%s\n' "$total"
    printf 'valid_array_range=0-%s\n' "$((total - 1))"
    printf 'ordering=%s\n' "$ordering"
}

st_select_index() {
    local total="$1"
    local raw_index="${SLURM_ARRAY_TASK_ID:-}"

    [[ -n "$raw_index" ]] || \
        st_die "SLURM_ARRAY_TASK_ID is required for the run stage; inspect the grid first"
    [[ "$raw_index" =~ ^(0|[1-9][0-9]*)$ ]] || \
        st_die "SLURM_ARRAY_TASK_ID must be a non-negative base-10 integer"
    ST_TASK_INDEX=$((10#$raw_index))
    ((ST_TASK_INDEX < total)) || \
        st_die "SLURM_ARRAY_TASK_ID=$raw_index is out of range; valid range is 0-$((total - 1))"
}

st_print_command() {
    local argument
    printf 'command='
    for argument in "$@"; do
        printf '%q ' "$argument"
    done
    printf '\n'
}

st_execute() {
    local task_label="$1"
    local command_name="$2"
    local -n command_ref="$command_name"
    local scratch_base
    local st_scratch_dir
    local st_nvidia_status

    command -v -- "$PYTHON_BIN" >/dev/null 2>&1 || \
        st_die "PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN"

    if [[ -n "${SLURM_TMPDIR:-}" ]]; then
        scratch_base="$SLURM_TMPDIR"
    else
        scratch_base="${TMPDIR:-/tmp}"
    fi
    st_scratch_dir="$scratch_base/silenttwin/$ST_EXPERIMENT/$task_label"
    mkdir -p -- "$st_scratch_dir"

    printf 'scratch_dir=%s\n' "$st_scratch_dir"
    if [[ "$task_label" != aggregate ]] && command -v nvidia-smi >/dev/null 2>&1; then
        if st_nvidia_status="$(nvidia-smi 2>&1)"; then
            printf '%s\n' "$st_nvidia_status"
        fi
    fi
    printf 'TMPDIR=%q ' "$st_scratch_dir"
    st_print_command "${command_ref[@]}"
    TMPDIR="$st_scratch_dir" "${command_ref[@]}"
}

st_append_common_run_flags() {
    local command_name="$1"
    local -n command_ref="$command_name"

    if [[ "$OVERWRITE" == 1 ]]; then
        command_ref+=(--overwrite)
    fi
    command_ref+=("${ST_EXTRA_ARGV[@]}")
}

st_aggregate() {
    local expected_runs="$1"
    local output_dir="$OUT_ROOT/$ST_EXPERIMENT/aggregate"
    local aggregate_command=(
        "$PYTHON_BIN" -m silenttwin.cli aggregate
        --experiment "$ST_EXPERIMENT"
        --input-root "$OUT_ROOT/$ST_EXPERIMENT"
        --output-dir "$output_dir"
        --expected-runs "$expected_runs"
    )
    aggregate_command+=("${ST_EXTRA_ARGV[@]}")
    printf 'aggregation_output=%s\n' "$output_dir"
    st_execute aggregate aggregate_command
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'This file is a shared helper; run one of run_experiment_*.sh instead.'
fi
