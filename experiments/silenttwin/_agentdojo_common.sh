#!/bin/bash
# Shared orchestration for the explicit AgentDojo Tier-2 entrypoints.

set -euo pipefail

agentdojo_die() {
    printf 'silenttwin-agentdojo: error: %s\n' "$*" >&2
    exit 2
}

agentdojo_init() {
    AGENTDOJO_EXPERIMENT_ID="$1"
    AGENTDOJO_STAGE_ALIAS="$2"
    AGENTDOJO_TRACK="$3"
    local caller_source="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    AGENTDOJO_REPO_ROOT="${AGENTDOJO_REPO_ROOT:-$(cd -- "$(dirname -- "$caller_source")/../.." && pwd -P)}"

    local alias_name="${AGENTDOJO_STAGE_ALIAS}_STAGE"
    AGENTDOJO_STAGE="${!alias_name:-${STAGE:-run}}"
    case "$AGENTDOJO_STAGE" in
        grid|run|aggregate) ;;
        *) agentdojo_die "STAGE must be grid, run, or aggregate (got: $AGENTDOJO_STAGE)" ;;
    esac

    PYTHON_BIN="${PYTHON_BIN:-python3}"
    OUT_ROOT="${OUT_ROOT:-$AGENTDOJO_REPO_ROOT/outputs/silenttwin/agentdojo}"
    AGENTDOJO_CONFIG_ROOT="${AGENTDOJO_CONFIG_ROOT:-$AGENTDOJO_REPO_ROOT/configs/silenttwin/agentdojo}"
    AGENTDOJO_CATALOG="${AGENTDOJO_CATALOG:-$AGENTDOJO_CONFIG_ROOT/catalog-v1.json}"
    AGENTDOJO_SPLITS="${AGENTDOJO_SPLITS:-$AGENTDOJO_CONFIG_ROOT/splits-v1.json}"
    AGENTDOJO_GRID_PLAN="${AGENTDOJO_GRID_PLAN:-$AGENTDOJO_CONFIG_ROOT/grid-plans/controlled-fake-smoke-v1.json}"
    local checked_fake_plan="$AGENTDOJO_CONFIG_ROOT/grid-plans/controlled-fake-smoke-v1.json"
    local default_strategy_catalog="$AGENTDOJO_CONFIG_ROOT/candidate-strategies-v1.json"
    local default_pair_registry="$AGENTDOJO_CONFIG_ROOT/pair-registry-v1.json"
    case "$AGENTDOJO_EXPERIMENT_ID" in
        e1|e2|e3|e4|e5|ecological)
            if [[ "$(realpath -m -- "$AGENTDOJO_GRID_PLAN")" == \
                "$(realpath -m -- "$checked_fake_plan")" ]]; then
                default_strategy_catalog="$AGENTDOJO_CONFIG_ROOT/fixtures/deterministic-fake-smoke-candidate-strategies-v1.json"
                default_pair_registry="$AGENTDOJO_CONFIG_ROOT/fixtures/deterministic-fake-smoke-pair-registry-v1.json"
            fi
            ;;
    esac
    # Pair mining and every non-smoke plan deliberately retain the production
    # defaults above. Those artifacts must be authored/mined by the operator.
    AGENTDOJO_STRATEGY_CATALOG="${AGENTDOJO_STRATEGY_CATALOG:-$default_strategy_catalog}"
    AGENTDOJO_PAIR_REGISTRY="${AGENTDOJO_PAIR_REGISTRY:-$default_pair_registry}"
    if [[ "$AGENTDOJO_TRACK" == ecological ]]; then
        AGENTDOJO_ANALYSIS_PLAN="${AGENTDOJO_ANALYSIS_PLAN:-$AGENTDOJO_CONFIG_ROOT/analysis/ecological-v1.json}"
    else
        AGENTDOJO_ANALYSIS_PLAN="${AGENTDOJO_ANALYSIS_PLAN:-$AGENTDOJO_CONFIG_ROOT/analysis/controlled-v1.json}"
    fi
    AGENTDOJO_DEPENDENCY_LOCK="${AGENTDOJO_DEPENDENCY_LOCK:-$AGENTDOJO_REPO_ROOT/requirements-tier2-agentdojo.lock}"
    AGENTDOJO_PYTHON_PIN="${AGENTDOJO_PYTHON_PIN:-$AGENTDOJO_CONFIG_ROOT/python-version.txt}"
    AGENTDOJO_DATASET_SPLIT="${AGENTDOJO_DATASET_SPLIT:-development}"
    AGENTDOJO_REPLICATES="${AGENTDOJO_REPLICATES:-0}"
    AGENTDOJO_GROUPS_PER_BUNDLE="${AGENTDOJO_GROUPS_PER_BUNDLE:-8}"
    AGENTDOJO_SAMPLE_SIZE_FREEZE="${AGENTDOJO_SAMPLE_SIZE_FREEZE:-}"
    AGENTDOJO_DEVELOPMENT_ANALYSIS_MANIFEST="${AGENTDOJO_DEVELOPMENT_ANALYSIS_MANIFEST:-}"
    AGENTDOJO_RUN_ROOT="$OUT_ROOT/$AGENTDOJO_EXPERIMENT_ID/runs"
    AGENTDOJO_GRID_DIR="$OUT_ROOT/$AGENTDOJO_EXPERIMENT_ID/grid"
    GRID_MANIFEST="${GRID_MANIFEST:-$AGENTDOJO_GRID_DIR/grid-manifest.jsonl}"
    AGENTDOJO_AGGREGATE_DIR="$OUT_ROOT/$AGENTDOJO_EXPERIMENT_ID/aggregate"

    case "$AGENTDOJO_DATASET_SPLIT" in
        train|development|test) ;;
        *) agentdojo_die "AGENTDOJO_DATASET_SPLIT must be train, development, or test" ;;
    esac
    [[ "$AGENTDOJO_GROUPS_PER_BUNDLE" =~ ^[1-9][0-9]*$ ]] || \
        agentdojo_die "AGENTDOJO_GROUPS_PER_BUNDLE must be a positive integer"

    cd -- "$AGENTDOJO_REPO_ROOT"
    if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="$AGENTDOJO_REPO_ROOT/src:$PYTHONPATH"
    else
        export PYTHONPATH="$AGENTDOJO_REPO_ROOT/src"
    fi
}

agentdojo_offline_environment() {
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export SILENTTWIN_OFFLINE=1
    export AGENTDOJO_ALLOW_API_FALLBACK=0
    export AGENTDOJO_ALLOW_MOCK_FALLBACK=0
    export AGENTDOJO_CATALOG AGENTDOJO_SPLITS AGENTDOJO_STRATEGY_CATALOG
    export AGENTDOJO_PAIR_REGISTRY AGENTDOJO_ANALYSIS_PLAN
    export AGENTDOJO_DEPENDENCY_LOCK
    export AGENTDOJO_SAMPLE_SIZE_FREEZE
    export AGENTDOJO_DEVELOPMENT_ANALYSIS_MANIFEST
    export AGENTDOJO_RUNTIME_FINGERPRINT
    export AGENTDOJO_ATTACKER_CHECKPOINT AGENTDOJO_VICTIM_CHECKPOINT
    export AGENTDOJO_MONITOR_CHECKPOINT
}

agentdojo_activate_and_require_python311() {
    local pin observed
    if [[ -n "${ENV_ACTIVATE:-}" ]]; then
        [[ -f "$ENV_ACTIVATE" ]] || agentdojo_die "ENV_ACTIVATE is not a readable file"
        # shellcheck disable=SC1090
        source "$ENV_ACTIVATE"
    fi
    [[ -f "$AGENTDOJO_PYTHON_PIN" ]] || \
        agentdojo_die "missing Python version pin: $AGENTDOJO_PYTHON_PIN"
    IFS= read -r pin < "$AGENTDOJO_PYTHON_PIN" || \
        agentdojo_die "cannot read Python version pin"
    [[ "$pin" == 3.11 ]] || agentdojo_die "AgentDojo Python pin must be exactly 3.11"
    command -v -- "$PYTHON_BIN" >/dev/null 2>&1 || \
        agentdojo_die "PYTHON_BIN is unavailable: $PYTHON_BIN"
    observed="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" || \
        agentdojo_die "could not inspect PYTHON_BIN"
    [[ "$observed" == "$pin" ]] || \
        agentdojo_die "AgentDojo Tier 2 requires Python $pin (observed $observed)"
}

agentdojo_path_is_within() {
    local candidate root
    candidate="$(realpath -m -- "$1")" || return 1
    root="$(realpath -m -- "$2")" || return 1
    [[ "$candidate" == "$root" || "$candidate" == "$root/"* ]]
}

agentdojo_require_scheduler_job() {
    # Scheduler variables are an explicit authorization assertion, not a
    # substitute for model/runtime identity checks.  Reject ambiguous mixed
    # contexts instead of guessing which scheduler supplied the allocation.
    local operation="${1:-run stage}"
    local slurm_present=0 pbs_present=0
    [[ -n "${SLURM_JOB_ID:-}" ]] && slurm_present=1
    [[ -n "${PBS_JOBID:-}" ]] && pbs_present=1
    if ((slurm_present + pbs_present == 0)); then
        agentdojo_die "$operation is forbidden outside an authorized Slurm or PBS job"
    fi
    if ((slurm_present + pbs_present > 1)); then
        agentdojo_die "ambiguous scheduler context exposes both SLURM_JOB_ID and PBS_JOBID"
    fi
    if ((slurm_present == 1)); then
        AGENTDOJO_SCHEDULER_KIND=slurm
        AGENTDOJO_SCHEDULER_JOB_ID="$SLURM_JOB_ID"
    else
        [[ "${PBS_ENVIRONMENT:-}" == PBS_BATCH ]] || \
            agentdojo_die "$operation requires PBS_ENVIRONMENT=PBS_BATCH"
        AGENTDOJO_SCHEDULER_KIND=pbs
        AGENTDOJO_SCHEDULER_JOB_ID="$PBS_JOBID"
    fi
    export AGENTDOJO_SCHEDULER_KIND AGENTDOJO_SCHEDULER_JOB_ID
}

agentdojo_scheduler_array_index() {
    local variable raw_index
    case "$AGENTDOJO_SCHEDULER_KIND" in
        slurm) variable=SLURM_ARRAY_TASK_ID ;;
        pbs) variable=PBS_ARRAY_INDEX ;;
        *) agentdojo_die "scheduler authorization was not initialized" ;;
    esac
    raw_index="${!variable:-}"
    [[ -n "$raw_index" ]] || \
        agentdojo_die "$variable is required; inspect STAGE=grid first"
    [[ "$raw_index" =~ ^(0|[1-9][0-9]*)$ ]] || \
        agentdojo_die "$variable must be a non-negative base-10 integer"
    AGENTDOJO_SCHEDULER_ARRAY_VARIABLE="$variable"
    AGENTDOJO_SCHEDULER_ARRAY_TASK_ID="$raw_index"
    AGENTDOJO_TASK_ID=$((10#$raw_index))
    export AGENTDOJO_SCHEDULER_ARRAY_VARIABLE
    export AGENTDOJO_SCHEDULER_ARRAY_TASK_ID AGENTDOJO_TASK_ID
}

agentdojo_reject_ephemeral_runtime_paths() {
    local -a scratch_variables=(SLURM_TMPDIR)
    if [[ -n "${PBS_JOBID:-}" ]]; then
        # PBS_JOBDIR is the staging/execution directory.  Under PBS's default
        # HOME sandbox it is PBS_O_HOME and is persistent; under a PRIVATE
        # sandbox it is job-specific and ephemeral.  Fail closed when PBS did
        # not expose PBS_O_HOME, but do not misclassify the normal home tree.
        if [[ -n "${PBS_JOBDIR:-}" ]] && {
            [[ -z "${PBS_O_HOME:-}" ]] ||
                [[ "$(realpath -m -- "$PBS_JOBDIR")" != \
                    "$(realpath -m -- "$PBS_O_HOME")" ]]
        }; then
            scratch_variables+=(PBS_JOBDIR)
        fi
        # TMPDIR is the job scratch directory when PBS assigns one.
        scratch_variables+=(TMPDIR)
    fi
    local scratch_variable scratch_root name value
    for scratch_variable in "${scratch_variables[@]}"; do
        scratch_root="${!scratch_variable:-}"
        [[ -n "$scratch_root" ]] || continue
        for name in AGENTDOJO_MODEL_CACHE HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE \
            AGENTDOJO_ATTACKER_CHECKPOINT AGENTDOJO_VICTIM_CHECKPOINT \
            AGENTDOJO_MONITOR_CHECKPOINT CONFORMANCE_SPEC CONFORMANCE_OUTPUT \
            OUT_ROOT; do
            value="${!name:-}"
            if [[ -n "$value" ]] && agentdojo_path_is_within "$value" "$scratch_root"; then
                agentdojo_die "$name must be persistent, not inside scheduler scratch $scratch_variable"
            fi
        done
        # Pair-observation generation may select a checkpoint per frozen
        # monitor profile.  Reject every configured override before Python or
        # a model client is reached, not just the generic fallback above.
        while IFS= read -r name; do
            value="${!name:-}"
            if [[ -n "$value" ]] && agentdojo_path_is_within "$value" "$scratch_root"; then
                agentdojo_die "$name must be persistent, not inside scheduler scratch $scratch_variable"
            fi
        done < <(compgen -A variable AGENTDOJO_MONITOR_CHECKPOINT_)
    done
}

agentdojo_manifest_total_tasks() {
    local first_line remainder value
    [[ -f "$GRID_MANIFEST" ]] || \
        agentdojo_die "frozen grid manifest does not exist: $GRID_MANIFEST"
    IFS= read -r first_line < "$GRID_MANIFEST" || \
        agentdojo_die "cannot read frozen grid manifest"
    [[ "$first_line" == *'"record_type":"grid_metadata"'* ]] || \
        agentdojo_die "grid manifest first record is not canonical metadata"
    [[ "$first_line" == *'"environment_backend":"agentdojo"'* ]] || \
        agentdojo_die "grid manifest is not AgentDojo"
    [[ "$first_line" == *'"model_free":true'* ]] || \
        agentdojo_die "grid manifest lacks its model-free marker"
    remainder="${first_line#*\"total_tasks\":}"
    [[ "$remainder" != "$first_line" ]] || agentdojo_die "grid metadata lacks total_tasks"
    value="${remainder%%,*}"
    value="${value%%\}*}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || agentdojo_die "grid total_tasks is invalid"
    AGENTDOJO_TOTAL_TASKS="$value"
}

agentdojo_selected_task_runtime_requirements() {
    # Inspect only the already-frozen JSONL bytes.  This remains a shell-only
    # authorization boundary: no environment activation or Python/model import
    # is allowed before the selected array member has been classified.
    local line matched=0 fixture_mode="" member_fixture_mode=""
    AGENTDOJO_SELECTED_TASK_REQUIRES_LEARNED_MODEL=0
    while IFS= read -r line; do
        [[ "$line" == *'"record_type":"grid_member"'* ]] || continue
        if [[ "$line" != *"\"task_id\":$AGENTDOJO_TASK_ID}"* && \
              "$line" != *"\"task_id\":$AGENTDOJO_TASK_ID,"* ]]; then
            continue
        fi
        matched=1
        if [[ "$line" == *'"fixture_mode":true'* ]]; then
            member_fixture_mode=1
        elif [[ "$line" == *'"fixture_mode":false'* ]]; then
            member_fixture_mode=0
        else
            agentdojo_die "selected grid member lacks an explicit fixture_mode"
        fi
        if [[ -n "$fixture_mode" && "$member_fixture_mode" != "$fixture_mode" ]]; then
            agentdojo_die "selected task mixes fixture and production members"
        fi
        fixture_mode="$member_fixture_mode"
        if [[ "$line" == *'"implementation":"local_transformers"'* || \
              "$line" == *'"implementation":"transformers_pi_detector"'* ]]; then
            AGENTDOJO_SELECTED_TASK_REQUIRES_LEARNED_MODEL=1
        fi
    done < "$GRID_MANIFEST"
    ((matched == 1)) || \
        agentdojo_die "frozen grid manifest has no members for selected task"
    AGENTDOJO_SELECTED_TASK_FIXTURE_MODE="$fixture_mode"
}

agentdojo_run_preflight_before_python() {
    # This function deliberately uses no Python.  Authorization and array
    # bounds must fail before imports, environment activation, or model checks.
    agentdojo_require_scheduler_job "run stage"
    agentdojo_scheduler_array_index
    agentdojo_manifest_total_tasks
    ((AGENTDOJO_TASK_ID < AGENTDOJO_TOTAL_TASKS)) || \
        agentdojo_die "$AGENTDOJO_SCHEDULER_ARRAY_VARIABLE=$AGENTDOJO_SCHEDULER_ARRAY_TASK_ID is out of range; valid range is 0-$((AGENTDOJO_TOTAL_TASKS - 1))"

    agentdojo_reject_ephemeral_runtime_paths
    agentdojo_selected_task_runtime_requirements

    # Runtime identity is checked against the selected frozen members, never
    # inferred from a plan filename or an operator-controlled cache path.
    local fake_model="${AGENTDOJO_FAKE_MODEL:-0}"
    [[ "$fake_model" == 0 || "$fake_model" == 1 ]] || \
        agentdojo_die "AGENTDOJO_FAKE_MODEL must be 0 or 1"
    [[ "$fake_model" == "$AGENTDOJO_SELECTED_TASK_FIXTURE_MODE" ]] || \
        agentdojo_die "AGENTDOJO_FAKE_MODEL disagrees with the selected frozen grid members"
    if [[ "$AGENTDOJO_SELECTED_TASK_REQUIRES_LEARNED_MODEL" == 1 ]]; then
        [[ -n "${AGENTDOJO_MODEL_CACHE:-}" ]] || \
            agentdojo_die "selected learned models require AGENTDOJO_MODEL_CACHE"
    fi
    local requires_gpu="${AGENTDOJO_REQUIRES_GPU:-$AGENTDOJO_SELECTED_TASK_REQUIRES_LEARNED_MODEL}"
    [[ "$requires_gpu" == 0 || "$requires_gpu" == 1 ]] || \
        agentdojo_die "AGENTDOJO_REQUIRES_GPU must be 0 or 1"
    # Keep the Python-side artifact validator on the same selected-role
    # decision.  In particular, a nonfixture deterministic-monitor E4/E5
    # shard is genuinely model-free and must not inherit the validator's
    # conservative production default of one GPU.
    AGENTDOJO_REQUIRES_GPU="$requires_gpu"
    export AGENTDOJO_REQUIRES_GPU
    local default_device role_device_name role_device
    default_device=cpu
    if [[ "$requires_gpu" == 1 ]]; then
        default_device=cuda
    fi
    for role_device_name in ATTACKER_DEVICE VICTIM_DEVICE MONITOR_DEVICE; do
        role_device="${!role_device_name:-$default_device}"
        if [[ "$requires_gpu" == 0 && "$role_device" == cuda* ]]; then
            agentdojo_die "$role_device_name requests CUDA while AGENTDOJO_REQUIRES_GPU=0"
        fi
        printf -v "$role_device_name" '%s' "$role_device"
        export "$role_device_name"
    done
    if [[ "$requires_gpu" == 1 ]]; then
        command -v nvidia-smi >/dev/null 2>&1 || \
            agentdojo_die "the scheduler allocation exposes no nvidia-smi"
        nvidia-smi -L >/dev/null 2>&1 || \
            agentdojo_die "the scheduler allocation exposes no visible GPU"
    fi
}

agentdojo_grid_arguments() {
    AGENTDOJO_GRID_ARGUMENTS=(
        --experiment "$AGENTDOJO_EXPERIMENT_ID"
        --track "$AGENTDOJO_TRACK"
        --dataset-split "$AGENTDOJO_DATASET_SPLIT"
        --catalog "$AGENTDOJO_CATALOG"
        --splits "$AGENTDOJO_SPLITS"
        --strategy-catalog "$AGENTDOJO_STRATEGY_CATALOG"
        --pair-registry "$AGENTDOJO_PAIR_REGISTRY"
        --analysis-plan "$AGENTDOJO_ANALYSIS_PLAN"
        --dependency-lock "$AGENTDOJO_DEPENDENCY_LOCK"
        --grid-plan "$AGENTDOJO_GRID_PLAN"
        --groups-per-bundle "$AGENTDOJO_GROUPS_PER_BUNDLE"
    )
    local replicate
    for replicate in $AGENTDOJO_REPLICATES; do
        [[ "$replicate" =~ ^(0|[1-9][0-9]*)$ ]] || \
            agentdojo_die "AGENTDOJO_REPLICATES contains a non-integer"
        AGENTDOJO_GRID_ARGUMENTS+=(--replicate "$replicate")
    done
    if [[ -n "$AGENTDOJO_SAMPLE_SIZE_FREEZE" ]]; then
        AGENTDOJO_GRID_ARGUMENTS+=(--sample-size-freeze "$AGENTDOJO_SAMPLE_SIZE_FREEZE")
    fi
}

agentdojo_grid_stage() {
    agentdojo_offline_environment
    agentdojo_activate_and_require_python311
    agentdojo_grid_arguments
    mkdir -p -- "$AGENTDOJO_GRID_DIR"
    "$PYTHON_BIN" -m silenttwin.agentdojo.grid manifest \
        "${AGENTDOJO_GRID_ARGUMENTS[@]}" --output "$GRID_MANIFEST" >/dev/null
    "$PYTHON_BIN" -m silenttwin.agentdojo.grid print "${AGENTDOJO_GRID_ARGUMENTS[@]}"
    printf 'grid_manifest=%s\n' "$GRID_MANIFEST"
}

agentdojo_run_stage() {
    agentdojo_run_preflight_before_python
    agentdojo_offline_environment
    agentdojo_activate_and_require_python311
    AGENTDOJO_TASK_OUTPUT_DIR="$AGENTDOJO_RUN_ROOT/task-$AGENTDOJO_TASK_ID"
    export AGENTDOJO_TASK_OUTPUT_DIR
    mkdir -p -- "$AGENTDOJO_TASK_OUTPUT_DIR"
    exec "$PYTHON_BIN" -m silenttwin.agentdojo.runner run-grid-task \
        --grid-manifest "$GRID_MANIFEST" --task-id "$AGENTDOJO_TASK_ID"
}

agentdojo_aggregate_stage() {
    agentdojo_offline_environment
    agentdojo_activate_and_require_python311
    [[ -f "$GRID_MANIFEST" ]] || agentdojo_die "missing frozen grid manifest"
    local command=(
        "$PYTHON_BIN" -m silenttwin.agentdojo.aggregate
        --input-root "$AGENTDOJO_RUN_ROOT"
        --output-dir "$AGENTDOJO_AGGREGATE_DIR"
        --expected-grid-manifest "$GRID_MANIFEST"
        --analysis-plan "$AGENTDOJO_ANALYSIS_PLAN"
    )
    if [[ "${AGENTDOJO_ALLOW_DEVELOPMENT_PARTIAL:-0}" == 1 ]]; then
        command+=(--allow-development-partial)
    fi
    if [[ -n "${E1_ANALYSIS_MANIFEST:-}" ]]; then
        command+=(--upstream-e1-analysis-manifest "$E1_ANALYSIS_MANIFEST")
    fi
    "${command[@]}"
}

agentdojo_dispatch_experiment() {
    case "$AGENTDOJO_STAGE" in
        grid) agentdojo_grid_stage ;;
        run) agentdojo_run_stage ;;
        aggregate) agentdojo_aggregate_stage ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'This is a source-only AgentDojo helper; run an AgentDojo entrypoint.'
fi
