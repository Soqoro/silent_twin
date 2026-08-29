#!/bin/bash
# Scientific-v6 train-only E1/E2 entrypoint.

set -euo pipefail

if [[ -n "${AGENTDOJO_REPO_ROOT:-}" ]]; then
    script_dir="$AGENTDOJO_REPO_ROOT/experiments/silenttwin"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi

# Do not inherit the ordinary fake-smoke defaults.  Every scientific-v6 run
# must name the clean-commit materialized plan and both immutable design
# artifacts explicitly.
: "${AGENTDOJO_STRATEGY_CATALOG:?set the scientific-v6 recipient-separation candidate catalog}"
: "${AGENTDOJO_PAIR_REGISTRY:?set the scientific-v6 train-only pair registry}"
: "${AGENTDOJO_GRID_PLAN:?set the clean-commit materialized scientific-v6 train grid plan}"

RECIPIENT_EXPERIMENT="${RECIPIENT_EXPERIMENT:-}"
case "$RECIPIENT_EXPERIMENT" in
    e1) stage_alias=E1 ;;
    e2) stage_alias=E2 ;;
    *)
        printf '%s\n' \
            'silenttwin-agentdojo: error: RECIPIENT_EXPERIMENT must be e1 or e2' >&2
        exit 2
        ;;
esac

if [[ -n "${AGENTDOJO_DATASET_SPLIT:-}" && \
      "$AGENTDOJO_DATASET_SPLIT" != train ]]; then
    printf '%s\n' \
        'silenttwin-agentdojo: error: scientific-v6 recipient separation is train-only' >&2
    exit 2
fi
export AGENTDOJO_DATASET_SPLIT=train
export AGENTDOJO_ANALYSIS_PLAN="${AGENTDOJO_ANALYSIS_PLAN:-$script_dir/../../configs/silenttwin/agentdojo/analysis/recipient-separation-v1.json}"

# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"
agentdojo_init "$RECIPIENT_EXPERIMENT" "$stage_alias" controlled
agentdojo_dispatch_experiment
