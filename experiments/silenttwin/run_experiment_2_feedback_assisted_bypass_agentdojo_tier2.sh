#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"
agentdojo_init e2 E2 controlled
if [[ "$AGENTDOJO_STAGE" == aggregate && -z "${E1_ANALYSIS_MANIFEST:-}" ]]; then
    candidate="$OUT_ROOT/e1/aggregate/analysis_manifest.json"
    if [[ -f "$candidate" ]]; then
        E1_ANALYSIS_MANIFEST="$candidate"
        export E1_ANALYSIS_MANIFEST
    fi
fi
agentdojo_dispatch_experiment
