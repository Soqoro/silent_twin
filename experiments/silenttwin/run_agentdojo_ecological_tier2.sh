#!/bin/bash
set -euo pipefail
if [[ -n "${AGENTDOJO_REPO_ROOT:-}" ]]; then
    script_dir="$AGENTDOJO_REPO_ROOT/experiments/silenttwin"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"
agentdojo_init ecological ECOLOGICAL ecological
agentdojo_dispatch_experiment
