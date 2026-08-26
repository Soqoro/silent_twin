#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"
agentdojo_init e5 E5 controlled
agentdojo_dispatch_experiment
