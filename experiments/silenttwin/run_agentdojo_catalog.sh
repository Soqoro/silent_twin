#!/bin/bash
# Freeze the pinned AgentDojo catalog and structural split registry on CPU.
set -euo pipefail
if [[ -n "${AGENTDOJO_REPO_ROOT:-}" ]]; then
    script_dir="$AGENTDOJO_REPO_ROOT/experiments/silenttwin"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
# shellcheck source=_agentdojo_common.sh
source "$script_dir/_agentdojo_common.sh"
agentdojo_init catalog CATALOG controlled

case "$AGENTDOJO_STAGE" in
    grid)
        printf 'environment_backend=agentdojo\n'
        printf 'agentdojo_package_version=0.1.35\n'
        printf 'agentdojo_source_revision=a75aba7631d3ca5fb7ab938965c97ead2f9ff84b\n'
        printf 'agentdojo_benchmark_version=v1.2.2\n'
        printf 'catalog_output=%s\n' "$AGENTDOJO_CATALOG"
        printf 'splits_output=%s\n' "$AGENTDOJO_SPLITS"
        printf '%s\n' 'run_command=STAGE=run bash experiments/silenttwin/run_agentdojo_catalog.sh'
        ;;
    run)
        agentdojo_offline_environment
        agentdojo_activate_and_require_python311
        exec "$PYTHON_BIN" -m silenttwin.agentdojo.cli freeze-catalog \
            --source-revision a75aba7631d3ca5fb7ab938965c97ead2f9ff84b \
            --catalog-output "$AGENTDOJO_CATALOG" \
            --splits-output "$AGENTDOJO_SPLITS"
        ;;
    aggregate)
        [[ -f "$AGENTDOJO_CATALOG" ]] || agentdojo_die "catalog artifact is absent"
        [[ -f "$AGENTDOJO_SPLITS" ]] || agentdojo_die "split artifact is absent"
        agentdojo_offline_environment
        agentdojo_activate_and_require_python311
        "$PYTHON_BIN" -m silenttwin.agentdojo.cli verify-catalog \
            --catalog "$AGENTDOJO_CATALOG" --splits "$AGENTDOJO_SPLITS"
        read -r catalog_digest _ < <(sha256sum -- "$AGENTDOJO_CATALOG")
        read -r splits_digest _ < <(sha256sum -- "$AGENTDOJO_SPLITS")
        printf 'catalog_file_sha256=%s\n' "$catalog_digest"
        printf 'splits_file_sha256=%s\n' "$splits_digest"
        ;;
esac
