# AgentDojo Tier-2 pre-implementation audit

This record captures the checked-out repository state before the AgentDojo
Tier-2 implementation. It is an audit record, not benchmark evidence.

## Confirmed discrepancies

The repository had strong finite-state Tier-1 infrastructure, but it had no
Python 3.11 AgentDojo environment pin, resolved AgentDojo dependency lock,
compatibility/catalog layer, AgentDojo scenario registry, frozen monitor-pair
registry, AgentDojo-specific grid, or AgentDojo SLURM entrypoints. The existing
files named Tier 2 used a local `transformers` attacker against synthetic
finite-state `WorldPair` instances. Pilot C/D therefore remain synthetic
development pilots and are not AgentDojo evidence.

`requirements-dev.lock` was present. A separate, fully resolved 71-package
AgentDojo core lock, `requirements-tier2-agentdojo.lock`, was still needed
because the development environment and AgentDojo's Python 3.11 dependency
graph are distinct. Site/CUDA-specific learned packages are not guessed by
that core lock; the implementation captures every active distribution's
version and RECORD identity in a self-verifying runtime manifest instead.

The audit also confirmed five protocol confounds:

1. The final learned prediction seed could vary with the number of probes;
   final/prediction streams therefore were not demonstrably independent of
   `Q`.
2. Operational model-cache paths participated in scientific configuration
   identity instead of relying only on immutable checkpoint fingerprints.
3. Exact finite-state transcript-distribution metrics could be serialized as
   numeric values for learned Tier-2 runs, where exact enumeration is not
   available.
4. The E2 go/no-go logic did not conjunctively require the preregistered effect
   threshold and a paired confidence-interval lower bound above zero for both
   causal controls.
5. Public/private validation relied on a generic content-key denylist that is
   incompatible with legitimate AgentDojo application data. AgentDojo needs
   typed public and trusted namespaces.

## Implemented boundary

The AgentDojo namespace now keeps package version `0.1.35`, upstream source
revision `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`, and benchmark-data version
`v1.2.2` as separate immutable fields. Catalog, structural split,
candidate-strategy, pair-registry, analysis-plan, dependency-lock,
development-evidence, and sample-size-freeze hashes form an explicit upstream
chain. Operational output, device, and cache paths do not enter scientific
hashes.

The controlled analysis treats a structural scenario/user task as the
independent unit, averages nested stochastic rows within that unit, stratifies
the cluster bootstrap by suite, and gives each observed suite equal primary
weight. Task-weighted pooling is a declared sensitivity analysis. E2 uses the
exact condition names `no_probe`, `genuine`, `matched_shuffled`, `constant`,
`random_final`, `oracle`, and `silenttwin`. Its confirmatory gate requires both
effect size and positive paired-CI evidence against Q0 and matched shuffled,
after a self-hashed compatible E1 gate manifest and local oracle headroom pass.

## Claim boundary

All environments and effects are simulated locally. Reports must call the
outcome an **AgentDojo attack success** or a **benchmark prohibited effect**,
never real-world harm or a live external side effect. The causal claim under
test is whether feedback improves inference of a private enforcement state and
selection of a state-matched bypass under one fixed final-effect opportunity.
SilentTwin is evaluated only for removal of that incremental feedback-assisted
advantage while preserving useful independent work; it is not claimed to fix
the monitor's baseline false negatives.

No checked-in grid, fake-model run, or unexecuted local-model configuration is
itself evidence of leakage, prohibited effects, mitigation, or ecological
prevalence.

The checked deterministic fake-smoke strategy and pair artifacts make
model-free orchestration reproducible on a fresh checkout. They are derived
from a synthetic two-by-two monitor truth table, carry an explicit
`engineering_smoke_only` evidence class, and are ineligible for scientific
analysis. Production strategy and pair artifacts remain operator-authored and
train/development-mined inputs; the smoke fixtures cannot satisfy a
non-fixture grid.
