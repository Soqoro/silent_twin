# SilentTwin configuration reference

[`tier1.json`](tier1.json) is a machine-readable record of the finite-state
defaults used by the experiment launchers. It is deliberately dependency-free
JSON so tooling can inspect it with Python's standard library.

The human-facing launch contract is environment-variable based. The shell
entrypoints do not load this file implicitly: their defaults mirror it, and an
operator overrides a grid with variables such as `RUNTIMES`, `ATTACKERS`, or
`QUERY_BUDGETS`. Keeping grid expansion in the launchers makes the exact mapping
from a SLURM array index visible before submission.

The E5 ablation list includes `none`, the exact-SilentTwin reference, followed
by the fourteen named failures. This reference is required for attributable
leakage comparisons.

Tier 2 model and environment settings should be added as a separate config,
without credentials. Supply secrets through the deployment's secret manager or
process environment, never in a tracked file.
