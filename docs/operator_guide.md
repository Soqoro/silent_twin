# Operator guide

All human-facing experiment commands are Bash launchers under
`experiments/silenttwin/`. They use the same three stages:

- `grid` validates environment values and prints every deterministic array-task
  mapping without running Python;
- `run` requires `SLURM_ARRAY_TASK_ID` and executes exactly that configuration;
- `aggregate` recursively validates and summarizes the experiment's leaf runs.

The launchers change to the repository root, prepend `src/` to `PYTHONPATH`,
construct commands as arrays, and print every argument with shell escaping. They
never use `eval`, call an external API, download a model, or submit a job.
Run the documented commands from the repository root; SLURM resolves its log
path before the script can change directories.

## Configuration

The common defaults are:

| Variable | Default | Meaning |
|---|---|---|
| `TIERS` | `tier1` | Execution tier; Tier 2 is an unconfigured extension interface |
| `WORLD_SUITES` | `email files calendar payments` | Public workflow templates |
| `RUNTIMES` | eight baseline/full runtime names | Runtime grid; E3 adds three selected degraded runtimes, while E4/E5 use experiment-specific subsets |
| `ATTACKERS` | `bayesian black_box mock_llm` | Attacker grid; E4 defaults to `mock_llm` |
| `QUERY_BUDGETS` | `0 4 16 32` | Adaptive probe budgets; E4 defaults to `0` |
| `SEEDS` | `42` | Non-negative integer seeds |
| `NUM_SAMPLES` | `-1` | Use the runner's deterministic default (currently 16); set `2` for smoke runs |
| `OUT_ROOT` | `outputs/silenttwin` | Result root, relative to the repository unless absolute |
| `OVERWRITE` | `0` | `1` explicitly authorizes replacement instead of compatible reuse |
| `PYTHON_BIN` | `python3` | Python executable or absolute executable path |
| `EXTRA_ARGS` | empty | Additional whitespace-delimited internal CLI arguments |

Experiment-specific grids add `E2_CONDITIONS` (`adaptive random oracle
shuffled`), `E4_WORKFLOWS` (`independent rejection_dependent atomic`), and the
exact `none` plus 14 failures in `E5_ABLATIONS`. The full reference is
[`configs/silenttwin/tier1.json`](../configs/silenttwin/tier1.json).

List values are split on shell whitespace. Quoting inside `EXTRA_ARGS` is not
reinterpreted: this is deliberate because the scripts never evaluate input as
shell code. For an argument that itself contains whitespace, invoke the internal
CLI in controlled development code rather than trying to encode shell syntax in
`EXTRA_ARGS`.

All factor tokens are restricted to letters, digits, `_`, `.`, and `-` so they
are safe path components, and duplicate factor values are rejected. Query
budgets and seeds must be non-negative decimal integers. `NUM_SAMPLES` must be
`-1` or positive, and `OVERWRITE` must be `0` or `1`. Invalid stages, values, or
array indices fail before the experiment starts.

## Grid order and output paths

The printed ordering is lexicographic in the operator-supplied list order. The
leftmost factor varies slowest; `seed` varies fastest. `select_config` and
`print_grid` use the same enumeration, so the printed line for task `i` is the
configuration selected by `SLURM_ARRAY_TASK_ID=i`.

Default sizes are:

| Experiment | Factor order | Jobs | Valid range |
|---|---|---:|---:|
| E1 | tier, world, runtime, attacker, budget, seed | 384 | `0-383` |
| E2 | tier, world, runtime, attacker, budget, condition, seed | 1,536 | `0-1535` |
| E3 | tier, world, baseline/full/degraded runtime, attacker, budget, seed | 528 | `0-527` |
| E4 | tier, world, runtime, attacker, budget, workflow, seed | 36 | `0-35` |
| E5 | tier, world, runtime, attacker, budget, ablation, seed | 720 | `0-719` |

Overrides change these counts. Always run the matching `grid` command and use
its `valid_array_range`; do not reuse a range from a different environment.
Repeat those same overrides for `aggregate`, which passes the expanded job count
to the validator and therefore treats an absent grid member as an error.
Use a dedicated `OUT_ROOT` for a reduced grid: extra leaf runs from a different
grid are also rejected rather than guessed away.

Every scientific factor appears in the leaf path. For example:

```text
outputs/silenttwin/e1/tier=tier1/world=email/runtime=generic_failure/attacker=mock_llm/q=4/seed=42/
outputs/silenttwin/e2/tier=tier1/world=email/runtime=generic_failure/attacker=mock_llm/q=4/condition=adaptive/seed=42/
outputs/silenttwin/e4/tier=tier1/world=email/runtime=silenttwin/attacker=mock_llm/q=0/workflow=independent/seed=42/
```

Each leaf contains `result.jsonl`, `manifest.json`, and `run.log`. Scheduler
stdout/stderr is separate under `logs/`.

## Exact commands by experiment

### E1 — feedback leakage

```bash
E1_STAGE=grid bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
ATTACKERS=mock_llm QUERY_BUDGETS=4 SEEDS=42 NUM_SAMPLES=2 \
SLURM_ARRAY_TASK_ID=0 \
bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

```bash
mkdir -p logs
sbatch --array=0-383 experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

```bash
E1_STAGE=aggregate bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

### E2 — feedback-assisted bypass

```bash
E2_STAGE=grid bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
ATTACKERS=mock_llm QUERY_BUDGETS=4 E2_CONDITIONS=adaptive \
SEEDS=42 NUM_SAMPLES=2 SLURM_ARRAY_TASK_ID=0 \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

```bash
mkdir -p logs
sbatch --array=0-1535 experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

```bash
E2_STAGE=aggregate bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

### E3 — channel closure

```bash
E3_STAGE=grid bash experiments/silenttwin/run_experiment_3_channel_closure.sh
```

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
ATTACKERS=mock_llm QUERY_BUDGETS=4 SEEDS=42 NUM_SAMPLES=2 \
SLURM_ARRAY_TASK_ID=0 \
bash experiments/silenttwin/run_experiment_3_channel_closure.sh
```

```bash
mkdir -p logs
sbatch --array=0-527 experiments/silenttwin/run_experiment_3_channel_closure.sh
```

```bash
E3_STAGE=aggregate bash experiments/silenttwin/run_experiment_3_channel_closure.sh
```

### E4 — useful work

```bash
E4_STAGE=grid bash experiments/silenttwin/run_experiment_4_useful_work.sh
```

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
ATTACKERS=mock_llm QUERY_BUDGETS=0 E4_WORKFLOWS=independent \
SEEDS=42 NUM_SAMPLES=2 SLURM_ARRAY_TASK_ID=0 \
bash experiments/silenttwin/run_experiment_4_useful_work.sh
```

```bash
mkdir -p logs
sbatch --array=0-35 experiments/silenttwin/run_experiment_4_useful_work.sh
```

```bash
E4_STAGE=aggregate bash experiments/silenttwin/run_experiment_4_useful_work.sh
```

### E5 — assumption ablations

```bash
E5_STAGE=grid bash experiments/silenttwin/run_experiment_5_assumption_ablations.sh
```

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=silenttwin \
ATTACKERS=mock_llm QUERY_BUDGETS=4 E5_ABLATIONS=timing_leak \
SEEDS=42 NUM_SAMPLES=2 SLURM_ARRAY_TASK_ID=0 \
bash experiments/silenttwin/run_experiment_5_assumption_ablations.sh
```

```bash
mkdir -p logs
sbatch --array=0-719 experiments/silenttwin/run_experiment_5_assumption_ablations.sh
```

```bash
E5_STAGE=aggregate bash experiments/silenttwin/run_experiment_5_assumption_ablations.sh
```

For a custom grid, export the same overrides during both inspection and
submission. `sbatch --export=ALL,VARIABLE=value,...` is one site-portable way to
make that explicit; follow the local scheduler's policy for values containing
spaces.

## SLURM and scratch behavior

The launchers contain portable `#!/bin/bash`, `%A_%a` stdout/stderr paths, CPU,
memory, and time hints. The referenced `run_experiment_b.sh` was not available
in this repository, so only the conventions explicitly supplied with this task
could be retained. No cluster-specific account or partition is hard-coded. Add
those at submission time if the site requires them, for example:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --array=0-383 \
experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

No launcher requests or overwrites a GPU selection. If `nvidia-smi` exists and
succeeds, its status is printed for provenance; otherwise it is skipped. When SLURM provides
`SLURM_TMPDIR`, the command receives an experiment/task subdirectory there as
`TMPDIR`. Local jobs use `${TMPDIR:-/tmp}/silenttwin/...`. Scientific outputs
remain under `OUT_ROOT`, not ephemeral scratch.

`logs/` must exist before `sbatch` opens the configured output path. The scripts
also create it safely for local execution, but the scheduler opens its log
before the script begins, hence the explicit `mkdir -p logs` in every submission
example.

## Reuse, overwrite, and failure behavior

Before reusing a completed leaf, the internal runner validates its result schema,
scientific configuration hash, resolved sample count, and generation/evaluation
provenance. Compatible output is reused. An incomplete or incompatible leaf is
never silently accepted. Replacement requires an intentional rerun with:

```bash
OVERWRITE=1 SLURM_ARRAY_TASK_ID=0 \
bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

Do not set `OVERWRITE=1` in a broad array unless replacing every selected leaf
is intended. Atomic publication prevents a failed run from masquerading as a
complete result.

Aggregation writes `summary.json` and `summary.csv` under
`$OUT_ROOT/eN/aggregate/`. E1/E3 also export inference-accuracy and
transcript-AUC curves versus query budget; E2/E3 export prohibited-effect curves
and feedback-assisted gain; E3/E4/E5 export a privacy–safety–utility table; and
E5 exports the ablation table. Aggregation fails on missing/incomplete results,
schema conflicts, mixed task cohorts, or incompatible configuration groups
instead of silently averaging them.

## Local pilot suite and verification

The following runs one two-sample mock task for each experiment and then all
five aggregators:

```bash
OUT_ROOT=outputs/silenttwin-pilot \
bash experiments/silenttwin/run_all_pilots.sh
```

Run only one phase with `PILOT_STAGE=run` or `PILOT_STAGE=aggregate`. Standard
verification is:

```bash
bash -n experiments/silenttwin/*.sh
pytest -q
```

These commands are local; nothing in the repository invokes `sbatch`.
