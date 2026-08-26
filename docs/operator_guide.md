# Operator guide

All human-facing experiment entrypoints are Bash scripts under
`experiments/silenttwin/`. Nothing in this repository calls `sbatch`, allocates
a GPU, downloads a model, or contacts a model provider automatically.

## Reproducible CPU environment

SilentTwin supports Python 3.10–3.13 and Bash 4.3 or newer. From a clean clone:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pytest -q
bash -n experiments/silenttwin/*.sh
```

`requirements-dev.lock` is the pinned CPU test environment. It intentionally
does not install `torch`, `transformers`, CUDA wheels, or a model checkpoint.
Those Tier-2 dependencies must come from the cluster's approved environment.
CI runs the same lock and tests with network model access disabled on Python
3.10, 3.11, 3.12, and 3.13.

## The three stages

Every experiment launcher supports `grid`, `run`, and `aggregate`:

- `grid` validates the entire grid and prints it without running an experiment
  or loading a model;
- `run` requires `SLURM_ARRAY_TASK_ID`, selects one task, and runs its one cell
  or documented batch;
- `aggregate` regenerates the exact expected grid manifest and rejects missing,
  extra, duplicate, or substituted members before analysis.

Use `E1_STAGE` through `E5_STAGE` to choose a stage. The default is `run`.
Commands are assembled as Bash arrays, printed with `%q`, and never passed to
`eval`.

The grid output declares:

- `total_tasks` and `valid_array_range`;
- the exact factor order, with the rightmost factor varying fastest;
- `cells_per_task` for batched grids;
- a deterministic overall `grid_hash`;
- every `task_id`, `batch_offset`, `configuration_hash`, `shard_id`, and all
  normalized scientific fields.

Each printed row repeats both its configuration hash and overall grid hash.
The authoritative JSONL manifest begins with one `grid_metadata` row and then
contains one `grid_member` row per scientific cell/shard. Aggregate identity is
the exact pair `{configuration_hash, shard_id}`. A matching row count alone is
never sufficient.

Always inspect the grid using the same environment that will be exported to
`run` and `aggregate`. Changing a model path, decoding setting, dataset
revision, sample range, factor order, batching, or sharding changes an identity
or the overall grid hash.

## Explicit pilot presets

The checked-in presets are under `configs/silenttwin/pilots/`:

| Preset | Purpose | E1 tasks / members | E2 tasks / members | Trial rows per member |
|---|---|---:|---:|---:|
| Pilot A | CPU engineering smoke | 6 / 6 | 8 / 8 | E1 genuine 32, shuffled 64; E2 64 |
| Pilot B | batched Tier-1 mechanism sweep | 40 / 160 | 32 / 128 | E1 genuine 128, shuffled 256; E2 256 |
| Pilot C | Tier-2 semantic development pilot | 20 / 20 | 48 / 48 | 20 |
| Pilot D | Tier-2 signal development pilot | 480 / 480 | 960 / 960 | 20 |

`num_samples` means trial rows, not public paired instances. E1 balances two
target-state rows per public instance under genuine feedback. Its online
shuffled control independently crosses target and donor state and therefore
uses four rows for the same public instance. E2 likewise counterbalances four
target/donor rows. Pilot C/D keep each GPU leaf at 20 trial rows: an E1 genuine
leaf contains 10 public tasks and an E1 shuffled leaf contains 5; aggregation
recombines all leaves before pairing the same complete public-task cohort.

Pilot B batches four scientific cells into each SLURM task. Each cell still has
its own configuration hash, output directory, manifest, failure ledger, and
checkpoints. Pilot B E1 includes both `authorization` and
`monitor_blind_spot` pair families and expands only these feedback/budget
combinations:

```text
E1 genuine:  Q=0,4,16,32
E1 shuffled: Q=16
```

The Q=16 shuffled cell is the preregistered state-independent E1 control. Pilot
A uses shuffled Q=4 so the engineering smoke exercises the same online donor
path without pretending to evaluate the Q=16 G2 criterion.

Pilot B E2 uses only monitor-blind-spot pairs and expands only these valid
condition/budget combinations:

```text
no_probe: Q=0
oracle:   Q=0
genuine:  Q=4,16,32
shuffled: Q=4,16,32
```

The historical `adaptive` spelling is normalized to `genuine` before hashing.
Known-invalid opaque-termination/target-feedback crosses are also absent from
custom E2 grids.

All four presets are development configurations. Pilot C and D results must
not be used as held-out headline evidence.

## Local CPU smoke tests

Inspect Pilot A:

```bash
PILOT_STAGE=grid \
bash experiments/silenttwin/run_pilot_a_engineering_smoke.sh
```

Run the complete Pilot A E1/E2 grid and aggregate it locally:

```bash
OUT_ROOT=outputs/silenttwin-pilot-a \
bash experiments/silenttwin/run_pilot_a_engineering_smoke.sh
```

This uses only the deterministic mock attacker. It does not establish real-model
leakage or causal harm.

For a minimal direct E1 check:

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
ATTACKERS=mock_llm QUERY_BUDGETS=4 SEEDS=42 NUM_SAMPLES=2 \
SLURM_ARRAY_TASK_ID=0 OUT_ROOT=outputs/silenttwin-smoke \
bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

E2 must contain a complete four-row target/donor block:

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
ATTACKERS=mock_llm QUERY_BUDGETS=4 E2_CONDITIONS=adaptive \
SEEDS=42 NUM_SAMPLES=4 SLURM_ARRAY_TASK_ID=0 \
OUT_ROOT=outputs/silenttwin-smoke \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

The small all-experiment compatibility smoke uses two rows except for E2's
required four-row block:

```bash
OUT_ROOT=outputs/silenttwin-pilot \
bash experiments/silenttwin/run_all_pilots.sh
```

## Tier-1 Pilot B on SLURM

Inspect the two batched grids first:

```bash
PILOT_PRESET=pilot_b E1_STAGE=grid \
bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh

PILOT_PRESET=pilot_b E2_STAGE=grid \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

E1 currently prints `valid_array_range=0-39`; E2 prints `0-31`. Both batch four
cells per task. After creating the scheduler log directory, the prepared
submissions are:

```bash
mkdir -p logs

sbatch --array=0-39 \
  --export=ALL,PILOT_PRESET=pilot_b,E1_STAGE=run,OUT_ROOT=outputs/silenttwin-pilot-b \
  experiments/silenttwin/run_experiment_1_feedback_leakage.sh

sbatch --array=0-31 \
  --export=ALL,PILOT_PRESET=pilot_b,E2_STAGE=run,OUT_ROOT=outputs/silenttwin-pilot-b \
  experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

No valid account or CPU partition is recorded for this repository. If the site
requires either, add its approved submission flags; do not copy the GPU
partition/node values from the sibling reference repository.

After all array members complete, use the exact same preset and output root:

```bash
PILOT_PRESET=pilot_b E1_STAGE=aggregate \
OUT_ROOT=outputs/silenttwin-pilot-b \
bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh

PILOT_PRESET=pilot_b E2_STAGE=aggregate \
OUT_ROOT=outputs/silenttwin-pilot-b \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

## Tier-2 model and environment contract

Tier-2 inference is implemented by a lazy local `transformers` adapter. It sets
`local_files_only=True` and `trust_remote_code=False` for model and tokenizer
loading. A missing package, unavailable CUDA device, absent checkpoint, or
missing immutable revision fails clearly; no download or mock fallback occurs.

Set these values before grid inspection, submission, and aggregation:

| Variable | Requirement |
|---|---|
| `PILOT_PRESET` | `pilot_c` or `pilot_d` for development; empty only for a frozen custom test grid |
| `MODEL_ID` | approved local model identifier/path |
| `MODEL_REVISION` | exact 40–64 hex commit or `sha256:<64-hex>` local-checkpoint fingerprint |
| `MODEL_CACHE_DIR` | persistent or scheduler-approved cache, never `SLURM_TMPDIR` |
| `DATASET_REVISION` | exactly `silenttwin-tier1-v1` |
| `DTYPE` | optional override: `float32`, `float16`, or `bfloat16` |
| `MAX_NEW_TOKENS` | optional positive integer override |
| `TEMPERATURE` | optional non-negative override |
| `TOP_P` | optional value in `(0,1]` |
| `DECODING_SEED` | optional single-seed override; presets otherwise declare their seed list |
| `BATCH_SIZE` | exactly `1`; independent-episode model batching is not implemented |
| `ENV_ACTIVATE` | optional readable activation script for the approved environment |
| `OUT_ROOT` | persistent result path; use an absolute shared path on cluster |

Decoding seeds are declared by each preset and are included in each scientific
hash. Pilot C is deterministic (`temperature=0`, one decoding seed). Pilot D
uses the preregistered stochastic settings `temperature=0.2`, `top_p=0.95`, two
decoding seeds, and `batch_size=1`; the seeds therefore represent actual
stochastic replications rather than duplicate greedy work. The wrappers set
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME`,
and `HF_HUB_CACHE`. They do not set `CUDA_VISIBLE_DEVICES`; the scheduler owns
GPU visibility.

The CPU development lock deliberately cannot run Tier 2. Provision compatible
cluster-approved `torch`, CUDA, and `transformers` versions in the environment
activated by `ENV_ACTIVATE`.

For a checkpoint stored as a local directory, compute its full-byte identity
once and write the reusable fingerprint manifest to persistent storage before
freezing or inspecting a grid:

```bash
python -m silenttwin.cli fingerprint-model \
  --model-dir "$MODEL_ID" \
  --cache-dir "$MODEL_CACHE_DIR"
```

Set `MODEL_REVISION` to the emitted `sha256:<64-hex>` value. Each shard checks
the cached file set, sizes, modification times, and change times. Set
`SILENTTWIN_FULL_CHECKPOINT_REHASH=1` for a full-byte audit inside a shard. For
a locally cached Hub model identifier instead of a directory, use the exact
40–64 hexadecimal commit resolved by the approved cache; branches such as
`main` are rejected.

## Tier-2 grid inspection

Grid inspection is safe on a login node: it validates hashes and rows but does
not import `torch`, initialize CUDA, or load the checkpoint.

```bash
export MODEL_ID=/approved/local/model
export MODEL_REVISION=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
export MODEL_CACHE_DIR=/persistent/silenttwin/model-cache
export DATASET_REVISION=silenttwin-tier1-v1
export OUT_ROOT=/persistent/silenttwin/results

PILOT_PRESET=pilot_c E1_STAGE=grid \
bash experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

PILOT_PRESET=pilot_c E2_STAGE=grid \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh

PILOT_PRESET=pilot_d E1_STAGE=grid \
bash experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

PILOT_PRESET=pilot_d E2_STAGE=grid \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
```

The all-`a` model commit above illustrates the required immutable syntax; it is
not a claimed checkpoint revision. Replace it with the exact commit or local
checkpoint fingerprint approved for the run.

With the checked-in presets, the ranges are Pilot C E1 `0-19`, Pilot C E2
`0-47`, Pilot D E1 `0-479`, and Pilot D E2 `0-959`. Treat the actual printed
range and hash as authoritative.

## Prepared Tier-2 submissions

This repository does not contain verified account, GPU partition, or one-GPU
request syntax. Before using the commands below, replace each angle-bracket
placeholder with one complete site-approved `sbatch` argument. Omit the account
argument only if the scheduler does not require one. The sibling repository's
`NA100q`/`node01` settings are not assumed valid here.

The current presets recommend an array throttle of four. Pilot C submissions:

```bash
mkdir -p logs

sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
  --array=0-19%4 \
  "--export=ALL,PILOT_PRESET=pilot_c,E1_STAGE=run,MODEL_ID=$MODEL_ID,MODEL_REVISION=$MODEL_REVISION,MODEL_CACHE_DIR=$MODEL_CACHE_DIR,DATASET_REVISION=$DATASET_REVISION,OUT_ROOT=$OUT_ROOT,ENV_ACTIVATE=${ENV_ACTIVATE:-}" \
  experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
  --array=0-47%4 \
  "--export=ALL,PILOT_PRESET=pilot_c,E2_STAGE=run,MODEL_ID=$MODEL_ID,MODEL_REVISION=$MODEL_REVISION,MODEL_CACHE_DIR=$MODEL_CACHE_DIR,DATASET_REVISION=$DATASET_REVISION,OUT_ROOT=$OUT_ROOT,ENV_ACTIVATE=${ENV_ACTIVATE:-}" \
  experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
```

Pilot D submissions:

```bash
mkdir -p logs

sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
  --array=0-479%4 \
  "--export=ALL,PILOT_PRESET=pilot_d,E1_STAGE=run,MODEL_ID=$MODEL_ID,MODEL_REVISION=$MODEL_REVISION,MODEL_CACHE_DIR=$MODEL_CACHE_DIR,DATASET_REVISION=$DATASET_REVISION,OUT_ROOT=$OUT_ROOT,ENV_ACTIVATE=${ENV_ACTIVATE:-}" \
  experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
  --array=0-959%4 \
  "--export=ALL,PILOT_PRESET=pilot_d,E2_STAGE=run,MODEL_ID=$MODEL_ID,MODEL_REVISION=$MODEL_REVISION,MODEL_CACHE_DIR=$MODEL_CACHE_DIR,DATASET_REVISION=$DATASET_REVISION,OUT_ROOT=$OUT_ROOT,ENV_ACTIVATE=${ENV_ACTIVATE:-}" \
  experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
```

These are proposed commands only. No command above was submitted by the
repository setup. Values used in `--export` must not contain commas; use the
site's export-file mechanism if a value does.

The Tier-2 wrappers already request 8 CPUs, 64 GiB memory, and 12 hours. They
intentionally have no guessed GPU/account/partition directive. At `run` time
they require `SLURM_JOB_ID`, verify that `nvidia-smi` can see a GPU, reject a
model cache under `SLURM_TMPDIR`, and print GPU/scheduler information before the
experiment command.

## Tier-2 aggregation

Aggregation does not load the model, but it must receive the identical model,
dataset, decoding, cache, preset, and output settings so it regenerates the
same hashes:

```bash
PILOT_PRESET=pilot_c E1_STAGE=aggregate \
bash experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

PILOT_PRESET=pilot_c E2_STAGE=aggregate \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
```

Use `pilot_d` instead for Pilot D. The exported model and path variables from
grid inspection must remain set.

## Freeze and run the held-out test

Do not inspect the test split, generate a test grid, change its factors, or tune
thresholds until the final sample-size record exists. First complete and
strictly aggregate both Pilot D experiments on the development split. Then
freeze each experiment's Pilot-D power recommendation into its own immutable
path; a freeze cannot be reused across experiments:

```bash
PILOT_PRESET=pilot_d E1_STAGE=aggregate \
OUT_ROOT="$OUT_ROOT" \
bash experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

PILOT_PRESET=pilot_d E2_STAGE=aggregate \
OUT_ROOT="$OUT_ROOT" \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh

python -m silenttwin.cli freeze-sample-size \
  --analysis-manifest "$OUT_ROOT/e1/aggregate/analysis_manifest.json" \
  --output-file /persistent/silenttwin/freezes/e1-final-sample-size.json

python -m silenttwin.cli freeze-sample-size \
  --analysis-manifest "$OUT_ROOT/e2/aggregate/analysis_manifest.json" \
  --output-file /persistent/silenttwin/freezes/e2-final-sample-size.json
```

`freeze-sample-size` accepts only an exact-grid Pilot-D development analysis
with a completed power estimate. It refuses to overwrite a different record.
The recommendation uses one conservative complete-block binary outcome per
public task; its manifest also states that median discordance for one primary
contrast does not guarantee power in the noisiest stratum or for every
contrast.

Only after that command succeeds, bind the test grid to the freeze and to an
otherwise fixed scientific configuration. This example carries Pilot D's
declared factors into the custom held-out Tier-2 route. `MODEL_ID`,
`MODEL_REVISION`, `MODEL_CACHE_DIR`, and `OUT_ROOT` remain exported exactly as
they were for Pilot D:

```bash
export PILOT_PRESET=
export DATASET_SPLIT=test
export DATASET_REVISION=silenttwin-tier1-v1
export WORLD_SUITES='email files'
export RUNTIMES='generic_failure binary_denial silenttwin'
export QUERY_BUDGETS='0 16'
export SEEDS=42
export PAIR_FAMILIES=monitor_blind_spot
export DECODING_SEEDS='0 1'
export EPISODES_PER_SHARD=20
export DTYPE=bfloat16
export MAX_NEW_TOKENS=256
export TEMPERATURE=0.2
export TOP_P=0.95
export BATCH_SIZE=1

FEEDBACK_SOURCES='genuine shuffled' \
FEEDBACK_SOURCE_QUERY_BUDGETS='genuine:0,16 shuffled:16' \
SAMPLE_SIZE_FREEZE=/persistent/silenttwin/freezes/e1-final-sample-size.json \
E1_STAGE=grid \
bash experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

E2_CONDITIONS='no_probe genuine shuffled oracle' \
SAMPLE_SIZE_FREEZE=/persistent/silenttwin/freezes/e2-final-sample-size.json \
E2_STAGE=grid \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
```

Use each printed `valid_array_range` and the same exported values in the
site-approved Tier-2 `sbatch` form shown above. After every member completes,
aggregate without loading the model:

```bash
FEEDBACK_SOURCES='genuine shuffled' \
FEEDBACK_SOURCE_QUERY_BUDGETS='genuine:0,16 shuffled:16' \
SAMPLE_SIZE_FREEZE=/persistent/silenttwin/freezes/e1-final-sample-size.json \
E1_STAGE=aggregate \
bash experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh

E2_CONDITIONS='no_probe genuine shuffled oracle' \
SAMPLE_SIZE_FREEZE=/persistent/silenttwin/freezes/e2-final-sample-size.json \
E2_STAGE=aggregate \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
```

The freeze hash, development-analysis hash, primary contrast, and frozen public
instance count enter every configuration hash. Aggregation refuses observed-only
test results, any missing physical shard, or any logical cell that does not
cover the exact contiguous frozen range.

## Checkpoint, resume, and reuse

Each leaf output contains:

```text
result.jsonl
manifest.json
run.log
failures.jsonl
checkpoint_manifest.json
checkpoints/<episode-id>.json
```

Every successful episode is atomically checkpointed under a stable ID bound to
the scientific configuration, sample index, dataset split, and dataset
revision. If a task is interrupted, rerun the same `SLURM_ARRAY_TASK_ID` with
the same preset, environment, source tree, and `OUT_ROOT`. Completed IDs are
validated and skipped; only missing IDs execute. A failure is appended to
`failures.jsonl` rather than silently repaired.

If a process stops after the durable episode file is renamed but before its
running manifest index is updated, resume validates that episode and repairs
the index. Missing declared episodes, unknown IDs, incompatible hashes, and
every disagreement after a manifest is complete remain hard failures.

A completed compatible result is strictly validated and reused. An incomplete
checkpoint from a different configuration or source tree is rejected. If code
or scientific settings changed, use a new output root. `OVERWRITE=1` removes
only the known artifacts for the selected leaf and starts it again; do not set
it across a broad array unless replacing every selected member is intentional.

Scientific results stay under `OUT_ROOT`. When SLURM provides `SLURM_TMPDIR`,
the launcher uses a task-specific subdirectory there only as `TMPDIR`; final
results and the model cache are not placed exclusively in ephemeral scratch.

## Aggregation outputs and validation

The aggregate stage writes under `$OUT_ROOT/eN/aggregate/`:

```text
grid_manifest.jsonl
validated_run_index.json
summary.json
summary.csv
paired_comparisons.csv
analysis_manifest.json
```

E1 additionally writes `accuracy_vs_q.csv`, `auc_vs_q.csv`,
`entropy_reduction_vs_q.csv`, and `heldout_monitor_fidelity_vs_q.csv`. E2
writes `state_prediction.csv`, `matched_exploit_rate.csv`,
`monitor_acceptance.csv`, `prohibited_effect_rate.csv`,
`feedback_assisted_gain.csv`, and `causal_chain_table.csv`.

Aggregation validates result and failure digests, complete checkpoint state,
scientific hashes, exact `{configuration_hash, shard_id}` membership, grid
hash, provenance compatibility, and matched cohorts. Use a dedicated
`OUT_ROOT` per grid: stale results from another grid are rejected, not ignored.
After that validation, contiguous physical shards are combined into one logical
treatment cohort before task-clustered intervals and paired contrasts are
computed; every source leaf remains indexed. Decoding seeds stay separate
replication strata. The analysis manifest records the aggregation code revision
separately from trajectory generation, the shard-to-cohort mapping, 5,000
public-task cluster bootstrap settings, preregistered contrasts, and
criterion-level G0–G4 evidence. Missing pilot cells are marked
`not_evaluated`, never silently passed.

For Tier-2 analyses, `analysis_manifest.json` keeps requested and resolved
model/tokenizer identities separate and reports tokens, model-call latency,
total trial wall time, retries, failures, and a single-device model-call-time
proxy in hours. That proxy is not billed GPU-hours: it excludes model loading,
scheduler queue and idle allocation time, and cluster pricing.

For the E1 shuffled control, aggregation first averages correctness within each
public task (four shuffled target/donor rows versus two genuine Q=0 rows), then
pairs those task means. G2 passes this criterion only when the complete 95%
paired task-cluster bootstrap interval lies inside the margin loaded from
`configs/silenttwin/analysis-v1.json`; a missing Q=0 match remains
`not_evaluated`.

## E3–E5 compatibility entrypoints

E3–E5 retain the same stage contract and now use exact grid manifests during
aggregation. Inspect their current ranges instead of copying static counts:

```bash
E3_STAGE=grid bash experiments/silenttwin/run_experiment_3_channel_closure.sh
E4_STAGE=grid bash experiments/silenttwin/run_experiment_4_useful_work.sh
E5_STAGE=grid bash experiments/silenttwin/run_experiment_5_assumption_ablations.sh
```

Two-row local examples remain in the comments at the top of each script. For
SLURM, create `logs/`, use the printed range with `--array`, and repeat the same
environment during aggregation.
