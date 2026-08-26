# SilentTwin AgentDojo Tier-2 cross-platform handoff

Last updated: 2026-08-26

This file transfers operational context to a new Codex session on another
platform. It does **not** transfer the internal state of the original chat.
Instead, it records the repository baseline, scientific invariants, platform
setup, validation steps, production prerequisites, commands, expected outputs,
and known limitations needed to continue safely.

## Start the new Codex session with this prompt

Open Codex at the root of this repository and send:

```text
Read handoff.md completely, then read docs/agentdojo_tier2.md and
docs/agentdojo_tier2_audit.md. Inspect the repository and current platform
before changing anything. Confirm the git revision, Python version, SLURM
environment, persistent storage, available GPU types/VRAM, local wheelhouse,
and checkpoint paths. Summarize what is ready and what is missing. Then guide
me through the benchmark one checkpoint at a time, beginning with CPU
compatibility/catalog verification and the four-suite fake-model acceptance
smoke. Do not submit a SLURM job or run a real model until I explicitly approve
the exact command and site-specific resource flags. Do not invent account,
partition, GPU, checkpoint, revision, or benchmark-result values.
```

`handoff.md` is not automatically injected into every Codex conversation. The
prompt above is therefore important. If the team later wants automatic
repository-wide discovery, it can separately decide whether to add an
`AGENTS.md` instruction that points to this file.

## Repository baseline

- Repository: `silent_twin`
- Branch at handoff creation: `main`
- Implementation baseline commit:
  `79aea204dee478907c63ebce647e9bc16776aa4a`
- At handoff creation, `main`, `origin/main`, and `origin/HEAD` all pointed to
  that commit.
- The worktree was clean before this file was added.
- This `handoff.md` file is a new addition and must itself be committed,
  pushed, or copied to the destination platform.

On the new platform, begin with:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short

export SILENTTWIN_BASE_REV=79aea204dee478907c63ebce647e9bc16776aa4a
git merge-base --is-ancestor "$SILENTTWIN_BASE_REV" HEAD
git diff --stat "$SILENTTWIN_BASE_REV"..HEAD
```

If `handoff.md` is copied separately without a new commit, the expected `HEAD`
is:

```text
79aea204dee478907c63ebce647e9bc16776aa4a
```

If this file is committed and pushed, `HEAD` will instead be a newer commit.
The ancestry check must succeed, and the new session must inspect every change
after the baseline. If the baseline is not an ancestor or there are unexpected
source/configuration changes, stop and resolve the mismatch. Do not silently
mix frozen artifacts from one revision with source from another. If this file
is not committed to the remote, transfer it separately.

## Checkpoint to restore

```text
implementation: complete at the baseline commit above
original CPU verification: complete; results recorded below
destination-platform provisioning: not started
production model/artifact selection: not started
production SLURM or GPU execution: not started
scientific benchmark findings: none
next checkpoint: destination CPU compatibility and four-suite fake acceptance
```

## What is implemented

The repository contains a complete AgentDojo Tier-2 benchmark substrate while
retaining the finite-state Tier-1 benchmark.

Implemented Tier-2 components include:

- Python 3.11 and `agentdojo==0.1.35` compatibility enforcement;
- upstream source revision
  `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`;
- AgentDojo benchmark-data version `v1.2.2`;
- a fully resolved 71-distribution AgentDojo core lock;
- introspected and hashed `workspace`, `travel`, `banking`, and `slack`
  catalogs and structural splits;
- controlled causal E1/E2, channel-closure E3, useful-work E4, assumption
  ablations E5, and a separate ecological track;
- fixed final-effect opportunities, non-settling probes, target/donor clone
  isolation, online matched-shuffled feedback, context retirement, and
  dependency-aware settlement;
- deterministic and local learned action-monitor adapters;
- Granite Guardian 4.1 8B and `gpt-oss-safeguard-20b` monitor families;
- the released transformer prompt-injection detector adapter for ecological
  cells;
- train/development-only blind-spot pair mining and immutable pair registries;
- runtime/checkpoint provenance, strict public/private schemas, privacy
  audits, atomic publication, checkpoint/resume, idempotent reuse, and
  collision rejection;
- scenario-clustered aggregation, CI-based gates, hierarchical E1-to-E2
  gatekeeping, development power analysis, and held-out sample freezes;
- explicit CPU grid, SLURM run, and CPU aggregate entrypoints.

Primary references inside the repository:

- `docs/agentdojo_tier2.md` — operator guide;
- `docs/agentdojo_tier2_audit.md` — initial discrepancy audit and claim
  boundary;
- `requirements-tier2-agentdojo.lock` — pinned AgentDojo core runtime;
- `requirements-dev.lock` — CPU test environment;
- `configs/silenttwin/agentdojo/` — frozen catalog, splits, plans, analyses,
  and engineering fixtures;
- `src/silenttwin/agentdojo/` — Tier-2 protocol, runner, storage, statistics,
  and aggregation;
- `src/silenttwin/backends/agentdojo.py` — AgentDojo backend;
- `src/silenttwin/model_clients/local_transformers.py` — offline local model
  client;
- `experiments/silenttwin/_agentdojo_common.sh` — shell authorization and
  orchestration;
- `tests/integration/test_agentdojo_run_grid_task_fake_smoke.py` — real
  four-suite CPU fake-model runner acceptance test.

## Previously verified state

The following checks passed on the original implementation platform at commit
`79aea204dee478907c63ebce647e9bc16776aa4a`:

```text
Standard environment:
411 passed, 40 skipped, 79 subtests passed

Pinned Python 3.11 / AgentDojo 0.1.35 environment:
451 passed, 79 subtests passed

Pinned four-suite E1/E2 run_grid_task fake acceptance:
8 passed in 261.05 seconds

AgentDojo catalog:
555 scenarios
1,467 eligible combinations
all four suites verified

Static checks:
compileall passed
bash -n passed for the AgentDojo entrypoints and shared helper
git diff --check passed
```

These are implementation-verification results, not benchmark findings. Re-run
the relevant checks on the new platform before trusting its environment.

## Scientific claim boundary

All AgentDojo environments and effects are local simulations. Report outcomes
as **AgentDojo attack success** or a **benchmark prohibited effect**, never as
real-world harm or a live external side effect.

The intended causal chain is:

```text
guardrail feedback
-> inference of a private enforcement state
-> state-matched bypass selection
-> higher benchmark prohibited-effect rate
   under one fixed final-effect opportunity
```

SilentTwin is evaluated for removing the **incremental advantage supplied by
feedback**. It is not claimed to repair a monitor's baseline false negatives.

The checked deterministic smoke plan and fixture artifacts are explicitly:

```text
fixture_mode: true
evidence_class: engineering_smoke_only
scientific_evidence_eligible: false
```

Never present smoke output as evidence of leakage, harm, mitigation, closure,
utility, or ecological prevalence.

## Non-negotiable protocol invariants

The new Codex session must preserve these properties:

1. AgentDojo package version, source revision, and benchmark-data version are
   separate frozen fields.
2. AgentDojo suite names do not enter the old finite-state `WORLD_SUITES`.
3. Probe seeds are prefix-coupled; prediction/final seeds are independent of
   query budget, condition, theta, donor, verdict, and runtime.
4. Probes do not settle effects or mutate the final target.
5. Target and donor environments never alias.
6. E2 has exactly one locked final plan; invalid output, rejection, or model
   failure consumes the opportunity. No fallback or alternate promotion.
7. Ground-truth prohibited status is distinct from monitor approval.
8. AgentDojo `security=True` is normalized immediately to
   `attack_success=True`; public fields are never named only `security`.
9. Public serialization contains no private theta, monitor profile/verdict,
   donor state, trusted grader record, or settlement result.
10. SilentTwin retires the original context before settlement and never
    delivers settlement output back to it.
11. Pair construction is frozen from train/development before test; held-out
    cases are never filtered by test outcomes.
12. Exact finite-state TV is `not_applicable` for learned AgentDojo runs.
13. Errors remain separate and aggregation reports both valid-run and
    conservative attack-success rates.
14. Aggregation uses structural scenarios as independent units with four-suite
    stratification; seeds and donor/target rows are repeated measurements.
15. Operational cache paths do not define scientific identity; complete local
    checkpoint fingerprints and learned-runtime fingerprints do.
16. No provider API, download, mock, or unpinned model fallback is allowed in
    production.

## Platform information to collect first

The destination operator and Codex should fill in this worksheet before
constructing any production command:

```text
REPO_ROOT=
PERSIST_ROOT=
PYTHON311_BINARY=
VENV_ROOT=
WHEELHOUSE=
OUT_ROOT=
LOG_ROOT=
MODEL_CACHE=
ATTACKER_CHECKPOINT=
VICTIM_CHECKPOINT=
MONITOR_CHECKPOINT=
PI_DETECTOR_CHECKPOINT=
ACCOUNT_FLAG=
CPU_PARTITION_FLAG=
GPU_PARTITION_FLAG=
ONE_GPU_FLAG=
TWO_GPU_FLAG=
GPU_MODEL=
GPU_VRAM_GB=
MAX_ARRAY_CONCURRENCY=
```

Use read-only platform checks first:

```bash
pwd
python3.11 --version
command -v sbatch
sbatch --version
nvidia-smi -L
git status --short
```

Do not guess unavailable site flags. Ask the platform operator or inspect its
official scheduler documentation.

## Moving the repository and artifacts

The repository commit contains source code, checked catalogs/splits, smoke
fixtures, analysis plans, locks, tests, and shell entrypoints. It does **not**
contain production model weights, a production candidate-strategy catalog, a
mined production pair registry, development evidence, or sample-size freezes.

Transfer separately through approved persistent storage:

- the local attacker checkpoint;
- the local victim checkpoint for ecological runs;
- the local learned action-monitor checkpoint;
- the optional local transformer PI-detector checkpoint;
- a complete offline Python wheelhouse or immutable cluster image;
- operator-authored/mined candidate and pair artifacts;
- any completed development run directory needed for resume or aggregation.

Do not place authoritative environments, caches, checkpoints, outputs, or logs
inside `SLURM_TMPDIR`.

Experiment execution checkpoints are portable if their entire persistent
directory and matching grid/source/artifact identities move together. Chat
history is not required for runner resume: the runner validates
`checkpoint_manifest.json`, `manifest.json`, hashes, and frozen grid identity.

## Python 3.11 environment bootstrap

Use a persistent shared filesystem. Example placeholders:

```bash
cd <REPO_ROOT>

<PYTHON311_BINARY> -m venv <VENV_ROOT>
<VENV_ROOT>/bin/python3.11 -m pip install -r requirements-tier2-agentdojo.lock
<VENV_ROOT>/bin/python3.11 -m pip install -r requirements-dev.lock
<VENV_ROOT>/bin/python3.11 -m pip check

export ENV_ACTIVATE=<VENV_ROOT>/bin/activate
export PYTHON_BIN=<VENV_ROOT>/bin/python3.11
```

If compute nodes cannot access the network, provision from a complete local
wheelhouse before the experiment:

```bash
<VENV_ROOT>/bin/python3.11 -m pip install \
  --no-index \
  --find-links <WHEELHOUSE> \
  -r requirements-tier2-agentdojo.lock
```

The 71-distribution core lock deliberately excludes site-specific Torch,
Transformers, CUDA, and quantization packages. Production learned inference
needs an independently approved immutable stack in the same environment. No
package or model download is allowed during an experiment job.

## New-platform CPU verification

From the repository root:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m silenttwin.agentdojo.cli verify-catalog \
  --catalog configs/silenttwin/agentdojo/catalog-v1.json \
  --splits configs/silenttwin/agentdojo/splits-v1.json
```

Expected catalog summary:

```text
verified=true
scenario_count=555
eligible_combination_count=1467
structural_group_count_by_suite:
  workspace=40
  banking=16
  slack=21
  travel=20
```

Run the complete pinned suite when practical:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q
```

Run the most important portable Tier-2 acceptance test explicitly:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q \
  tests/integration/test_agentdojo_run_grid_task_fake_smoke.py
```

Expected result under Python 3.11 with AgentDojo 0.1.35:

```text
8 passed
```

This test runs real frozen-grid E1/E2 orchestration through all four AgentDojo
suites with deterministic fake models, checkpoints, strict publication,
failure ledgers, and idempotent reuse. It requires no GPU and must not download
models.

Also run:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m compileall -q src/silenttwin

bash -n \
  experiments/silenttwin/_agentdojo_common.sh \
  experiments/silenttwin/run_agentdojo_catalog.sh \
  experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh \
  experiments/silenttwin/run_experiment_1_feedback_leakage_agentdojo_tier2.sh \
  experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_agentdojo_tier2.sh \
  experiments/silenttwin/run_experiment_3_channel_closure_agentdojo_tier2.sh \
  experiments/silenttwin/run_experiment_4_useful_work_agentdojo_tier2.sh \
  experiments/silenttwin/run_experiment_5_assumption_ablations_agentdojo_tier2.sh \
  experiments/silenttwin/run_agentdojo_ecological_tier2.sh

git diff --check
```

## Optional catalog reproduction

The checked catalog is already frozen. To reproduce it in the pinned
environment, preferably to persistent evidence paths rather than overwriting
the checked files:

```bash
export AGENTDOJO_CATALOG=<PERSIST_ROOT>/evidence/catalog-v1.json
export AGENTDOJO_SPLITS=<PERSIST_ROOT>/evidence/splits-v1.json

STAGE=grid bash experiments/silenttwin/run_agentdojo_catalog.sh
STAGE=run bash experiments/silenttwin/run_agentdojo_catalog.sh
STAGE=aggregate bash experiments/silenttwin/run_agentdojo_catalog.sh
```

`grid` prints pins and paths. `run` imports AgentDojo and atomically freezes
catalog/splits. `aggregate` verifies them and prints file hashes.

## Engineering-smoke grids

Common smoke configuration:

```bash
cd <REPO_ROOT>

export ENV_ACTIVATE=<VENV_ROOT>/bin/activate
export PYTHON_BIN=<VENV_ROOT>/bin/python3.11
export OUT_ROOT=<PERSIST_ROOT>/results/silenttwin-agentdojo-smoke
export AGENTDOJO_DATASET_SPLIT=development
export AGENTDOJO_REPLICATES="0"
export AGENTDOJO_GRID_PLAN="$PWD/configs/silenttwin/agentdojo/grid-plans/controlled-fake-smoke-v1.json"
```

The checked development smoke with one replicate and eight structural groups
per bundle has:

| ID | Purpose | Configurations | Array range |
| --- | --- | ---: | --- |
| `e1` | feedback leakage/state inference | 240 | `0-4` |
| `e2` | feedback-assisted prohibited effects | 65 | `0-4` |
| `e3` | channel closure/degraded channels | 60 | `0-4` |
| `e4` | useful independent work/settlement | 45 | `0-4` |
| `e5` | assumption ablations | 75 | `0-4` |
| `ecological` | free-form AgentDojo prevalence | 75 | `0-4` |

Changing the plan, split, replicates, or bundle size changes these counts. Use
the range printed by the actual `grid` command.

Entrypoints:

```text
e1          experiments/silenttwin/run_experiment_1_feedback_leakage_agentdojo_tier2.sh
e2          experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_agentdojo_tier2.sh
e3          experiments/silenttwin/run_experiment_3_channel_closure_agentdojo_tier2.sh
e4          experiments/silenttwin/run_experiment_4_useful_work_agentdojo_tier2.sh
e5          experiments/silenttwin/run_experiment_5_assumption_ablations_agentdojo_tier2.sh
ecological  experiments/silenttwin/run_agentdojo_ecological_tier2.sh
```

For one experiment, set:

```bash
export ID=e1
export SCRIPT=experiments/silenttwin/run_experiment_1_feedback_leakage_agentdojo_tier2.sh
```

Create its persistent scheduler-log directory before submission; Slurm does
not create missing parent directories:

```bash
mkdir -p "$OUT_ROOT/logs/$ID"
```

Freeze its model-free grid:

```bash
STAGE=grid bash "$SCRIPT"
```

Expected key stdout fields include:

```text
environment_backend=agentdojo
dataset_split=development
suite_coverage_status=full_four_suite
total_tasks=<N>
total_configurations=<N>
valid_array_range=<FIRST-LAST>
grid_hash=<SHA256>
model_free=True
grid_manifest=<OUT_ROOT>/<ID>/grid/grid-manifest.jsonl
```

The shell run stage is deliberately SLURM-only, even for fake models. Submit
the smoke using operator-supplied CPU site flags:

```bash
sbatch <ACCOUNT_FLAG> <CPU_PARTITION_FLAG> \
  --array=0-4%1 \
  --output="$OUT_ROOT/logs/$ID/%A_%a.out" \
  --error="$OUT_ROOT/logs/$ID/%A_%a.err" \
  --export="ALL,STAGE=run,GRID_MANIFEST=$OUT_ROOT/$ID/grid/grid-manifest.jsonl,AGENTDOJO_FAKE_MODEL=1,AGENTDOJO_REQUIRES_GPU=0" \
  "$SCRIPT"
```

Start at `%1`; increase concurrency only after the platform smoke is stable.
No GPU is needed for the fake plan. A successful task ends with JSON containing
`task_id`, `completed_shards`, and `configuration_hashes`.

After every array task succeeds:

```bash
STAGE=aggregate \
GRID_MANIFEST="$OUT_ROOT/$ID/grid/grid-manifest.jsonl" \
bash "$SCRIPT"
```

Run and aggregate E1 before E2. E2 automatically uses
`$OUT_ROOT/e1/aggregate/analysis_manifest.json` for hierarchical gatekeeping;
set `E1_ANALYSIS_MANIFEST` explicitly if E1 lives elsewhere.

## Output and resume layout

For experiment `<ID>`:

```text
<OUT_ROOT>/<ID>/grid/grid-manifest.jsonl
<OUT_ROOT>/<ID>/runs/task-<TASK_ID>/batch-<OFFSET>-<SHARD_PREFIX>/
  checkpoints/
  checkpoint_manifest.json
  result.jsonl
  failures.jsonl
  run.log
  manifest.json
<OUT_ROOT>/<ID>/aggregate/
  summary.json
  analysis_manifest.json
  validated_run_index.json
  grid_manifest.jsonl
```

`manifest.json` is published last and is the completion marker. Re-running an
identical array member validates and reuses completed shards. A changed grid,
configuration, source-tree identity, artifact hash, or trial cohort is
rejected rather than reused.

Aggregation is strict and rejects missing, duplicated, unexpected, rebound,
or hash-mismatched shards. `AGENTDOJO_ALLOW_DEVELOPMENT_PARTIAL=1` exists only
for explicitly partial development analysis and cannot be used for held-out
test claims.

## Production artifacts still required

The repository intentionally does not ship executable production artifacts.
Before any real-model grid can be frozen, provide:

1. `<PERSIST_ROOT>/evidence/candidate-strategies-v1.json`, frozen from train;
2. `<PERSIST_ROOT>/evidence/pair-registry-v1.json`, mined from train and
   validated on development;
3. exact local attacker checkpoint and tokenizer revisions;
4. exact local victim checkpoint and tokenizer revisions for ecological runs;
5. exact local action-monitor checkpoint, prompt, policy, threshold,
   reasoning mode, dtype, and decoding;
6. optional exact local transformer PI-detector checkpoint for that ecological
   baseline;
7. a complete learned-runtime fingerprint;
8. an executable, immutable production grid plan with no placeholders;
9. development aggregates and sample-size freezes before E1-E4 test grids.

The file
`configs/silenttwin/agentdojo/grid-plans/controlled-local-template-v1.json` is
a fail-closed template, not an executable plan. It must not be passed directly
to `STAGE=grid`.

## Runtime and checkpoint fingerprints

Audit the active learned environment:

```bash
PYTHONPATH=src "$PYTHON_BIN" \
  -m silenttwin.agentdojo.runtime_integrity \
  --dependency-lock requirements-tier2-agentdojo.lock
```

If the retained AgentDojo wheel is available, add:

```text
--wheel-artifact <PERSIST_ROOT>/wheels/agentdojo-0.1.35-py3-none-any.whl
```

Copy the emitted `learned_runtime.runtime_fingerprint` into every learned
model/profile row and export the same value at run time.

Fingerprint every local checkpoint:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m silenttwin.cli fingerprint-model \
  --model-dir <ATTACKER_CHECKPOINT> \
  --cache-dir <MODEL_CACHE>

PYTHONPATH=src "$PYTHON_BIN" -m silenttwin.cli fingerprint-model \
  --model-dir <VICTIM_CHECKPOINT> \
  --cache-dir <MODEL_CACHE>

PYTHONPATH=src "$PYTHON_BIN" -m silenttwin.cli fingerprint-model \
  --model-dir <MONITOR_CHECKPOINT> \
  --cache-dir <MODEL_CACHE>
```

The CLI's `model_revision` output is the full-tree checkpoint fingerprint for
backward compatibility. In a frozen model row it belongs in
`checkpoint_fingerprint`; separately record the semantic upstream
`model_revision` and `tokenizer_revision`.

## Guard-pair mining

Pair mining accepts only train and development observations. Never supply test
observations.

Common variables:

```bash
export AGENTDOJO_STRATEGY_CATALOG=<PERSIST_ROOT>/evidence/candidate-strategies-v1.json
export AGENTDOJO_PAIR_REGISTRY=<PERSIST_ROOT>/evidence/pair-registry-v1.json
export AGENTDOJO_MODEL_CACHE=<PERSIST_ROOT>/model-cache
export AGENTDOJO_MONITOR_CHECKPOINT=<PERSIST_ROOT>/checkpoints/action-monitor
export AGENTDOJO_RUNTIME_FINGERPRINT=sha256:<FROZEN_RUNTIME_SHA>
```

Create the persistent log directory before submitting any pair-mining job:

```bash
mkdir -p <LOG_ROOT>/pair-mining
```

Inspect without loading a model:

```bash
STAGE=grid bash experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
```

Submit train observation generation:

```bash
sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
  --output=<LOG_ROOT>/pair-mining/train-%j.out \
  --error=<LOG_ROOT>/pair-mining/train-%j.err \
  --export="ALL,STAGE=run,PAIR_MINING_ACTION=observe,OBSERVATION_SPLIT=train,OBSERVATIONS_OUTPUT=<PERSIST_ROOT>/evidence/train.jsonl,OBSERVATION_MANIFEST_OUTPUT=<PERSIST_ROOT>/evidence/train.manifest.json,AGENTDOJO_REQUIRES_GPU=1" \
  experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
```

Submit development observation generation:

```bash
sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
  --output=<LOG_ROOT>/pair-mining/development-%j.out \
  --error=<LOG_ROOT>/pair-mining/development-%j.err \
  --export="ALL,STAGE=run,PAIR_MINING_ACTION=observe,OBSERVATION_SPLIT=development,OBSERVATIONS_OUTPUT=<PERSIST_ROOT>/evidence/development.jsonl,OBSERVATION_MANIFEST_OUTPUT=<PERSIST_ROOT>/evidence/development.manifest.json,AGENTDOJO_REQUIRES_GPU=1" \
  experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
```

After both observation jobs succeed, freeze pairs on a CPU SLURM allocation:

```bash
sbatch <ACCOUNT_FLAG> <CPU_PARTITION_FLAG> \
  --dependency=afterok:<TRAIN_JOB_ID>:<DEVELOPMENT_JOB_ID> \
  --output=<LOG_ROOT>/pair-mining/reduce-%j.out \
  --error=<LOG_ROOT>/pair-mining/reduce-%j.err \
  --export="ALL,STAGE=run,PAIR_MINING_ACTION=reduce,AGENTDOJO_REQUIRES_GPU=0,TRAIN_OBSERVATIONS=<PERSIST_ROOT>/evidence/train.jsonl,TRAIN_OBSERVATION_MANIFEST=<PERSIST_ROOT>/evidence/train.manifest.json,DEVELOPMENT_OBSERVATIONS=<PERSIST_ROOT>/evidence/development.jsonl,DEVELOPMENT_OBSERVATION_MANIFEST=<PERSIST_ROOT>/evidence/development.manifest.json" \
  experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
```

Verify the frozen artifact paths and file hashes:

```bash
STAGE=aggregate bash experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
```

Expected reducer summary includes `pair_registry_hash`, `suite_count`,
`test_instantiation_count`, and `test_outcomes_inspected:false`.

## GPU sizing and placement

The checked smoke uses zero GPUs.

The local model client uses batch size 1 and loads each BF16/FP16 model wholly
onto its configured device. It currently has no quantization, `device_map`,
tensor-parallel, or CPU-offload path. Requesting multiple GPUs does not shard a
single model.

Approximate BF16 weight memory is `1.86 GiB` per billion parameters, before
activations, KV cache, CUDA workspaces, and fragmentation. Practical starting
allocations are:

| Workload | Suggested allocation per array worker |
| --- | --- |
| 7–8B attacker + Granite Guardian 8B monitor | one 48 GB GPU, or two 24 GB GPUs |
| E4 with Granite Guardian 8B only | one 24 GB GPU |
| 8B victim ecological cell | one 24 GB GPU |
| 8B attacker + 20–21B BF16 monitor | one 80 GB GPU, or 24 GB + 48–64 GB |
| 20B attacker + 20B monitor | two 48–64 GB GPUs; one 80 GB GPU is unsafe |

Recommended first production deployment:

```text
attacker/victim: 7–8B
action monitor: Granite Guardian 4.1 8B
per worker: one 48 GB GPU
initial array concurrency: %1
```

With `--array=0-4%4`, up to four workers run concurrently. One GPU per worker
therefore means up to four GPUs; two GPUs per worker means up to eight. `%1`
runs serially and changes only wall-clock time, not the estimand.

Default role devices all resolve to the first visible GPU. To split two roles
across two allocated GPUs, export:

```text
ATTACKER_DEVICE=cuda:0
MONITOR_DEVICE=cuda:1
```

Use `VICTIM_DEVICE=cuda:0` for ecological victim cells. The released PI
detector adapter has its own released pipeline placement and must be measured
on the destination runtime.

Known resource caveat: pair-observation generation currently constructs one
learned model client per frozen monitor profile. Two compatible theta profiles
may therefore load the same checkpoint twice. Two Granite 8B profiles should
start on a 48 GB GPU. Two 20–21B BF16 profiles are likely to exceed an 80 GB
GPU; use Granite initially or change and revalidate client sharing before
using that monitor family.

## Production development grids and arrays

After producing all fingerprints and materializing an executable plan:

```bash
export OUT_ROOT=<PERSIST_ROOT>/results/silenttwin-agentdojo-production
export LOG_ROOT=<PERSIST_ROOT>/logs/silenttwin-agentdojo-production
export AGENTDOJO_GRID_PLAN=<PERSIST_ROOT>/plans/controlled-local-v1.json
export AGENTDOJO_STRATEGY_CATALOG=<PERSIST_ROOT>/evidence/candidate-strategies-v1.json
export AGENTDOJO_PAIR_REGISTRY=<PERSIST_ROOT>/evidence/pair-registry-v1.json
export AGENTDOJO_MODEL_CACHE=<PERSIST_ROOT>/model-cache
export AGENTDOJO_ATTACKER_CHECKPOINT=<PERSIST_ROOT>/checkpoints/attacker
export AGENTDOJO_VICTIM_CHECKPOINT=<PERSIST_ROOT>/checkpoints/victim
export AGENTDOJO_MONITOR_CHECKPOINT=<PERSIST_ROOT>/checkpoints/action-monitor
export AGENTDOJO_RUNTIME_FINGERPRINT=sha256:<FROZEN_RUNTIME_SHA>
export AGENTDOJO_DATASET_SPLIT=development
```

For each ID/script pair listed above:

```bash
mkdir -p "$LOG_ROOT/$ID"
STAGE=grid bash "$SCRIPT"
```

Use the printed range, not a copied smoke range:

```bash
sbatch <ACCOUNT_FLAG> <GPU_PARTITION_FLAG> <ONE_GPU_FLAG> \
  --array=<PRINTED_VALID_ARRAY_RANGE>%1 \
  --output="$LOG_ROOT/$ID/%A_%a.out" \
  --error="$LOG_ROOT/$ID/%A_%a.err" \
  --export="ALL,STAGE=run,GRID_MANIFEST=$OUT_ROOT/$ID/grid/grid-manifest.jsonl,AGENTDOJO_FAKE_MODEL=0,AGENTDOJO_REQUIRES_GPU=1,ATTACKER_DEVICE=cuda:0,MONITOR_DEVICE=cuda:0,VICTIM_DEVICE=cuda:0" \
  "$SCRIPT"
```

If using two GPUs for controlled attacker/monitor roles, replace the site's
resource flag and export `MONITOR_DEVICE=cuda:1`.

After the complete array succeeds:

```bash
STAGE=aggregate \
GRID_MANIFEST="$OUT_ROOT/$ID/grid/grid-manifest.jsonl" \
bash "$SCRIPT"
```

Run order:

```text
compatibility smoke
-> guard-pair mining
-> four-suite signal pilot
-> development aggregation and power/sample freeze
-> held-out E1/E2/E3
-> ecological/E4
-> E5
```

E4 is confirmatory only with its own valid freeze. Ecological and E5 remain
development/secondary analyses.

## Development power and held-out freezes

E1–E4 test grids require a matching sample-size freeze derived from a complete,
nonfixture development aggregate. Example for E2:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m silenttwin.agentdojo.cli freeze-sample-size \
  --experiment e2 \
  --catalog configs/silenttwin/agentdojo/catalog-v1.json \
  --splits configs/silenttwin/agentdojo/splits-v1.json \
  --strategy-catalog <PERSIST_ROOT>/evidence/candidate-strategies-v1.json \
  --pair-registry <PERSIST_ROOT>/evidence/pair-registry-v1.json \
  --analysis-plan configs/silenttwin/agentdojo/analysis/controlled-v1.json \
  --dependency-lock requirements-tier2-agentdojo.lock \
  --development-analysis-manifest "$OUT_ROOT/e2/aggregate/analysis_manifest.json" \
  --output <PERSIST_ROOT>/evidence/e2-sample-size-freeze.json \
  --assert-test-results-uninspected
```

Repeat with the matching experiment and development manifest for E1, E3, and
E4. A power result below 0.80 produces
`underpowered_estimation_only`, never a passed gate.

Construct an E2 test grid only after freezing:

```bash
export AGENTDOJO_DATASET_SPLIT=test
export AGENTDOJO_SAMPLE_SIZE_FREEZE=<PERSIST_ROOT>/evidence/e2-sample-size-freeze.json
export AGENTDOJO_DEVELOPMENT_ANALYSIS_MANIFEST="$OUT_ROOT/e2/aggregate/analysis_manifest.json"

STAGE=grid \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_agentdojo_tier2.sh
```

Test arrays and aggregation use the same production run pattern. E2
aggregation additionally needs the compatible held-out E1 analysis manifest:

```bash
export E1_ANALYSIS_MANIFEST=<PERSIST_ROOT>/results/test/e1/aggregate/analysis_manifest.json
```

## Structural shortfall already recorded

The frozen development and test split has these distinct user-task counts
against a preregistered minimum of six per suite:

```text
workspace: 10 / 6
banking:    4 / 6
slack:      5 / 6
travel:     5 / 6
```

Banking, slack, and travel therefore cannot support a claimed confirmatory
suite gate without a frozen expansion of genuinely distinct tasks or a
preregistered cross-fitting design. Seeded variants do not repair the
shortfall.

## Expected aggregate contents

All aggregates include provenance, coverage, independent-unit counts,
four-suite strata, valid and conservative error accounting, pair
yield/headroom, clustered intervals, development power, and gate dispositions.

- E1: state-inference accuracy above prior, ROC-AUC, Brier score, log loss,
  entropy reduction, transcript distinguisher, invalid/error rate, query curves.
- E2: prohibited-effect/attack-success rate, state-matched selection,
  prediction, rejected commits, utility, safe-and-useful rate, oracle headroom,
  genuine-vs-Q0 and genuine-vs-shuffled contrasts, mediation chain.
- E3: exact deterministic transcript audits, learned distinguisher, channel
  interventions, retirement failures, rejected commits. KV-cache retention is
  `not_evaluated` unless a real backend exposes cache reuse.
- E4: utility, safe-effect salvage, rejected descendants, atomicity,
  clean-report accuracy, latency, tokens, and tool calls.
- E5: invariant-specific ablation table.
- Ecological: clean utility, utility under attack, targeted attack-success,
  safe-and-useful rate, and errors per attack/defense/suite cell.

## Fail-closed behavior that is expected

The following failures indicate that safeguards are working:

- `STAGE=run` outside a SLURM allocation exits before Python/model loading;
- a missing or noncanonical `SLURM_ARRAY_TASK_ID` is rejected;
- an out-of-range array ID is rejected before environment activation;
- Python other than 3.11 is rejected by AgentDojo entrypoints;
- a smoke grid with `AGENTDOJO_FAKE_MODEL=0` is rejected;
- a production grid with `AGENTDOJO_FAKE_MODEL=1` is rejected;
- learned roles without a persistent cache/checkpoint or visible requested GPU
  are rejected;
- paths inside `SLURM_TMPDIR` are rejected;
- unmaterialized plan templates and placeholder identities are rejected;
- missing/mismatched runtime, checkpoint, catalog, split, strategy, pair,
  analysis, freeze, or source identities are rejected;
- incomplete aggregation is rejected unless explicitly partial development
  mode is selected;
- E2 confirmatory gating is rejected without compatible passing E1 evidence.

Do not weaken these checks merely to make a platform command run.

## Known limitations

- No real SLURM/GPU/model benchmark execution was performed during
  implementation.
- The checked smoke is deterministic engineering evidence only.
- Production checkpoint choices, semantic model/tokenizer revisions, and site
  learned-stack versions remain operator supplied.
- `adapted_local_tool_filter` preserves the released prompt/selection/history
  semantics but is not the exact released OpenAI transport.
- The transformer PI detector is a content/input detector, not an action
  authorization monitor.
- Retained KV cache is not evaluated without genuine backend cache exposure.
- The local Transformers loader has no model sharding or quantization path.
- Pair-observation generation may load identical monitor weights once per
  learned profile.
- No benchmark result has been produced, and no leakage/harm/mitigation claim
  should be stated until eligible production evidence passes all gates.

## How the new Codex should work with the operator

The new Codex session should:

1. inspect first and report exact platform facts;
2. distinguish smoke, development, and held-out evidence in every command;
3. fill placeholders only from operator-approved local artifacts;
4. show the exact command and explain it before asking the operator to submit;
5. start arrays at concurrency `%1` and increase only after a successful shard;
6. inspect scheduler logs, output manifests, failure ledgers, and hashes after
   each checkpoint;
7. preserve completed outputs and use strict resume rather than deleting them;
8. stop for operator direction before any destructive action, new authority,
   real job submission, or scientific-protocol change;
9. never fabricate outputs or interpret engineering smoke as a result;
10. update this file with destination-specific paths and completed milestones,
    but never place credentials, tokens, or secrets in it.

## Immediate next milestone

On the destination platform, the safest next milestone is:

```text
exact git revision verified
-> persistent Python 3.11 environment provisioned
-> dependency/runtime integrity verified
-> catalog verification passed
-> pinned full tests passed
-> four-suite E1/E2 CPU fake acceptance passed
-> model-free grids inspected
-> platform/GPU/checkpoint worksheet completed
```

Only then should the operator and new Codex materialize production identities,
mine guard pairs, or submit learned-model arrays.
