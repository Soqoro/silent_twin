# SilentTwin AgentDojo Tier-2 cross-platform handoff

Last updated: 2026-08-28 (H200 checkpoint-conformance package frozen; commit and submission pending)

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
before changing anything. Confirm the git revision, Python version, scheduler
environment, persistent storage, available GPU types/VRAM, local wheelhouse,
and checkpoint paths. Summarize what is ready and what is missing. Then guide
me through the benchmark one checkpoint at a time, beginning with CPU
compatibility/catalog verification and the four-suite fake-model acceptance
smoke. Do not submit a scheduler job or run a real model until I explicitly approve
the exact command and site-specific resource flags. Do not invent account,
project, queue, partition, GPU, checkpoint, revision, or benchmark-result values.
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
destination Python 3.11/core provisioning: complete; results recorded below
destination catalog/static/full CPU verification: complete
destination checked model-free grid inspection: complete
destination PBS-native authorization support: implemented; validation recorded below
destination E1 engineering-smoke submission: job 53127 attempted; no run shards produced
destination PBS HOME-sandbox correction: implemented and focused validation passed
production model/artifact selection: not started
production scheduler or GPU execution: not started
scientific benchmark findings: none
next checkpoint: commit the PBS HOME-sandbox correction and resubmit E1 job 53127's array
```

## Destination CPU checkpoint completed on 2026-08-26

This section records the first destination-platform session. It supersedes the
earlier `not started` destination fields above without changing the scientific
claim boundary. No scheduler job, GPU workload, real model, pair mining,
production aggregation, or benchmark analysis was run.

### Repository and executable source

- Repository root: `/home/suaq0001/projects/silent_twin`.
- Branch: `main`.
- Verified `HEAD`: `1e682cd301600d390efd305ef24a8ca81f3afb98`.
- `main`, `origin/main`, and `origin/HEAD` all pointed to that revision.
- Baseline `79aea204dee478907c63ebce647e9bc16776aa4a` is an ancestor.
- The only committed post-baseline change was the original addition of this
  `handoff.md`; there were no result-affecting source/config changes.
- Destination LinkRadius-independent SilentTwin `source_tree_hash`:
  `7c781308eb566cb369a593d044e447c195b42249fa5c85c8bd6bf58f4fb2e94e`.
- The checkout was clean before this destination record was added. The
  expected worktree change after this update is `M handoff.md` only;
  generated editable-install metadata is ignored. Commit or transfer this
  documentation update before a production run so provenance does not retain
  an avoidable dirty-worktree marker.

### Platform and persistent paths

- Login host: `hpc-gaas-hn2`, Linux x86_64.
- Scheduler: PBS Professional
  `2025.2.2.20260209105947`, configured server `gaas`; `sbatch` and Slurm are
  absent.
- This login session had no PBS allocation and exposed no GPU. `nvidia-smi -L`
  could not communicate with a driver, as expected on the head node.
- A read-only PBS query showed 25 configured GPU nodes (`hpc-gaas-g01` through
  `hpc-gaas-g25`) with 8 configured GPUs each. PBS does not advertise GPU
  model or VRAM, so neither value is authenticated. Do not infer them from
  node names or memory.
- Persistent root: `/home/suaq0001/projects`, an NFS mount with approximately
  887 GiB free at inspection time.
- Python prefix:
  `/home/suaq0001/projects/.venvs/silenttwin-agentdojo-py311`.
- Python binary:
  `/home/suaq0001/projects/.venvs/silenttwin-agentdojo-py311/bin/python`.
- Engineering-smoke output root:
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-smoke`.
- The prefix was created with Conda and has no venv-style `bin/activate`.
  Leave `ENV_ACTIVATE` unset and use the absolute `PYTHON_BIN`, while placing
  the prefix's `bin` first on `PATH` so shell-test subprocesses also use 3.11.

### Destination runtime verification

- Python `3.11.15`.
- AgentDojo `0.1.35` and pytest `9.1.1`.
- `pip check`: no broken requirements.
- Exact 71-distribution core lock verified; lock SHA-256:
  `0c1da0a4be1b183d243bd308751d3622a09a1553cae2f1ce031dc5e1250a6458`.
- Installed AgentDojo payload verified against the frozen wheel payload
  manifest: 115 files, payload SHA-256
  `bce8c4f279da44fe88e7e69625d0e37a93ba60fda3a057b11b239be2b3b10b77`.
- No retained AgentDojo wheel artifact was available, so the published wheel
  SHA-256 was not independently rehashed from a local wheel file.
- This is a CPU/core verification environment, not an approved learned-model
  runtime. It has no frozen Torch/Transformers/CUDA stack. Also,
  `requirements-dev.lock` installs SilentTwin editably, which exposes both
  `.dist-info` and source `.egg-info` records; the full learned-runtime CLI
  therefore rejects this environment as containing duplicate `silenttwin`
  metadata before checking the missing learned stack. Do not freeze a learned
  runtime fingerprint from this environment. Build a clean non-editable
  learned environment once the operator approves the stack.

### Destination CPU and static verification

The complete test collection was run as two disjoint invocations so the long
acceptance test could be reported explicitly:

```text
all tests except the explicit fake acceptance:
  443 passed, 79 subtests passed in 667.29 seconds

tests/integration/test_agentdojo_run_grid_task_fake_smoke.py:
  8 passed in 354.95 seconds

combined complete collection:
  451 passed, 79 subtests passed
```

The frozen catalog verification passed with:

```text
verified=true
scenario_count=555
eligible_combination_count=1467
structural_group_count_by_suite:
  workspace=40
  banking=16
  slack=21
  travel=20
catalog_hash=d4e4cda9ab44689953852b98a23aec819f5ccc91330fbd145f9fa284591b3015
split_manifest_hash=8aeea9a4fd17304b3d8e02dc4aeaa96687944ce731fb04a8f08e60858c6d4978
```

`compileall`, `bash -n` for all nine documented AgentDojo shell files, and
`git diff --check` also passed.

### Checked engineering-smoke grids

All six development grids were frozen on persistent storage with replicate
`0`, eight structural groups per bundle, full four-suite coverage, array range
`0-4`, and `model_free=True`:

| ID | Configurations | Grid hash |
| --- | ---: | --- |
| `e1` | 240 | `f1614beb27b9f3e8fe51f337744df4fcd544b339a54ff72c32be9238d65f3abc` |
| `e2` | 65 | `316f7e43c7f4741fea2ad01d385c16c53c282a16f94b07e2f3d7f51d54dff5d4` |
| `e3` | 60 | `641d4afc187d4ee8816230fe10ec2e5ff3a9e72a729121ff86de2bfe462d195e` |
| `e4` | 45 | `5f208db10a6b15d16b7b05fc36c2314d920426582e7a39dbbb07d8cbd548e11a` |
| `e5` | 75 | `7a76f6cc5161589a23757e478e3a3d0184189afef79dfa59ee95fb14819aa079` |
| `ecological` | 75 | `4532a870e1426f667ab8a013d67211e2c63b1d663a3a9209f125cdf12991ed72` |

These manifests live under
`/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-smoke/<ID>/grid/`.
They select checked fixtures marked `fixture_mode: true`,
`evidence_class: engineering_smoke_only`, and
`scientific_evidence_eligible: false`. They are not benchmark findings.

### Destination worksheet and next authorization boundary

```text
REPO_ROOT=/home/suaq0001/projects/silent_twin
PERSIST_ROOT=/home/suaq0001/projects
PYTHON311_BINARY=/home/suaq0001/projects/.venvs/silenttwin-agentdojo-py311/bin/python
VENV_ROOT=/home/suaq0001/projects/.venvs/silenttwin-agentdojo-py311
WHEELHOUSE=MISSING
OUT_ROOT=/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-smoke
LOG_ROOT=/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-smoke/logs
MODEL_CACHE=UNSET
ATTACKER_CHECKPOINT=MISSING
VICTIM_CHECKPOINT=MISSING
MONITOR_CHECKPOINT=MISSING
PI_DETECTOR_CHECKPOINT=MISSING
ACCOUNT_OR_PROJECT_FLAG=-P fs_ccds_asysong
CPU_QUEUE_AND_RESOURCE_FLAGS=NO_CPU_ONLY_QUEUE_OBSERVED
GPU_QUEUE_AND_RESOURCE_FLAGS=-q gpu_free
ONE_GPU_RESOURCE_FLAGS=-l select=1:ncpus=12:ngpus=1:mpiprocs=1:mem=250gb
TWO_GPU_RESOURCE_FLAGS=UNAPPROVED_PBS_SITE_VALUE
GPU_MODEL=UNAUTHENTICATED_WITHOUT_ALLOCATION
GPU_VRAM_GB=UNAUTHENTICATED_WITHOUT_ALLOCATION
MAX_ARRAY_CONCURRENCY=1
```

The operator selected PBS/qsub, supplied project `fs_ccds_asysong`, and approved
the first engineering-smoke submission. Live PBS queries authenticated
`gpu_free`, which requires exactly one GPU, permits at most four hours, and
limits this user/project to one running GPU. Existing project launchers and
accepted queue jobs use the chunk shape recorded above. A 30-minute E1 array
using those flags was accepted as job `53127[].gaas`; its diagnostic outcome is
recorded below. Separately, production model checkpoints, a learned stack,
candidate strategies, mined pairs, wheelhouse/image, cache, and production plan
are still missing. Do not select production artifacts without explicit
operator approval.

## Destination PBS adaptation checkpoint on 2026-08-26

The AgentDojo run boundary now supports both PBS Professional and Slurm while
retaining fail-closed authorization. This is an engineering implementation
checkpoint, not scheduler execution or benchmark evidence.

- PBS batch authorization requires `PBS_JOBID` and
  `PBS_ENVIRONMENT=PBS_BATCH`; grid arrays additionally require canonical
  non-negative `PBS_ARRAY_INDEX`.
- Slurm authorization remains supported through the existing `SLURM_JOB_ID`
  and `SLURM_ARRAY_TASK_ID` contract. A mixed PBS/Slurm context is rejected.
- PBS `PBS_ARRAY_INDEX` is bounds-checked against the frozen manifest before
  Python activation, model validation, or GPU inspection.
- Persistent runtime paths are rejected inside Slurm `SLURM_TMPDIR`, PBS
  private-sandbox `PBS_JOBDIR` when it differs from `PBS_O_HOME`, and
  PBS-assigned `TMPDIR`, in both shell and Python preflight. PBS's default HOME
  sandbox is persistent and is not misclassified as scratch.
- Runtime provenance now normalizes PBS job, array, queue, node-file, and CPU
  metadata while retaining Slurm fields.
- Explicit `AGENTDOJO_REPO_ROOT` support lets a PBS-spooled entrypoint resolve
  the authoritative checkout without relying on the spool path.
- The scripts still contain no submission command and no guessed site account,
  project, queue, GPU, memory, or walltime request.
- Post-adaptation executable `source_tree_hash`:
  `f3fe9da0437cc08441888c465fb6077edc16a6bcebac748a7b425b58938d1505`.
- Complete CPU validation passed as two disjoint invocations:
  457 tests plus 79 subtests outside the explicit acceptance in 661.88 seconds,
  and 8 four-suite fake-model acceptance tests in 356.40 seconds. One additional
  PBS missing/noncanonical-index regression was then added and passed in the
  focused scheduler suite. The current collection therefore has 466 validated
  tests plus 79 subtests; the final focused PBS/Slurm subset contained 56
  passing tests.
- `compileall`, `bash -n` for all nine AgentDojo shell files, and
  `git diff --check` passed.
- All six model-free grids were regenerated under `/tmp` and retained the exact
  checked hashes, configuration counts, full-four-suite coverage, and `0-4`
  range recorded above. Persistent grid files were not overwritten.
- The adaptation was committed as `27dc54b036980d98192218a94a86e3cb849cd2c2`
  on top of `1e682cd301600d390efd305ef24a8ca81f3afb98` before the first submission.
- Live read-only PBS queries then authenticated server `gaas`, queue access,
  project GPU limits, and the accepted resource shape. Engineering-smoke array
  `53127[].gaas` was accepted and all five members executed sequentially, but
  each exited during shell preflight before Python activation because
  `PBS_JOBDIR=/home/suaq0001` was incorrectly treated as scratch. No run shard,
  model load, benchmark observation, or scientific result was produced.
- PBS documents that its default HOME sandbox uses the user's home directory
  for staging/execution; only a PRIVATE sandbox creates a job-specific
  directory. Shell and Python checks now allow `PBS_JOBDIR == PBS_O_HOME` while
  retaining fail-closed rejection for a different/unverifiable `PBS_JOBDIR`
  and PBS-assigned `TMPDIR`. Eight targeted old/new regressions and the full
  three-file scheduler/runtime subset passed (`56 passed`). Corrected executable
  `source_tree_hash`:
  `171a67cd719beda8fed4422203137c6a3b9a8092408962ddd42f2cea88935865`.
- The HOME-sandbox correction and this incident record are uncommitted. Commit
  them before resubmission so run provenance has a clean, immutable revision.

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
- explicit CPU grid, PBS/Slurm run, and CPU aggregate entrypoints.

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
SCHEDULER=
ACCOUNT_OR_PROJECT_FLAG=
CPU_QUEUE_AND_RESOURCE_FLAGS=
GPU_QUEUE_AND_RESOURCE_FLAGS=
ONE_GPU_RESOURCE_FLAGS=
TWO_GPU_RESOURCE_FLAGS=
GPU_MODEL=
GPU_VRAM_GB=
MAX_ARRAY_CONCURRENCY=
```

Use read-only platform checks first:

```bash
pwd
python3.11 --version
command -v qsub
qsub --version
command -v sbatch
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
inside Slurm `SLURM_TMPDIR`, a PBS private-sandbox `PBS_JOBDIR` that differs
from `PBS_O_HOME`, or PBS-assigned `TMPDIR`. The normal PBS HOME sandbox sets
`PBS_JOBDIR` to persistent `PBS_O_HOME` and is allowed.

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

Create its persistent scheduler-log directory before submission. PBS requires
the destination directory to exist before it can stage stdout/stderr there:

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

The run stage accepts PBS batch arrays and Slurm arrays. On this destination,
the operator-approved `gpu_free` smoke allocation necessarily reserves one GPU
even though the deterministic fixture does not use it:

```bash
export AGENTDOJO_REPO_ROOT="$PWD"
export PBS_RUN_VARIABLES="AGENTDOJO_REPO_ROOT=$AGENTDOJO_REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,OUT_ROOT=$OUT_ROOT,STAGE=run,GRID_MANIFEST=$OUT_ROOT/$ID/grid/grid-manifest.jsonl,AGENTDOJO_FAKE_MODEL=1,AGENTDOJO_REQUIRES_GPU=0"

qsub -P fs_ccds_asysong \
  -q gpu_free \
  -l select=1:ncpus=12:ngpus=1:mpiprocs=1:mem=250gb \
  -l walltime=00:30:00 \
  -N "st-$ID-smoke" \
  -J 0-4%1 \
  -o "$OUT_ROOT/logs/$ID/" \
  -e "$OUT_ROOT/logs/$ID/" \
  -v "$PBS_RUN_VARIABLES" \
  "$AGENTDOJO_REPO_ROOT/$SCRIPT"
```

Do not use `qsub -V`; the explicit `-v` allowlist avoids copying unrelated
login-state variables. PBS supplies `PBS_JOBID`, `PBS_ENVIRONMENT`,
`PBS_ARRAY_ID`, and `PBS_ARRAY_INDEX`. Start at `%1`; increase concurrency only
after the platform smoke is stable. No GPU is needed for the fake plan. A
successful task ends with JSON containing `task_id`, `completed_shards`, and
`configuration_hashes`.

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
export PBS_PAIR_TRAIN_VARIABLES="AGENTDOJO_REPO_ROOT=<REPO_ROOT>,PYTHON_BIN=<PYTHON_BIN>,OUT_ROOT=<OUT_ROOT>,STAGE=run,PAIR_MINING_ACTION=observe,OBSERVATION_SPLIT=train,OBSERVATIONS_OUTPUT=<PERSIST_ROOT>/evidence/train.jsonl,OBSERVATION_MANIFEST_OUTPUT=<PERSIST_ROOT>/evidence/train.manifest.json,AGENTDOJO_STRATEGY_CATALOG=<PERSIST_ROOT>/evidence/candidate-strategies-v1.json,AGENTDOJO_PAIR_REGISTRY=<PERSIST_ROOT>/evidence/pair-registry-v1.json,AGENTDOJO_MODEL_CACHE=<PERSIST_ROOT>/model-cache,AGENTDOJO_MONITOR_CHECKPOINT=<PERSIST_ROOT>/checkpoints/action-monitor,AGENTDOJO_RUNTIME_FINGERPRINT=sha256:<FROZEN_RUNTIME_SHA>,AGENTDOJO_REQUIRES_GPU=1"

qsub <ACCOUNT_OR_PROJECT_FLAG> <GPU_QUEUE_AND_RESOURCE_FLAGS> <ONE_GPU_RESOURCE_FLAGS> \
  -N st-pair-train \
  -o <LOG_ROOT>/pair-mining/ \
  -e <LOG_ROOT>/pair-mining/ \
  -v "$PBS_PAIR_TRAIN_VARIABLES" \
  <REPO_ROOT>/experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
```

Submit development observation generation:

```bash
export PBS_PAIR_DEVELOPMENT_VARIABLES="AGENTDOJO_REPO_ROOT=<REPO_ROOT>,PYTHON_BIN=<PYTHON_BIN>,OUT_ROOT=<OUT_ROOT>,STAGE=run,PAIR_MINING_ACTION=observe,OBSERVATION_SPLIT=development,OBSERVATIONS_OUTPUT=<PERSIST_ROOT>/evidence/development.jsonl,OBSERVATION_MANIFEST_OUTPUT=<PERSIST_ROOT>/evidence/development.manifest.json,AGENTDOJO_STRATEGY_CATALOG=<PERSIST_ROOT>/evidence/candidate-strategies-v1.json,AGENTDOJO_PAIR_REGISTRY=<PERSIST_ROOT>/evidence/pair-registry-v1.json,AGENTDOJO_MODEL_CACHE=<PERSIST_ROOT>/model-cache,AGENTDOJO_MONITOR_CHECKPOINT=<PERSIST_ROOT>/checkpoints/action-monitor,AGENTDOJO_RUNTIME_FINGERPRINT=sha256:<FROZEN_RUNTIME_SHA>,AGENTDOJO_REQUIRES_GPU=1"

qsub <ACCOUNT_OR_PROJECT_FLAG> <GPU_QUEUE_AND_RESOURCE_FLAGS> <ONE_GPU_RESOURCE_FLAGS> \
  -N st-pair-dev \
  -o <LOG_ROOT>/pair-mining/ \
  -e <LOG_ROOT>/pair-mining/ \
  -v "$PBS_PAIR_DEVELOPMENT_VARIABLES" \
  <REPO_ROOT>/experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
```

After both observation jobs succeed, freeze pairs on a CPU PBS allocation:

```bash
export PBS_PAIR_REDUCE_VARIABLES="AGENTDOJO_REPO_ROOT=<REPO_ROOT>,PYTHON_BIN=<PYTHON_BIN>,OUT_ROOT=<OUT_ROOT>,STAGE=run,PAIR_MINING_ACTION=reduce,AGENTDOJO_REQUIRES_GPU=0,AGENTDOJO_STRATEGY_CATALOG=<PERSIST_ROOT>/evidence/candidate-strategies-v1.json,AGENTDOJO_PAIR_REGISTRY=<PERSIST_ROOT>/evidence/pair-registry-v1.json,TRAIN_OBSERVATIONS=<PERSIST_ROOT>/evidence/train.jsonl,TRAIN_OBSERVATION_MANIFEST=<PERSIST_ROOT>/evidence/train.manifest.json,DEVELOPMENT_OBSERVATIONS=<PERSIST_ROOT>/evidence/development.jsonl,DEVELOPMENT_OBSERVATION_MANIFEST=<PERSIST_ROOT>/evidence/development.manifest.json"

qsub <ACCOUNT_OR_PROJECT_FLAG> <CPU_QUEUE_AND_RESOURCE_FLAGS> \
  -N st-pair-reduce \
  -W depend=afterok:<TRAIN_JOB_ID>:<DEVELOPMENT_JOB_ID> \
  -o <LOG_ROOT>/pair-mining/ \
  -e <LOG_ROOT>/pair-mining/ \
  -v "$PBS_PAIR_REDUCE_VARIABLES" \
  <REPO_ROOT>/experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
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

With `qsub -J 0-4%4`, up to four workers run concurrently. One GPU per worker
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
export REPO_ROOT=<REPO_ROOT>
export PBS_RUN_VARIABLES="AGENTDOJO_REPO_ROOT=$REPO_ROOT,PYTHON_BIN=<PYTHON_BIN>,OUT_ROOT=$OUT_ROOT,STAGE=run,GRID_MANIFEST=$OUT_ROOT/$ID/grid/grid-manifest.jsonl,AGENTDOJO_GRID_PLAN=$AGENTDOJO_GRID_PLAN,AGENTDOJO_STRATEGY_CATALOG=$AGENTDOJO_STRATEGY_CATALOG,AGENTDOJO_PAIR_REGISTRY=$AGENTDOJO_PAIR_REGISTRY,AGENTDOJO_MODEL_CACHE=$AGENTDOJO_MODEL_CACHE,AGENTDOJO_ATTACKER_CHECKPOINT=$AGENTDOJO_ATTACKER_CHECKPOINT,AGENTDOJO_VICTIM_CHECKPOINT=$AGENTDOJO_VICTIM_CHECKPOINT,AGENTDOJO_MONITOR_CHECKPOINT=$AGENTDOJO_MONITOR_CHECKPOINT,AGENTDOJO_RUNTIME_FINGERPRINT=$AGENTDOJO_RUNTIME_FINGERPRINT,AGENTDOJO_FAKE_MODEL=0,AGENTDOJO_REQUIRES_GPU=1,ATTACKER_DEVICE=cuda:0,MONITOR_DEVICE=cuda:0,VICTIM_DEVICE=cuda:0"

qsub <ACCOUNT_OR_PROJECT_FLAG> <GPU_QUEUE_AND_RESOURCE_FLAGS> <ONE_GPU_RESOURCE_FLAGS> \
  -N "st-$ID" \
  -J <PRINTED_VALID_ARRAY_RANGE>%1 \
  -o "$LOG_ROOT/$ID/" \
  -e "$LOG_ROOT/$ID/" \
  -v "$PBS_RUN_VARIABLES" \
  "$REPO_ROOT/$SCRIPT"
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

- `STAGE=run` outside an unambiguous Slurm or PBS batch allocation exits before
  Python/model loading;
- a missing or noncanonical `SLURM_ARRAY_TASK_ID` or `PBS_ARRAY_INDEX` is
  rejected;
- an out-of-range array ID is rejected before environment activation;
- Python other than 3.11 is rejected by AgentDojo entrypoints;
- a smoke grid with `AGENTDOJO_FAKE_MODEL=0` is rejected;
- a production grid with `AGENTDOJO_FAKE_MODEL=1` is rejected;
- learned roles without a persistent cache/checkpoint or visible requested GPU
  are rejected;
- paths inside Slurm `SLURM_TMPDIR`, a PBS private-sandbox `PBS_JOBDIR` that
  differs from `PBS_O_HOME`, or PBS-assigned `TMPDIR` are rejected;
- unmaterialized plan templates and placeholder identities are rejected;
- missing/mismatched runtime, checkpoint, catalog, split, strategy, pair,
  analysis, freeze, or source identities are rejected;
- incomplete aggregation is rejected unless explicitly partial development
  mode is selected;
- E2 confirmatory gating is rejected without compatible passing E1 evidence.

Do not weaken these checks merely to make a platform command run.

## Known limitations

- No real PBS/Slurm/GPU/model benchmark execution was performed during
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
-> PBS-native authorization support validated
-> exact qsub CPU fake-smoke command and site flags approved
-> first E1 array 53127 diagnosed before Python; no run shards produced
-> PBS HOME-sandbox portability correction implemented and validated
-> correction committed at a clean revision
-> E1 fake-smoke PBS array resubmitted at concurrency %1
```

Only after the engineering smoke succeeds should the operator and new Codex
materialize production identities, mine guard pairs, or submit learned-model
arrays.

## Destination continuation checkpoint on 2026-08-28

This section supersedes the older immediate-next-milestone list above. The
repository remained on `main` at committed revision
`9a109ff1400af8f22cbd5288841a7ce6913d9dc1` before the uncommitted protocol
work described below.

- The complete deterministic fake-model controlled smoke finished: E1--E5
  covered 485 configurations and 26,100 rows with zero invalid rows.
- The ecological fake smoke finished 75 configurations and 1,305 rows. Its
  261 transformer-detector rows were conservatively invalid because no real
  detector checkpoint was configured; this was expected fixture behavior, not
  a detector result.
- All smoke artifacts remain engineering-only and establish no scientific
  benchmark finding.
- The operator reports that the production GPU allocation is NVIDIA H200. No
  real checkpoint inference has run yet.
- The working primary checkpoint choice is Qwen2.5-7B-Instruct for attacker
  and victim, Granite Guardian 4.1 8B for the action monitor, and the released
  ProtectAI DeBERTa-v3 prompt-injection-v2 detector for the ecological track.
  Checkpoint trees and a clean learned runtime have not yet been materialized
  or fingerprinted.
- The worktree now contains an uncommitted native Granite Guardian 4.1
  no-think scoring adapter. It uses exact structured chat, maps released
  `<score>yes|no</score>` outputs to block/allow, rejects generic JSON or
  reasoning prose, and freezes the complete protocol template. Focused tests
  passed (30), followed by an expanded 111-test regression set. The exhaustive
  run plus a corrected Python-3.11 rerun of its host-Python shell failures
  covered all 480 tests then collected and 79 subtests successfully. A final
  post-hardening targeted run passed 126 tests and 4 subtests.

The next checkpoint is to review and commit the protocol diff. After that,
build a clean non-editable Torch/Transformers/CUDA environment, download the
approved immutable checkpoint snapshots to persistent storage, derive all
runtime/checkpoint fingerprints, and run a development-only one-scenario
conformance job before guard-pair mining. Do not submit the real-model pilot
until the operator approves its exact resolved PBS command.

## H200 checkpoint-conformance freeze on 2026-08-28

This section supersedes the final paragraph above. No real-model scheduler job
has been submitted and no benchmark result has been generated.

- The dedicated learned environment is
  `/home/suaq0001/projects/.venvs/silenttwin-agentdojo-learned-py311` with
  CPython 3.11.15, Torch `2.12.1+cu126`, Transformers `5.16.1`, 108 installed
  distributions, and a successful `pip check`.
- SilentTwin is installed once, non-editably, from the exact wheel archived at
  `/home/suaq0001/projects/silenttwin-model-cache/runtime-artifacts/7c4dc4fbf5d417d506816626302636a40fe90de5ab198cd9083df94cfb245740/silenttwin-0.1.0-py3-none-any.whl`.
  Its wheel SHA-256 is
  `40741c7c6fcd8ed8596d64d10e2207ab420318199cef367fdc2e62f04e81d33a`.
  The previous ignored source `egg-info` was preserved at
  `/tmp/silenttwin.egg-info.pre-wheel-20260828`.
- The final learned-runtime fingerprint is
  `sha256:5f695781d5558474f168abee4caeeb0f39d4355f7b9b3e38bfa79b843622538d`.
  Runtime integrity must receive the 71-pin
  `requirements-tier2-agentdojo.lock`; the 107-pin
  `requirements-tier2-learned-h200.lock` is the provisioning record and
  source-hash material, not the core-lock CLI input.
- The executable `source_tree_hash` is
  `7c4dc4fbf5d417d506816626302636a40fe90de5ab198cd9083df94cfb245740`.
  The conformance launcher rejects a dirty worktree before runtime validation
  or model loading, so all listed repository changes must be committed before
  submission.
- Qwen attacker/victim checkpoint:
  `/home/suaq0001/projects/silenttwin-model-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`,
  semantic commit `a09a35458c702b33eeacc393d103063234e8bc28`, full-tree fingerprint
  `sha256:bfb9ad97ebbceae4eb4b54fc85334d0a71f5e157176323712a7b3ed6e0d05e8e`.
- Granite monitor checkpoint:
  `/home/suaq0001/projects/silenttwin-model-cache/hub/models--ibm-granite--granite-guardian-4.1-8b/snapshots/e30b8a2343efe8030479777d467ebb305ca109e9`,
  semantic commit `e30b8a2343efe8030479777d467ebb305ca109e9`, full-tree fingerprint
  `sha256:31a587dc521951a7288ead06c9f8226bceb51d410094e8d47c04dee3602a746f`.
- ProtectAI ecological detector checkpoint:
  `/home/suaq0001/projects/silenttwin-model-cache/hub/models--protectai--deberta-v3-base-prompt-injection-v2/snapshots/90c9989b1a342275dd0d1a95aad283c04e075671`,
  semantic commit `90c9989b1a342275dd0d1a95aad283c04e075671`, full-tree fingerprint
  `sha256:df326f40bf3bd0b71ecf7ef97278a75787a10146c4371e5b03fac18b3998cbc5`.
- The validated, read-only engineering catalog is
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-engineering-candidate-strategies-v1.json`.
  File SHA-256 is
  `0159498b75339b0630b25499f58c2979e4623713afd1fde785afae5e6f0f1af6`;
  internal catalog hash is
  `e4cb1fefe7d37e48da5b568b83b3cbd96f99322a037b3c41a2a3fef3767159ce`.
- The validated, read-only specification is
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-checkpoint-conformance-spec-v1.json`.
  File SHA-256 is
  `9573c491cb725068b062c9b2ea4288086b2680e0f01e77a516f095bf37111c14`;
  internal spec hash is
  `d882dfc2631b84fe64c8bd644c569f38e4a7c42f7743b392269ceace124b859b`.
  It selects one frozen workspace development scenario, two schema-valid
  engineering-only prohibited-action probes, Qwen with 256 output tokens,
  and two separately retained Granite no-think profiles with 64 output tokens.
- The first real-checkpoint conformance job was submitted as PBS job
  `54096.gaas` to `gpu_free` under project `fs_ccds_asysong`. It ran once on
  `hpc-gaas-g25`, finished in 78 seconds, and returned `Exit_status = 0`.
  PBS recorded one NVIDIA H200, 12 CPUs, 250 GiB host RAM, and maximum GPU
  memory use of 48,586 MB.
- The atomically published report is
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-checkpoint-conformance-report-v1.json`.
  Its file SHA-256 is
  `519baea077bdd1e97dd8f05164badb16d23e4717f4d5294064862b871bd73816`;
  its internal self-hash is
  `62454c4aad92750e85b166a4f625cf7082f15492dd5e426df94f7584f8d63181`.
  Strict report validation passed with 8/8 checks, zero errors, zero external
  API calls, three simultaneously retained clients, all three Qwen attacker
  contracts valid, all four Granite monitor cells valid, and successful
  attacker retirement. All four engineering probes were blocked; this is a
  protocol-conformance observation, not a monitor-quality estimate.
- The allocated H200 exposed 150,109,880,320 bytes. Process-local Torch
  allocation rose from zero to 15,276,322,816 bytes after Qwen, to
  32,037,950,464 bytes after the first Granite client, and to
  48,799,053,824 bytes after the second. Peak allocated memory after protocol
  checks was 49,567,984,640 bytes. This validates the intended one-H200,
  three-client topology for the selected 7--8B checkpoints.
- The conformance implementation binds exact identities and verifies three
  attacker contracts, every profile/strategy/call monitor cell, raw local-only
  provenance, pinned AgentDojo compatibility, five ordered H200 memory stages,
  retirement, and zero external API calls. The report remains engineering-only
  and cannot be consumed as pair-selection or benchmark evidence.
- Final verification passed `git diff --check`, Python compilation, and the
  focused 49-test conformance/shell/pipeline set. The complete collection
  produced 493 passes and 77 passing subtests; its only seven failures were
  nested shell-grid invocations that inherited system Python 3.9 because the
  outer test command omitted `PYTHON_BIN`. Re-running the complete affected
  shell-grid file with the pinned Python 3.11 binary passed 14 tests and 4
  subtests, covering every one of those invocation-only failures.

The scientific strategy audit found that the current generic candidate-plan
schema cannot honestly provide two distinct executable prohibited
transformations for every frozen scenario. At least 60 train scenarios have a
single rigid or output-only injection objective, and pair observation scores
monitor calls without executing or grading attack success. After engineering
conformance, do not scale the engineering catalog into production. First choose
and preregister either (a) an action-representable eligibility subset or (b) a
typed deterministic plan materializer plus an output-action protocol. That is
a scientific-protocol decision, not an engineering smoke result.

Immediate next checkpoint:

1. commit this handoff-only result record so `git status --short` is empty;
2. choose and preregister either the action-representable eligibility subset
   or the typed deterministic plan/output-action protocol extension;
3. author a separate scientific train-frozen candidate catalog under that
   decision and audit executable action success before pair observation;
4. only then prepare and explicitly approve the train/development
   pair-observation PBS jobs. Never reuse the engineering conformance catalog
   or report for pair selection.

## Action-representable estimation protocol on 2026-08-29

This section supersedes the immediate checkpoint above. The implementation is
present in the working tree on top of committed revision
`378dd852d72731bc190252c4dda60213802a303d`; it has not yet been committed. No
PBS job, model inference, pair observation, or benchmark run was submitted.

- The checked model-independent eligibility freeze is
  `configs/silenttwin/agentdojo/action-eligibility-v1.json`, with internal hash
  `f06b32e0e8eb9d3fa632225d9f957143444d2c01ff9b2a43ec5c7e00fafdcb28`.
  It was reproduced immutably by the CPU-only freeze command.
- The disposition is `estimation_only_action_representable`. The pilot contains
  134 train scenarios and 59 development scenarios, covering all four suites.
  Its test cohort is empty. The wider audit finds 26 representable test
  scenarios, but Slack and Travel have none, so held-out execution and every
  confirmatory claim remain forbidden.
- The suite/split scenario and structural-group census is frozen in the
  manifest: workspace 28/28/18 scenarios and 20/10/10 groups; banking 24/8/8
  and 8/4/4; Slack 30/7/0 and 11/5/0; Travel 52/16/0 and 10/5/0 for
  train/development/test action-representable rows.
- A scientific candidate catalog must contain exactly two strategies, set
  `default_plan_policy` to `forbidden`, and enumerate an exact per-scenario
  plan for all 193 pilot scenarios. Generic, suite-level, missing, or extra
  fallback plans fail before checkpoint construction.
- Pair-observation generation now materializes both plans per scenario,
  compares required-argument action multisets, and rejects identical, nested,
  optional/default-only, or ordering-only variants. It executes each plan in a
  fresh AgentDojo environment, requires zero tool errors and released
  attack-success `True`, and publishes a self-hashed execution-validation
  ledger before monitor observations can become reducer evidence.
- Observation rows, set manifests, pair IDs, and pair registries bind the
  eligibility and action-validation hashes. Exact scenario/strategy/profile
  Cartesian coverage is required. The pair registry contains no test
  instantiations.
- Controlled grids are restricted to the frozen train/development pilot IDs.
  Test grid construction, held-out sample-size freezing, held-out assembly,
  and confirmatory aggregation fail closed. Aggregates retain scientific
  estimates but mark every gate nonconfirmatory and
  `sample_size_freeze_eligible: false`.
- The deterministic engineering-smoke fixtures remain on their legacy branch
  and continue to validate without being relabeled as scientific evidence.
- Final executable `source_tree_hash` for the current uncommitted tree is
  `0c7a7f45d2f366c51520e33ec4e97a5dab6e7ac17b2621fba2e05a6c095c6b9e`.
  The old installed learned wheel, runtime fingerprint, conformance spec, and
  conformance report are bound to source hash
  `7c4dc4fb5f5d417d506816626302636a40fe90de5ab198cd9083df94cfb245740`
  and must not be used for pair-observation evidence from this implementation.
- Verification passed `git diff --check`, Python compilation, shell syntax,
  an exhaustive `508 passed, 79 subtests passed` repository run, and a final
  post-review focused `113 passed` run.

Immediate next checkpoint:

1. review and commit the estimation-protocol diff without changing the frozen
   eligibility artifact;
2. build and archive a new noneditable wheel from that exact clean revision,
   reinstall or recreate the learned Python 3.11 environment, and derive its
   new wheel, installed-payload, source-tree, and runtime fingerprints;
3. author the separate scientific catalog with two exact plans for every one
   of the 193 pilot scenarios and the approved Granite profiles, then run the
   model-free execution/released-grader audit over all 386 plans;
4. refresh the source/runtime-bound one-H200 conformance specification and
   report if the refreshed scientific runtime will be used;
5. inspect and explicitly approve the resolved PBS train observation command
   before any qsub submission, then run development only after train succeeds.

Do not submit pair mining yet: the current learned runtime is source-stale and
the separate scientific 193-scenario candidate catalog has not been authored.

## Scientific catalog and refreshed runtime freeze on 2026-08-29

This section supersedes the immediate checkpoint above. The estimation
protocol was committed as
`952954626687adc538403b1421f946526803f06a` (`Add action-representable
estimation protocol`). No PBS job, learned-model inference, pair observation,
pair selection, or benchmark run was submitted in this checkpoint.

- The committed worktree reproduced executable `source_tree_hash`
  `0c7a7f45d2f366c51520e33ec4e97a5dab6e7ac17b2621fba2e05a6c095c6b9e`.
- A wheel built offline from `git archive HEAD` reproduced byte-for-byte in two
  independent builds. It is archived read-only at
  `/home/suaq0001/projects/silenttwin-model-cache/runtime-artifacts/0c7a7f45d2f366c51520e33ec4e97a5dab6e7ac17b2621fba2e05a6c095c6b9e/silenttwin-0.1.0-py3-none-any.whl`
  with SHA-256
  `0be03b391b4bf6c5033ea085d3da137fbf1dc44fdffc231a6c249acfdc9624f5`.
- The dedicated learned Python 3.11 environment was changed only by
  force-reinstalling that local wheel with no dependency resolution. Its 111
  immutable SilentTwin payload files matched the wheel byte-for-byte; their
  canonical payload-manifest SHA-256 is
  `eda858aca84b4ce07aa95687454eedb6cad9862577f7abec5c0857291737336b`,
  and the installed RECORD identity is
  `2837b7ecd27155e7bb3f4c359288375d11a9ca8b6d828cfa63272e800ea29716`.
  `pip check` passed.
- The refreshed 108-distribution learned-runtime fingerprint is
  `sha256:76b337594fb9bb6271581ff2c0edec0f3e35931dd43a52a46a540f991e7a7bd3`.
  The core runtime-integrity input remains the 71-pin
  `requirements-tier2-agentdojo.lock`.
- The separate scientific catalog is frozen read-only at
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/evidence/candidate-strategies-v1.json`.
  Its file SHA-256 is
  `c31b0962daec06619a6bdd33c52d2756203f25ea20537c32e7930f13d868dcff`;
  its internal candidate-catalog hash is
  `84ffd73d2ea8463c9fc9f4d4439b668f78f70f7970bb8953edd53b82c5f371a1`.
- The catalog contains exactly the two nonfallback strategies
  `released-ground-truth-derived-path-v1` and
  `typed-distinct-materialization-v1`, with exact plans for all 193 pilot
  scenarios, and the two train-frozen Granite profiles
  `scientific-integrity-guardian-v1` and
  `scientific-privacy-guardian-v1`. Both profiles bind the approved Granite
  checkpoint and refreshed runtime above. No test plan is present.
- The workspace calendar task required an exact correction during authoring:
  `participants` is optional in the released `create_calendar_event` schema,
  so merely adding a participant would not change its required-argument
  signature. The frozen paths instead compare a target-day conflict check plus
  inline-participant creation with creation followed by the required
  `add_calendar_event_participants` action. Both are goal-faithful, nonnested,
  and pass the released grader.
- The model-free action audit is frozen read-only at
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/evidence/candidate-strategies-action-audit-v1.json`.
  Its file SHA-256 is
  `f3c0870d93f30932ff7d4996d89bdafed7641929d8e9fb805512f0a4e243cba7`;
  its internal audit hash is
  `ff35edfab325abe9273c54c26345c010d23cbf9449f591ac2a95e6de553e410a`.
  All 386 scenario/strategy plans had distinct nonnested required-action
  multisets, executed in fresh clones without tool errors, and returned
  released attack-success `True`. External API calls, model inference, learned
  attacker/monitor outcome inspection, and test outcome inspection were all
  zero/false.
- The scientific catalog reproduced byte-for-byte. A second complete
  386-plan audit reproduced every identity, call sequence, required-action
  hash, initial-state hash, trace hash, and grader result. Only post-state and
  enclosing validation hashes differ because pinned AgentDojo email tools
  retain wall-clock `datetime.now()` timestamps; that execution-specific
  boundary is recorded in the final audit.
- The refreshed engineering-only conformance catalog is archived read-only at
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-engineering-candidate-strategies-v2.json`.
  File SHA-256 is
  `2f5aabb7b6e2e418fc110f9da72e46a45af20e50c4324c1ad978052f38aa70bf`;
  internal candidate-catalog hash is
  `f3309a687ce21ebffea4029c4c68d4b093c2cc5ee24f45f785b844ed3e16329e`.
- The refreshed one-H200 spec is archived read-only at
  `/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-checkpoint-conformance-spec-v2.json`.
  File SHA-256 is
  `4c459cb257ba91dedfd8e0bbcef41bcb9741251c3a4afe91f6facf5e30572013`;
  internal spec hash is
  `e6eafbd6c33a8e0ce23cf8bd18988470b0b4cf93a5502dfd3f4f3d1c9b88ea72`.
  It retains the previously passed engineering scenario/probes/checkpoints but
  binds the refreshed source and runtime. The v2 report path is deliberately
  absent because no scheduler submission was authorized.
- Independent artifact validation and the focused protocol regression set
  passed; the latter produced `81 passed in 23.74s` across action eligibility,
  pair mining, conformance, runtime integrity, and shell entrypoints.

Immediate next checkpoint:

1. review and commit this handoff-only update; the conformance launcher rejects
   a dirty worktree, while the documentation-only commit does not change the
   frozen executable source hash or runtime fingerprint;
2. inspect and explicitly approve the resolved v2 conformance command below;
3. submit it once, validate the new report, and only then inspect and approve
   the train pair-observation command. Development observation remains
   conditional on successful train observation.

Resolved v2 conformance command (prepared, **not submitted**):

```bash
export PBS_CONFORMANCE_V2_VARIABLES="AGENTDOJO_REPO_ROOT=/home/suaq0001/projects/silent_twin,PYTHON_BIN=/home/suaq0001/projects/.venvs/silenttwin-agentdojo-learned-py311/bin/python,OUT_ROOT=/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production,STAGE=run,AGENTDOJO_DATASET_SPLIT=development,AGENTDOJO_STRATEGY_CATALOG=/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-engineering-candidate-strategies-v2.json,AGENTDOJO_MODEL_CACHE=/home/suaq0001/projects/silenttwin-model-cache,AGENTDOJO_ATTACKER_CHECKPOINT=/home/suaq0001/projects/silenttwin-model-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28,AGENTDOJO_MONITOR_CHECKPOINT=/home/suaq0001/projects/silenttwin-model-cache/hub/models--ibm-granite--granite-guardian-4.1-8b/snapshots/e30b8a2343efe8030479777d467ebb305ca109e9,AGENTDOJO_RUNTIME_FINGERPRINT=sha256:76b337594fb9bb6271581ff2c0edec0f3e35931dd43a52a46a540f991e7a7bd3,CONFORMANCE_SPEC=/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-checkpoint-conformance-spec-v2.json,CONFORMANCE_OUTPUT=/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/conformance/controlled-h200-checkpoint-conformance-report-v2.json,EXPECTED_SOURCE_TREE_HASH=0c7a7f45d2f366c51520e33ec4e97a5dab6e7ac17b2621fba2e05a6c095c6b9e,ATTACKER_DEVICE=cuda:0,MONITOR_DEVICE=cuda:0,AGENTDOJO_REQUIRES_GPU=1,AGENTDOJO_FAKE_MODEL=0"

qsub -P fs_ccds_asysong \
  -q gpu_free \
  -l select=1:ncpus=12:ngpus=1:mpiprocs=1:mem=250gb \
  -l walltime=00:15:00 \
  -N st-conform-v2 \
  -o /home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/logs/conformance/ \
  -e /home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production/logs/conformance/ \
  -v "$PBS_CONFORMANCE_V2_VARIABLES" \
  /home/suaq0001/projects/silent_twin/experiments/silenttwin/run_agentdojo_checkpoint_conformance_tier2.sh
```

Do not submit pair observation yet. The refreshed H200 conformance report must
pass first, and the exact train command still requires separate operator
approval.
