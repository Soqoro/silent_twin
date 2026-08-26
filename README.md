# SilentTwin

Experimental code for **SilentTwin: The Guardrail Should Not Teach the
Attacker**. The project asks whether guardrail feedback lets an adaptive agent
infer private enforcement state, whether that information improves bypass
selection, and whether a feedback-independent twin can close the channel while
preserving independent safe work.

The checked-in implementation includes a deterministic Tier-1 finite-state
testbed, the legacy synthetic Tier-2 model adapter, and a separate AgentDojo
Tier-2 benchmark over all four pinned suites. Tier 1 and AgentDojo's model-free
catalog, grid, aggregate, and deterministic engineering-smoke paths need no
GPU. Real local-model runs use dedicated scheduler-authorized launchers;
AgentDojo accepts PBS Professional or Slurm, while the legacy synthetic Tier-2
wrappers remain Slurm-specific. Runs never download a checkpoint or fall back
to a provider API or mock model.

## Core separation

`TruePolicy` and `Monitor` answer different questions. The true policy labels
whether an effect is actually permitted. The monitor is an imperfect predictor
that may allow a prohibited effect. In particular, a monitor allow is never
treated as proof of safety.

SilentTwin gives the protected session guardrail-independent temporary
observations and stages external mutations. A separate controller sees private
monitor and policy records, checks complete data/control dependency closure and
atomic groups, and requires matching commit tokens. The protected context is
retired before settlement; only a distinct trusted process receives the final
truthful report.

## Quick start

Python 3.10–3.13 and Bash 4.3 or newer are supported. The reproducible CPU
development environment deliberately excludes `torch` and `transformers`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pytest -q
```

The launchers also prepend `src/` to `PYTHONPATH`, so an already provisioned
Python environment can run them without an editable install.

Inspect a grid without running it:

```bash
E1_STAGE=grid bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

For the distinct AgentDojo backend, Python 3.11 pin, catalog/pair freeze,
offline array commands, aggregation contract, and evidence limitations, see
the [AgentDojo Tier-2 operator guide](docs/agentdojo_tier2.md). Its checked
deterministic artifacts and fake model are engineering fixtures, not benchmark
evidence.

Run one deterministic two-sample configuration:

```bash
TIERS=tier1 WORLD_SUITES=email RUNTIMES=generic_failure \
ATTACKERS=mock_llm QUERY_BUDGETS=4 SEEDS=42 NUM_SAMPLES=2 \
SLURM_ARRAY_TASK_ID=0 \
bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
```

Run and aggregate one tiny configuration for every experiment:

```bash
bash experiments/silenttwin/run_all_pilots.sh
```

The pilot helper defaults to `outputs/silenttwin-pilot`; set `OUT_ROOT` to
change it. Re-running a compatible completed task reuses it. Set `OVERWRITE=1`
only when intentionally replacing prior output.

Run the complete explicit Pilot A engineering smoke (E1 and E2) locally:

```bash
OUT_ROOT=outputs/silenttwin-pilot-a \
bash experiments/silenttwin/run_pilot_a_engineering_smoke.sh
```

Inspect the batched Pilot B mapping without executing it:

```bash
PILOT_PRESET=pilot_b E1_STAGE=grid \
bash experiments/silenttwin/run_experiment_1_feedback_leakage.sh
PILOT_PRESET=pilot_b E2_STAGE=grid \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

## Experiments

| ID | Hypothesis | Main independent variables | Primary metrics |
|---|---|---|---|
| E1: feedback leakage | Decision-correlated feedback reveals hidden guardrail state; exact SilentTwin does not. | Runtime, genuine/shuffled feedback source, attacker, query budget, world suite, seed | State-inference accuracy and lift over prior, transcript AUC, proxy-guardrail accuracy versus query budget |
| E2: feedback-assisted bypass | Feedback helps an attacker choose a state-specific monitor false negative. | Runtime, attacker, query budget, adaptive/random/oracle/shuffled condition, world suite, seed | Prohibited-effect rate, feedback-assisted gain, monitor false-negative rate |
| E3: channel closure | Full SilentTwin removes transcript distinguishability and the downstream adaptive advantage. | Baseline/full/degraded runtime, attacker, query budget, world suite, seed | Exact transcript distance, inference accuracy/AUC, prohibited effects and gain versus query budget |
| E4: useful work | Dependency-aware staging preserves eligible independent effects while rejecting unsafe effects, descendants, and inconsistent atomic groups. | Runtime, workflow family, world suite, seed | Task success, salvage, rejected/descendant commits, atomicity violations, repair/report accuracy, cost |
| E5: assumption ablations | Reintroducing individual channels or commit-integrity failures identifies which protections are necessary. | Ablation, attacker, query budget, world suite, seed | Leakage/AUC, prohibited and descendant effects, utility/salvage, leakage attributable to each ablation |

E2 always gives exactly one final exploit selection. Its causal controls keep
that final budget fixed. SilentTwin is not claimed to remove a blind spot that
already succeeds with no feedback; the estimand is the additional adaptive
gain:

```text
feedback_assisted_gain = prohibited_effect_rate(Q > 0)
                         - prohibited_effect_rate(Q = 0)
```

E4 defines salvage as eligible approved effects committed divided by all
eligible approved effects.

## Entry points

Every experiment launcher supports `grid`, `run`, and `aggregate`. The grid
stage prints the exact ordered mapping, configuration hash, shard identity, and
overall grid hash for every member. The default stage is `run`, selected with
`SLURM_ARRAY_TASK_ID`.

```text
experiments/silenttwin/run_experiment_1_feedback_leakage.sh
experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
experiments/silenttwin/run_experiment_3_channel_closure.sh
experiments/silenttwin/run_experiment_4_useful_work.sh
experiments/silenttwin/run_experiment_5_assumption_ablations.sh
experiments/silenttwin/run_experiment_1_feedback_leakage_tier2.sh
experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_tier2.sh
```

The AgentDojo backend has eight explicit entrypoints and does not change the
meaning of those finite-state launchers:

```text
experiments/silenttwin/run_agentdojo_catalog.sh
experiments/silenttwin/run_agentdojo_pair_mining_tier2.sh
experiments/silenttwin/run_experiment_1_feedback_leakage_agentdojo_tier2.sh
experiments/silenttwin/run_experiment_2_feedback_assisted_bypass_agentdojo_tier2.sh
experiments/silenttwin/run_experiment_3_channel_closure_agentdojo_tier2.sh
experiments/silenttwin/run_experiment_4_useful_work_agentdojo_tier2.sh
experiments/silenttwin/run_experiment_5_assumption_ablations_agentdojo_tier2.sh
experiments/silenttwin/run_agentdojo_ecological_tier2.sh
```

For example:

```bash
E2_STAGE=aggregate \
bash experiments/silenttwin/run_experiment_2_feedback_assisted_bypass.sh
```

The analogous selectors are `E1_STAGE` through `E5_STAGE`. Commands are built
as Bash arrays, printed with shell escaping, and executed without `eval`.

## Repository map

```text
src/silenttwin/
  worlds/        paired finite-state worlds and public projections
  policy/        true policy and deliberately imperfect monitors
  runtime/       feedback methods, SilentTwin, lifecycle, dependency controller
  tools/         temporary tools plus Tier-2 adapter boundaries
  attackers/     deterministic, Bayesian, black-box, and model interfaces
  experiments/   E1-E5 runners and aggregation
  metrics/       privacy, safety, utility, and paired confidence intervals
  io/            atomic JSONL, manifests, and provenance
experiments/silenttwin/   human-facing shell launchers
configs/silenttwin/       documented Tier-1 defaults
tests/                    deterministic unit, integration, and smoke checks
outputs/silenttwin/       generated results (not source data)
logs/                     local/SLURM console logs
```

Each run directory contains `result.jsonl`, `manifest.json`, `run.log`,
`failures.jsonl`, a checkpoint manifest, and per-episode checkpoints. Completed
episodes are atomically durable, so rerunning the same task resumes only missing
episode IDs. Results are validated and atomically published. Finite-state
JSONL uses one final summary record; AgentDojo JSONL is sample-only and its
cross-cell summary is published by the separate aggregate step. Private state,
monitor verdicts, and ground-truth labels belong to trusted evaluation fields
and are excluded from the canonical agent-visible transcript.

See the detailed guides for the [AgentDojo Tier-2 benchmark](docs/agentdojo_tier2.md),
[research design](docs/research_design.md),
[output/provenance schema](docs/output_schema.md),
[local and SLURM operation](docs/operator_guide.md), and
[extension points and threat-model boundaries](docs/extending_and_threat_model.md).

## Scope and interpretation

Tier 1 establishes exact finite-state invariants and provides reproducible
synthetic evidence. It does not establish behavior for a particular production
model, provider, operating system, or distributed tool stack. Pilot C and D are
development-only and must not be reported as held-out headline evidence. Do not
report mock outputs as empirical LLM results. The repository never calls a paid
API, downloads a large model, allocates a GPU, or submits a SLURM job on its own.
