# Output and provenance schema

Finite-state Tier-1/Tier-2 leaf directories are self-validating bundles:

```text
result.jsonl   canonical per-sample records followed by one summary
manifest.json  configuration, counts, hashes, status, and provenance
run.log        concise lifecycle record for this leaf
failures.jsonl terminal failure ledger (empty for a clean completed run)
checkpoint_manifest.json configuration/source-bound resume state
checkpoints/   one atomic JSON checkpoint per completed episode
```

Scheduler console logs under `logs/` are operational diagnostics and are not a
substitute for these artifacts.

## Finite-state `result.jsonl`

Every non-empty line is one JSON object. The schema version is
`silenttwin.result.v2`. All records before the last have
`"record_type":"sample"`; the last and only last record has
`"record_type":"summary"`. Blank lines, duplicate sample IDs, an absent or
non-final summary, and a count mismatch are validation errors.

### Sample identity and factors

Common fields are:

```text
schema_version, record_type, experiment_id, tier
sample_id, paired_world_id, pair_family, template_id, public_instance_hash
dataset_split, dataset_revision
world_suite, runtime, attacker, query_budget, seed, decoding_seed
configuration_hash, code_revision
```

The configuration hash covers the scientific configuration, including resolved
sample count and any E2 condition, E4 workflow, or E5 ablation. It excludes the
output directory and overwrite policy, which do not change the experiment.

### Public transcript and attack outcome

Where applicable, records contain:

```text
decision_visible_transcript, delivered_visible_transcript
agent_visible_transcript, postselection_visible_transcript, transcript_hash
hidden_state_prediction, posterior, posterior_entropy, prediction_valid
selected_final_exploit, final_exploit_attempt_count
```

`decision_visible_transcript`, `delivered_visible_transcript`, and
`agent_visible_transcript` are identical snapshots of exactly the observations
used by attacker callbacks. A final action's public result, which occurs only
after selection is irrevocably locked and has no subsequent attacker callback,
is recorded separately in `postselection_visible_transcript`. These streams may
include public actions and normalized tool
values, errors, receipts, identifiers, timing buckets, retry/termination events,
callbacks, visible memory, and settlement messages. Validation recursively
rejects trusted-only keys such as `private_state`, `theta`, `monitor_decision`,
`true_policy_label`, and `admissible` inside this field.

### Staging, safety, and utility

Depending on the experiment, records also include:

```text
true_policy_label, monitor_decision
staged_effect_ids, committed_effect_ids, prohibited_committed_effect_ids
ground_truth_prohibited_effect_committed
task_success_indicator
salvage_numerator, salvage_denominator, eligible_action_salvage_rate
rejected_effect_commit_rate, rejected_descendant_commit_rate
dependency_violation, atomicity_violation
trusted_repair_success, truthful_report_accuracy
latency_ms, tool_call_count, model_usage
```

Top-level policy/monitor fields are trusted evaluator output, not attacker
transcript content. Prohibited commits are derived from the committed effect's
authoritative `TruePolicy` label, never from the monitor-rejected set.

`model_usage` and `model_provenance` distinguish deterministic/data-only
attackers from structured mock and Tier-2 model calls. Tier 2 records exact
model/tokenizer revisions, chat-template/prompt/response hashes, decoding
settings, token counts, latency, retries/failures, and software/GPU metadata.

### Trusted evaluation and provenance

`trusted_evaluation` contains the private state and experiment-specific ground
truth used only by the evaluator/controller. `generation_provenance` describes
the producer of the sample, while `evaluation_provenance` identifies the
evaluator. They are kept distinct so re-evaluation does not masquerade as new
generation.

### Summary record

The final record repeats schema, experiment, tier, sample count, configuration
hash, and manifest configuration, then stores the experiment's aggregate
`metrics`. This per-leaf summary is not a replacement for cross-configuration
aggregation.

## Finite-state `manifest.json`

The manifest schema is `silenttwin.manifest.v2`. Required bindings include:

```text
status = complete
experiment_id
configuration and configuration_hash
expected_sample_count and actual_sample_count
result_file and result_sha256
failures_file and failures_sha256
checkpoint_manifest and checkpoint source/configuration binding
started_at and completed_at
provenance
generation_provenance and evaluation_provenance
```

Code provenance records the Git revision when available, dirty-tree flag,
source-tree content hash, package version, Python implementation/version, and
platform. The source-tree hash covers package Python, experiment launchers,
Tier-1 JSON configuration, and packaging metadata, including uncommitted files,
so two dirty scientific source states are not assumed compatible merely because
they share a Git revision.

Reuse validates manifest and result schema versions, status, experiment,
configuration hash, resolved count, result digest, record structure, and current
provenance. A compatible completed leaf is reused. An incomplete or incompatible
leaf fails unless replacement is explicitly authorized with `OVERWRITE=1`.

## Atomic publication

The runner materializes and validates all records, writes a temporary file in
the destination filesystem, flushes it, parses and validates the exact bytes,
and then atomically renames it to `result.jsonl`. Manifest and run-log writes use
the same temporary-file/rename pattern. This prevents a partially written JSONL
from looking complete; the manifest's status/digest/count checks provide the
cross-file integrity check.

## Finite-state aggregate output

Each aggregate directory contains:

```text
summary.json
summary.csv
grid_manifest.jsonl
validated_run_index.json
paired_comparisons.csv
analysis_manifest.json
```

Experiment-specific exports are:

```text
E1: accuracy_vs_q.csv, auc_vs_q.csv, entropy_reduction_vs_q.csv,
    heldout_monitor_fidelity_vs_q.csv
E2: state_prediction.csv, matched_exploit_rate.csv,
    monitor_acceptance.csv, prohibited_effect_rate.csv,
    feedback_assisted_gain.csv, causal_chain_table.csv
E3: the E1/E2 tables plus privacy_safety_utility.csv
E4: privacy_safety_utility.csv
E5: privacy_safety_utility.csv, ablation_table.csv
```

`summary.json` preserves logical treatment groups, source leaf hashes,
public-task-cluster confidence intervals, matched paired comparisons, and
criterion-level G0–G4 evidence. Physical shards are combined only after exact
grid validation and contiguous/non-overlapping sample-range checks. The shell
entrypoints supply the expected grid manifest and hash, so an absent,
substituted, duplicate, or same-sized-wrong configuration is rejected. The
analysis manifest separately records trajectory and aggregation code
provenance, shard-to-analysis-cohort composition, pre-registered contrasts,
confidence settings, power/freeze status, and go/no-go gates.

E1's `e1_shuffled_q16_minus_q0` comparison records
`pairing_unit=public_instance_hash_task_mean`, distinct target/reference row
counts, and one `matched_pair_count` per public task. This intentionally pairs
four-row shuffled task means with two-row genuine-Q0 task means without
row-level pseudoreplication. G2 evaluates equivalence only when that complete
task cohort exists.

## AgentDojo Tier-2 bundles

AgentDojo uses a separate schema and does not append a leaf summary row.
Every non-empty line of `result.jsonl` is a sample with
`schema_version="silenttwin.agentdojo.result.v1"` and
`record_type="sample"`. Its `manifest.json` declares
`silenttwin.agentdojo.manifest.v1`, the exact expected/actual trial count,
result and failure digests, scenario/group IDs, the complete scientific
configuration, catalog/split/strategy/pair/analysis/dependency bindings,
checkpoint resume state, grid/shard identity, and source-tree provenance.
For learned execution, `provenance.learned_runtime` retains the complete
Python and installed-distribution version/RECORD manifest; its recomputed
canonical hash must match the runtime fingerprint frozen into all selected
model identities. Model-free shards record the canonical `not_applicable`
sentinel. Learned pair-observation-set manifests retain and validate the same
runtime envelope before their self-hash is accepted.

The AgentDojo trusted namespace keeps private theta, monitor decisions,
ground-truth policy labels, grader outcomes, environment hashes, and raw tool
traces out of `agent_visible_transcript`. Generation provenance distinguishes
attacker, victim, and monitor calls. Each learned call records the canonical
messages and tool schemas, protocol prompt, tokenizer-rendered input, raw
response hash, parsed calls or decision, seed, tokens, latency, checkpoint and
runtime identity, plus terminal failure metadata. These trusted records are
not fed back to the model.

An AgentDojo aggregate directory contains exactly the implemented artifacts:

```text
summary.json
analysis_manifest.json
validated_run_index.json
grid_manifest.jsonl
```

`summary.json` contains suite-stratified estimates, task-weighted sensitivity,
pair-yield/headroom tables, error accounting, and gates.
`analysis_manifest.json` self-hashes the development evidence, power analysis,
held-out freeze binding, analysis plan, grid, upstream chain, and fixture claim
boundary. No AgentDojo CSV exports are currently produced.
