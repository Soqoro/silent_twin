# Output and provenance schema

Every completed leaf directory is a self-validating bundle:

```text
result.jsonl   canonical per-sample records followed by one summary
manifest.json  configuration, counts, hashes, status, and provenance
run.log        concise lifecycle record for this leaf
```

Scheduler console logs under `logs/` are operational diagnostics and are not a
substitute for these artifacts.

## `result.jsonl`

Every non-empty line is one JSON object. The schema version is
`silenttwin.result.v1`. All records before the last have
`"record_type":"sample"`; the last and only last record has
`"record_type":"summary"`. Blank lines, duplicate sample IDs, an absent or
non-final summary, and a count mismatch are validation errors.

### Sample identity and factors

Common fields are:

```text
schema_version, record_type, experiment_id, tier
sample_id, paired_world_id, public_instance_hash
world_suite, runtime, attacker, query_budget, seed
configuration_hash, code_revision
```

The configuration hash covers the scientific configuration, including resolved
sample count and any E2 condition, E4 workflow, or E5 ablation. It excludes the
output directory and overwrite policy, which do not change the experiment.

### Public transcript and attack outcome

Where applicable, records contain:

```text
agent_visible_transcript, transcript_hash
secret_prediction, secret_prediction_score, secret_prediction_correct
paired_transcript_equal, paired_transcript_exact_distance
selected_exploit
```

`agent_visible_transcript` is the only serialized observation stream available
to the protected session. It may include public actions and normalized tool
values, errors, receipts, identifiers, timing buckets, retry/termination events,
callbacks, visible memory, and settlement messages. Validation recursively
rejects trusted-only keys such as `private_state`, `theta`, `monitor_decision`,
`true_policy_label`, and `admissible` inside this field.

### Staging, safety, and utility

Depending on the experiment, records also include:

```text
true_policy_label, monitor_decision
staged_effects, committed_effects, prohibited_effect_indicator
task_success_indicator
salvage_numerator, salvage_denominator, eligible_action_salvage_rate
rejected_effect_commit_rate, rejected_descendant_commit_rate
dependency_violation, atomicity_violation
trusted_repair_success, truthful_report_accuracy
latency_ms, tool_call_count, model_usage
```

Top-level policy/monitor fields are trusted evaluation output, not transcript
content. Each staged effect describes its effect ID, dependencies, atomic group,
private evaluation, policy epoch, payload/destination digests, admissibility, and
commit-token digest. Payloads and destinations themselves need not be exposed.

`model_usage` is zero/`null` for deterministic Tier 1. Tier-2 adapters must
populate model identity and token counts rather than silently treating them as
zero.

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

## `manifest.json`

The manifest schema is `silenttwin.manifest.v1`. Required bindings include:

```text
status = complete
experiment_id
configuration and configuration_hash
expected_sample_count and actual_sample_count
result_file and result_sha256
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

## Aggregate output

Each aggregate directory contains at least:

```text
summary.json
summary.csv
```

Experiment-specific exports are:

```text
E1: inference_accuracy_vs_q.csv, transcript_auc_vs_q.csv
E2: prohibited_effect_rate_vs_q.csv, feedback_assisted_gain.csv
E3: inference_accuracy_vs_q.csv, transcript_auc_vs_q.csv,
    prohibited_effect_rate_vs_q.csv, feedback_assisted_gain.csv,
    privacy_safety_utility.csv
E4: privacy_safety_utility.csv
E5: privacy_safety_utility.csv, ablation_table.csv
```

`summary.json` preserves configuration groups, cohort hashes, bootstrap
confidence intervals, and matched paired comparisons. `summary.csv` flattens
configuration and comparison rows for analysis tools. Configuration hashes are
kept separate; comparison strata use matched public-instance cohorts. The shell
entrypoints pass the expected expanded-grid count, so an absent configuration,
orphan file, incomplete bundle, duplicate configuration hash, cohort mismatch,
or mixed source provenance fails aggregation.
