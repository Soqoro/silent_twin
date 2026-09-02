# Scientific-v6 model-free submission audit

## Disposition

The submission audit passes **451/451 checks** with no failed checks. It binds
the paper's displayed E1, adaptive-interface, forced-choice, E2, strict-repair,
and native-repair results to the exact frozen train artifacts. It also maps six
abstract/conclusion claims to the frozen claim ledger and confirms that every
analysis keeps confirmatory, development, and held-out permission closed.

- Machine artifact: `docs/scientific_v6_submission_audit_v1.json`
- Machine artifact SHA-256:
  `9eb1e67afd5478ea51d2482342797fa9596f67161bc4fbdbbba1c43dbdc2592a`
- Internal audit self-hash:
  `aa3802a1ff96dd1a04be171b562fa52865b73cfaad347370686d8efdfaa47a39`
- Bound manuscript SHA-256:
  `7b8c01f2bd6e63fea4bfa3b883e9db425f2e0badc44ba124a16f96c833908089`
- Raw E1 corpus digest:
  `4eabcc5342f0ab4c9a9d74874f7b95345d14ca89fac987650837ec4cfc187c80`
- Raw E2 corpus digest:
  `6d275ed9cb77a6254b4d8de55e54d735134e081be97694d6d60b8a0f8ec880e6`

This is a model-free integrity and reporting audit. It submits no scheduler
job, invokes no model, and reads no development or test result path.

## Reproduction

From the repository root:

```bash
/home/suaq0001/projects/.venvs/silenttwin-agentdojo-py311/bin/python \
  -m silenttwin.agentdojo.submission_audit \
  --production-root /home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production \
  --manuscript docs/silenttwin_feedback_recipient_separation_proposal.tex \
  --output docs/scientific_v6_submission_audit_v1.json \
  --progress
```

The deep audit streams and hashes all 392 E1/E2 shards, about 7.4 GB. On the
current filesystem it completes in roughly 16 seconds. A quick structural run
is available with `--skip-deep-raw-scan`, but it deliberately reports
`incomplete` and exits with status 2; only the deep run can pass.

Expected final summary:

```text
status=pass
checks=451 passed, 0 failed
audit_hash=aa3802a1ff96dd1a04be171b562fa52865b73cfaad347370686d8efdfaa47a39
```

## Exact evidence inventory

All paths below are relative to
`/home/suaq0001/projects/silenttwin-results/silenttwin-agentdojo-production`.
Analysis self-hashes are independently recomputed where the artifact schema
provides one.

| ID | Relative path | File SHA-256 | Role |
| --- | --- | --- | --- |
| `e1_analysis` | `scientific-v6-train/e1/aggregate/analysis_manifest.json` | `5edaf84a10b0775eb4d3f53ace1bf39a3aaef4abb1203076e311f7389a5c1c03` | Frozen E1 estimands, intervals, strata, and disposition |
| `e1_summary` | `scientific-v6-train/e1/aggregate/summary.json` | `b1c3efed9479f353a80669bb12c24d5e5df7363b80994acea53026f7b0622cd5` | Frozen E1 cells and accounting |
| `e1_validated_index` | `scientific-v6-train/e1/aggregate/validated_run_index.json` | `c494f81c91b65dd3a6b6fc1ea3c3fb571d4126292da5d1bc4e36d23a8457af6b` | Exact 288-shard E1 raw index |
| `interface_analysis` | `scientific-v6-interface-realization-train/analysis/interface-realization-analysis-v1.json` | `3cf902ec9d9792774efcde4b5316724c568e4b131a4207c39b3ef31c685010ed` | Adaptive interface-realization analysis |
| `forced_choice_analysis` | `scientific-v6-forced-choice-readout-train/analysis/forced-choice-analysis-v1.json` | `6391ce62d0e6e078d506b474f2f365b5e930fa1d1c75b9d572df74b6bf80eacf` | Paired source-readout analysis |
| `forced_choice_result` | `scientific-v6-forced-choice-readout-train/run/result.jsonl` | `b2e1afb28f43132cdd0cdbcc97393be5f832d3f87a0073870d9c1bdafddb7167` | Raw readout for the displayed score boundary |
| `e2_analysis` | `scientific-v6-train/e2/aggregate/analysis_manifest.json` | `ac162941f7382366eb4f23eca269c8a54a6d246354124da73d9e71f8dee0cfb4` | Frozen E2 estimands, intervals, strata, and disposition |
| `e2_summary` | `scientific-v6-train/e2/aggregate/summary.json` | `a82a50cf390efc512484d1bec04fcc5fb1cdb3c354c731e3fd1683f62167742d` | Frozen E2 cells and accounting |
| `e2_validated_index` | `scientific-v6-train/e2/aggregate/validated_run_index.json` | `f27f0944f852919bbcef546485af5cedc88b80de0e6b80a89c508bf360096314` | Exact 104-shard E2 raw index |
| `strict_repair_analysis` | `scientific-v6-clean-repair-train/analysis-v1.json` | `42fc7908ef04dd83ec15bfd9bbd5750012f619f2401401d05a45258f786f5574` | Strict-JSON feasibility analysis |
| `strict_repair_result` | `scientific-v6-clean-repair-train/run-v1/result.jsonl` | `139c2a25f22b905f2f0d58ff84d69c329fd5beffc3df41e0622ad840ab18f42d` | Strict-interface failure taxonomy |
| `native_repair_analysis` | `scientific-v6-native-tool-interface-train/analysis-v1.json` | `1b677b8fa3ef75ec1a44829a4a9d53909be9fc7f95d98ab041e0ebcd8292f5e4` | Native-tool qualification analysis |
| `native_repair_result` | `scientific-v6-native-tool-interface-train/run-v1/result.jsonl` | `d2f323fa6256908b07db50361b168c6209a50cc043b5e4c57616ff9aee2dba66` | Native semantic/tool failure taxonomy |

## Final result index

| Stage | Scale | Principal audited observation | Scientific disposition |
| --- | ---: | --- | --- |
| E1 controlled | 288 shards; 8,928 rows; 49 groups | Genuine Q=16 minus Q=0 accuracy gain 0.2460; the matched-shuffled equivalence control fails | Source-aligned signal is train-only mechanism evidence, not a permission-bearing gate |
| Interface replay | 1,860 rows; 49 groups | Primary explicit-grounding contrast is -0.1781; implicit simplification and opaque labels improve realization | Exact schema realization is part of the measurement |
| Paired source readout | 2,976 passes; 1,488 pairs | Minimal genuine target AUC and shuffled donor AUC are 1.000; target-negative controls remain at chance | Source alignment is supported in every suite under the paired readout |
| E2 controlled | 104 shards; 4,836 rows; 49 groups | Selection gain is 0.3004 with positive intervals in all suites; effect gain is 0.1553 but suite-heterogeneous | Action transfer is broader than released-effect realization |
| Strict repair gate | 245 rows; 147 learned episodes | 0/147 learned episodes satisfy the flattened contract; oracle succeeds 49/49 | Learned repair efficacy is not estimable |
| Native repair gate | 49 tasks; 123 calls | 122/123 calls parse, but protocol validity is 0.7443 (35/49 task-weighted) | Syntax improves, but the checkpoint/interface pair does not qualify |

## Claim-to-evidence map

| Claim ID | Abstract/conclusion statement | Ledger disposition | Evidence |
| --- | --- | --- | --- |
| `source_aligned_private_state_information` | The paired readout recovers source-aligned evidence in all suites | Supported only as controlled train-only mechanism evidence | E1, interface replay, paired readout |
| `state_matched_action_selection` | Genuine feedback changes the state-matched final action in every suite | Supported under the authored private-authorization intervention on train | E2 selection contrast and suite strata |
| `released_effect_heterogeneity` | Only Banking and Slack preserve the action change at the conservative effect layer | Supported only as an aggregate, suite-heterogeneous train estimate | E2 matched-control effect contrast and suite strata |
| `recipient_separation_partial_action_layer` | Prediction rises while selection and commitment remain at baseline | Supported descriptively; full empirical closure is not supported | E2 SilentTwin cells and failed closure gate |
| `learned_repair_not_qualified` | Native syntax largely succeeds but episode grounding remains below release | Checkpoint/interface pair is not qualified | Strict and native prospective gates plus raw taxonomies |
| `train_only_scope` | Useful repair and cross-model/held-out robustness remain open | No generalization, prevalence, or deployment claim | Access flags in all six analyses |

## Raw accounting and descriptive diagnostics

The deep scan validates every indexed result file against its per-run SHA-256
and row count before forming the corpus digest.

- E1: 3,815/8,928 rows have errors and carry 3,837 ledger entries: 3,740
  invalid hidden-state predictions and 97 invalid probe selections. The 22-row
  difference is the set carrying both entries.
- E2: 1,986/4,836 rows have errors and carry 1,994 ledger entries: 1,935
  invalid hidden-state predictions, 31 invalid probe selections, and 28
  invalid final plans. Eight rows carry two entries.
- Strict repair: 81 missing/exact-key contract failures, 50 non-JSON fenced
  responses, and 16 mixed tool-call/final-content responses.
- Native repair: one malformed JSON episode, eight invalid entity identifiers,
  three unsuccessful searches, one invalid email, and one missing argument.
  The 14 invalid episodes occur in 8 Workspace, 3 Slack, 2 Banking, and 1
  Travel task. Four otherwise protocol-valid Travel rows separately fail the
  frozen prompt-binding flag.

The raw E1 posterior audit now has a fully specified deterministic procedure:
scenario repetitions are averaged within structural group and state, suites
receive equal weight, and 5,000 structural-group bootstrap resamples are used.
It reproduces the manuscript's point estimates and yields the audited rounded
intervals: overall genuine target AUC 0.898 [0.853, 0.952], Travel 0.720
[0.570, 0.910], Workspace 0.870 [0.760, 0.968], shuffled donor AUC 0.901
[0.860, 0.954], and SilentTwin target AUC 0.510 [0.485, 0.540].

## Reporting corrections made by this audit

Two presentation-only issues were corrected without changing a frozen result
or claim:

1. The SilentTwin table's values 0.6022 and 0.0780 are pooled raw-row error
   rates (224/372 and 29/372), not equal-suite estimates. The column and caption
   now state that weighting explicitly.
2. The exploratory raw-posterior intervals were regenerated after canonical
   sorting under the now-documented deterministic bootstrap. Only interval
   endpoints changed slightly; all point estimates and interpretations remain
   unchanged.

Development, test, learned-repair efficacy, full empirical closure, and
cross-model or deployment generalization remain closed or open questions as
recorded in the manuscript claim ledger.
