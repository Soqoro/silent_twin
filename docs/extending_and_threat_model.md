# Extension guide and threat-model boundaries

## Invariants extensions must preserve

An extension is compatible only if it preserves these boundaries:

1. Adjacent paired worlds have identical canonical initial public projections.
2. Hidden state, monitor decisions, policy labels, and commit tokens never enter
   `AgentVisibleTranscript`.
3. A monitor allow is not substituted for a `TruePolicy` permission.
4. External mutations are staged until a separate controller authorizes commit.
5. Complete data and control dependency closure and atomic groups are enforced.
6. The protected session is retired before settlement and has no post-retirement
   tools, callbacks, real reads, memory, or trusted repair messages.
7. Output manifests bind schema, configuration, resolved sample count, code, and
   generation/evaluation provenance.

Named E5 ablations may violate exactly one of these for measurement. Keep such a
violation opt-in and visibly named; never make it a production runtime default.

## Add a paired-world template

1. Add `src/silenttwin/worlds/templates/<name>.py` using the existing templates
   as the minimal pattern. Keep public task content, initial public environment,
   and tool schemas in the public projection. Keep `theta`, state-specific
   monitor rules, and trusted labels outside it.
2. Give every proposal a deterministic action ID and explicit effectfulness,
   data dependencies, control dependencies, atomic group, and public payload.
3. Construct both adjacent worlds through `WorldPair`. Do not bypass its public
   projection validation or public-instance hash.
4. Register the suite in the world-template dispatcher and configuration
   validator. Add it to `WORLD_SUITES` in shell grids only after Tier-1 samples
   and tests exist.
5. Test public-projection equality, deterministic hashing, balanced hidden-state
   sampling, and absence of private keys in serialized visible events.

If two worlds need different public initial state, they are not a valid adjacent
pair for this estimand. Split them into different public instances rather than
masking the mismatch after construction.

## Add a runtime

1. Implement the common runtime interface under `src/silenttwin/runtime/`.
   Consume a public action proposal plus private policy/monitor evaluation, and
   return visible events separately from trusted records.
2. Build visible observations through the schema types. Do not attach an
   unrestricted metadata mapping that can carry trusted fields.
3. Specify how the runtime treats values, error shapes, identifiers, receipts,
   timing buckets, retries, termination, callbacks, visible memory, and
   settlement. If the runtime claims feedback privacy, each must be independent
   of private enforcement decisions in adjacent worlds.
4. Stage every external effect. Full SilentTwin must retire the protected
   session before passing effects and trusted records to the dependency-aware
   controller.
5. Register the runtime in configuration/dispatch and, when scientifically
   appropriate, in the corresponding shell grid and Tier-1 JSON reference.
6. Add paired transcript tests, retirement tests, and controller safety tests.

A baseline is allowed to leak by design, but its name and documentation must say
what is visible. A degraded SilentTwin variant should change one mechanism at a
time so its causal interpretation remains clear.

## Add a model client or realistic agent

Tier-2 clients belong behind the attacker/model interface, not inside world,
policy, or controller code. A client adapter should accept a fully public prompt
and tool schema and return an action or hidden-state prediction plus usage
metadata. It must not receive trusted evaluation records.

The current `ModelClient` protocol exposes
`complete(prompt, seed=..., max_tokens=...)` and returns `ModelResponse` (or a
string for minimal adapters). `LLMAttacker` is the integration example; keep the
deterministic `MockModelClient` available for offline tests.

At minimum, a Tier-2 adapter must provide:

- stable provider, model, endpoint class, and adapter version identifiers;
- explicit decoding parameters and seed support where the provider offers it;
- bounded retries/timeouts and a recorded terminal failure mode;
- request/response and cached-result digests without storing credentials;
- input/output token counts, tool-call count, and latency;
- a mock implementation for offline tests;
- an explicit opt-in before any network or billable request.

Credentials must come from the deployment secret manager or process environment
and must never enter a config, transcript, manifest, log, or result record. Keep
generation provenance distinct from evaluation provenance so the evaluator can
be changed without falsely presenting samples as newly generated.

For realistic tools, implement adapters under `src/silenttwin/tools/` that use a
temporary/sandbox environment during the protected session. The adapter must not
silently turn a simulated receipt into a real external mutation. The trusted
controller is the only component allowed to cross that boundary after
retirement.

## Add an experiment or metric

Use the existing runner to emit the shared sample schema and exactly one summary
record. New scientific factors must appear in the configuration hash, manifest,
leaf output path, grid line, and aggregation grouping. Output location and
overwrite policy are operational settings and should not alter scientific
identity.

Metrics should declare their unit, denominator, missing-data behavior, and
direction. Preserve task-level matching when comparing conditions. A new
aggregator must reject incompatible cohorts/configurations and compute paired
uncertainty when the design is paired.

## Threat model

### In scope

- An adaptive protected agent that chooses probes based on its complete visible
  transcript and then chooses a final exploit.
- Private guardrail/monitor state that differs across otherwise public-identical
  paired worlds.
- Feedback through values, errors, receipts, identifiers, simulated timing
  buckets, retries, termination, callbacks, visible memory, settlement, and
  post-retirement access.
- State-specific monitor false negatives that remain prohibited under the true
  policy.
- Unsafe descendants caused by missing data/control dependencies and partial
  atomic commits.
- Utility lost by discarding safe independent branches or by a low-fidelity
  temporary twin.

### Assumptions

- The trusted policy, evaluator, retirement mechanism, controller, token secret,
  and provenance store are outside the attacker's authority.
- The controller receives a complete dependency graph and correct policy epoch
  in the full condition.
- The temporary environment cannot mutate the real world before commit.
- All observations available to the protected process are represented in the
  visible transcript model; exact privacy is relative to that boundary.
- Canonical serialization and deterministic Tier-1 simulation accurately expose
  every modeled difference.

E5 directly relaxes several assumptions to measure their effect. Those results
diagnose the finite-state mechanism; they do not prove that an unmodeled channel
is absent from a deployment.

### Out of scope or not established by Tier 1

- Compromise of the trusted controller/evaluator, token keys, host kernel,
  hypervisor, scheduler, or secret manager.
- Hardware and co-tenancy channels such as cache contention, power, network
  micro-timing, or resource accounting below the modeled timing buckets.
- Information already present in the public task, model weights, pretraining,
  or prior sessions before the paired experiment begins.
- Collusion through an external system not mediated by the declared tool and
  callback interfaces.
- Elimination of monitor blind spots. Exact SilentTwin removes an additional
  feedback-assisted advantage; it cannot make an imperfect monitor correct.
- Generalization from deterministic mocks to a particular LLM, API provider, or
  production workflow without a separately specified Tier-2 study.
- Claims of perfect simulator fidelity. State-independent mismatch can harm
  utility even when it does not reveal private enforcement state.

## Deployment review questions

Before describing a Tier-2 system as SilentTwin, verify:

- Can the protected process observe real latency, IDs, error types, retries,
  quota, billing, callbacks, memory, or settlement outside the normalized API?
- Can any protected-session code touch real external state before retirement?
- Does every staged effect include all data and control ancestors and the correct
  atomic group/policy epoch?
- Is trusted repair performed in a fresh context that cannot communicate back to
  the retired one?
- Are public twin responses independent of private verdicts, including failure
  paths?
- Do manifests make provider/model/version and evaluator changes visible?

A “no” or unknown answer is a boundary to document and test, not a detail to
hide behind aggregate metrics.
