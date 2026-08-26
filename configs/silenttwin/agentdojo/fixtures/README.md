# Deterministic AgentDojo engineering-smoke fixtures

The two JSON artifacts in this directory are synthetic protocol fixtures. They
encode a deterministic two-by-two monitor truth table and suite-specific
no-argument public read calls so that catalog/split binding, pair freezing, grid expansion, shell
orchestration, checkpointing, and aggregation can be exercised without model
inference. They are **not scientific evidence** and must never be used to make
claims about leakage, bypasses, AgentDojo attack success, monitor quality, or
mitigation efficacy.

The checked `controlled-fake-smoke-v1.json` plan binds their exact scientific
artifact hashes. The grid loader rejects either fixture when paired with a
non-fixture configuration, and it rejects an incomplete fixture claim
boundary. The experiment shell helper selects these files only for that exact
checked smoke plan.

Production plans retain the operator-owned defaults:

- `configs/silenttwin/agentdojo/candidate-strategies-v1.json`
- `configs/silenttwin/agentdojo/pair-registry-v1.json`

Those files are intentionally not supplied by the smoke fixtures. Create them
from benchmark-authored train-frozen strategies and train/development monitor
observations with `run_agentdojo_pair_mining_tier2.sh`; held-out test outcomes
must remain unobserved until execution.
