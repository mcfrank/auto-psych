---
name: exp2-iter4-order-probes
description: Exp2 iter_4 order-effect probes on round_number_range_boundaries — geometric position gradient pinned at 0 (±0.1 costs 30-50 nats); first-example weight 0.96 gives +1.1 nats
metadata:
  type: project
---
iter_4 candidate_1 (2026-11-01, order brief), base round_number_range_boundaries MAP LL -2244.54 (full 5302-concept matmul with per-position log-lik (sets,4,concepts) reproduces it; MAP ~80 s):
- Geometric position weight exp(s*(j-1)) on each example's log-lik (unnormalised): s 0.005, LL -2244.47, pool RMSE 0.0015 (near-duplicate). Sharply identified: fixed s=-0.15 costs 53 nats, s=+0.1 costs 33.
- First-example-only log-weight a (later examples weigh 1): a -0.042, LL -2243.44 (+1.1), pool RMSE 0.0065. Submitted as first_example_evidence_weight. Untested: whether it is really a singleton-set effect (23 one-example sets) vs a last-example weight.
Speed: the full (sets,4,5302) version took 0.058 s/grad (smoke check 9.5 min). Fix: the first example has the same likelihood for every member of a (width, round-edges) group that contains it, so split each group per set into contains-first/not, and compute membership only for the asked targets (sets, subgroups, T): 0.023 s/grad, identical MAP.\nProbe scripts /tmp/i4c1/map.py (MAP LL), pool.py (novelty-pool RMSE at MAP); may not persist.
**How to apply:** order gradients are dead; see [[exp2-memory-and-prior-anchoring-probes]], [[exp2-map-probe-results]].
