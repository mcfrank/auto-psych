---
name: working-memory-three-example-limit
description: exp3 iter_2 c1 — 4-example sets judged on a 3-example subset's shared profile w.p. drop: +5.9 MAP on rule_ambiguity incumbent; size-principle tilt and local doubt ~0
metadata:
  type: project
---
exp3 iter_2 candidate_1 (2026-11-02). Model `working_memory_three_example_limit`: the rule_ambiguity_single_doubt incumbent with one change. With probability `drop` (fit at ~0.68), a 4-example set is judged on the profile shared by the 3-example subsets, averaged over which example was lost. MAP -4685.2 vs the incumbent's -4691.1. Check passed; smoke loglik -4669.3, 212 s for 200 iterations on 1 chain. Implementation: prepare_observed builds (trials x 5) feature matrices, one column per version of the set.

In the same round, two things gave ~0 on the incumbent:
- a size-principle tilt on property weights, (|h|/100)^-alpha, with or without scaling by n;
- making the n=1 doubt yes-rate decay with distance from the example (+1.1).

The incumbent's residuals are small everywhere. Pair-level z^2/df is 1.16, and the n=1 set z^2 sum is 80 over 51 sets, so the n1_set_mean_variance critique is mostly pair noise. The one clear residual pattern is that n=4 high-match targets are over-predicted (-0.06).

**Why:** later rounds should avoid re-testing these and stop chasing n=1 variance.
**How to apply:** for exp3 gains, look at set-size effects for n>=2. Related: [[leaky-memory-recency]], [[specificity-weighted-overlap]], [[rule-ambiguity-single-doubt]]
