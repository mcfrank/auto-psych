---
name: set-size-evidence-exp3
description: exp3 iter0 cand3 — evidence weight (n/3)**g on the repdigit incumbent gains +9.2 loglik MAP (g≈0.29); a singleton-only weight gains +1.9
metadata:
  type: project
---
2026-10-31, exp3 iter_0/candidate_3 (`set_size_graded_evidence_weight`). The incumbent repdigit_pattern_chart_concepts has MAP loglik -3466.7. The two variants scale each set's loglik:
- (n/3)**g: -3457.6, with g≈0.29.
- exp(g) for singletons only: -3464.8, about null.
The critique's singleton near/far gradient (q=0.29) is only weak support. Only 300 rows are singletons; after one example people say yes to far targets about 30% of the time and to near ones about 37%.
The exp2 version (on a non-grouped base) was dropped as too slow, not on its merits; this one uses the grouped-interval base.
Check printed OK, smoke loglik -3463.9, 107 s for the 200-draw chain.
A working MAP screen is /tmp/c3s/map2.py: L-BFGS on the pymc logp gradient. pm.find_MAP gave garbage here.

**Why:** the same mechanism has replicated across experiments at roughly +5 to +9 nats.
**How to apply:** don't re-screen it. Related: [[set-size-amplified-evidence]], [[grouped-interval-speedup]].
