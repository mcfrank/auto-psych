---
name: prior-leaking-judgment
description: experiment2 round 0 cand 0: judged logit = posterior logit - c * centred prior-membership logit of target; c≈-0.15, LOO +1.3 over best
metadata:
  type: project
---
experiment2 iter_0/candidate_0 (2026-10-30), prior_leaking_membership_judgment: gamma_width_exception_family_concepts + target-specific prior contrast. Quick 2x300 NUTS LOO -2277.1 vs best -2278.3 (+1.3 nats, 1 param). contrast ≈ -0.15 ± 0.07 (prior leaks in, not base-rate subtraction). Prior membership per target is cheap: SIZE_TARGET_FRAC (share of size-L intervals containing t) and rule prior · membership. Fit ~25 s.
All 8 iter_0 PPC statistics non-significant (incl. participant yes-rate SD, in-set yes rate — the model predicts 0.65 there).
**Why:** saves re-running. **How to apply:** small gain only; see [[experiment2-data-structure]], [[incumbent-residual-diagnostics]].
