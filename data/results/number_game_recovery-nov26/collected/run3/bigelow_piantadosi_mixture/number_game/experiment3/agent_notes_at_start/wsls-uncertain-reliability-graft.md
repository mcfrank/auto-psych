---
name: wsls-uncertain-reliability-graft
description: WSLS running-guess model with the incumbent's Beta-marginalised stray likelihood — exact via (k,size) interval counts per prefix
metadata:
  type: project
---
iter_4/candidate_5 (2026-10-30): uncertain_reliability_win_stay_lose_sample = win_stay_lose_sample_concepts (Dirichlet rule prior) + uncertain-reliability (n,k,s) table. Interval quantities that are parameter-dependent (prefix evidence, share consistent with later examples, share also containing the target) are stored as (k,size) counts: int_count[S,t,5,100], cons_count[S,t',t,5,100], cell_count[C,t',5,100]. Verified against the original WSLS at alpha=.9, kappa=e^8 (max diff 4e-4, O(1/kappa)). Check OK, smoke loglik -831.4; the smoke check took 10 min wall (it shared the CPU with my test).
Gotcha: kappa=e^18 breaks gammaln precision (spurious 0.7 diffs); use about e^8 for limit tests.
**How to apply:** see [[wsls-exact-recursion]], [[interval-tempering-speed-trick]] (use replace_rvs_by_values when evaluating p_yes).
