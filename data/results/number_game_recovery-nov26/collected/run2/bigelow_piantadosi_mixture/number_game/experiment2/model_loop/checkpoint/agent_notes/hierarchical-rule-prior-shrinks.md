---
name: hierarchical-rule-prior-shrinks
description: Partially pooled (log-normal, non-centred) rule saliences on the equal-prior lapse model shrink to tau≈0.15; ELPD ≈ incumbent
metadata:
  type: project
---
Round 2 candidate_3 (2026-10-30), `hierarchical_rule_salience_with_guess_lapse`: incumbent `equal_prior_concepts_with_guess_lapse` + log w_r = tau*z_r, tau~HalfNormal(1). Smoke loglik -905.3 (incumbent -905.8); 2x500 fit ELPD-LOO -907.9 (incumbent -908.2), p_loo 5.2, tau 0.15±0.12. Fast: 200 iters in 10 s.

**Why:** data give no support for rule-salience heterogeneity once lapse is in; free 45 weights' in-sample gain is overfitting.
**How to apply:** don't try further rule-prior variants (family weights, hierarchical, free). Novelty-gate risk: predictions ~identical to the incumbent. See [[number-game-ceiling]].
