---
name: exp3-spacing-sparsity-and-nulls
description: exp3 iter1 cand2 — mean example gap logit shift +5.6 MAP on fives incumbent; fitted stray rate, contains bonus, order momentum, interval-only size weight all null
metadata:
  type: project
---
2026-10-31, exp3 iter_1/candidate_2 (`example_spacing_sparsity_bias`). The base is fives_privileged_decoupled_prime_odd, MAP loglik -3436.7, and it takes 25 s with L-BFGS (/tmp/c2s/mapfit.py).
- Logit shift × (mean sorted gap/100 − 0.25), singletons gap 0: +5.6 (coef −0.60). With singletons held neutral: +4.1. Using span instead of mean gap: +3.3.
- Null: fitted stray rate (goes to 0.094), contains-d bonus (−0.08), order momentum/trend extrapolation (+1.0), interval-only size-principle weight (0.97, +0.6).
- Response-level feature sweep: every raw cue gains ≤3.1 (round10 is driven by 8,64). The incumbent is near ceiling.
- Check OK, smoke loglik −3438.7, about 135 s per chain.

**Why:** the first per-set residual table was garbage because p_yes was compiled from RVs. Use model.replace_rvs_by_values ([[factor-graded-multiples-finding]] gotcha).
**How to apply:** the 83,38,58,68 "residual" was this artefact; the model fits it. Related: [[concept-space-saturated-exp3]], [[exp3-decision-stage-screens]].
