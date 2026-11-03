---
name: equal-rule-incumbent-scan
description: exp2 iter_4 c3: one-change variants of familiar_strays_equal_rule_prior all tie or lose at MAP; submitted rule-count familiarity (0 extra params)
metadata:
  type: project
---
exp2 iter_4/candidate_3 (2026-11-01). The incumbent familiar_strays_equal_rule_prior has MAP ll -2032.28 (alpha .917, kappa 18, mix .28, lambda_0 .19). Variants, as MAP ll and p_yes RMSE against the incumbent:
- Final-digit binding error, target only: eps→0.012, -2031.99, RMSE .003.
- Erlang interval prior: -2035.4, sigma≈38. The uniform interval prior wins again.
- Digit-slip strays as a third stray type (exact; slip factor ∏(1+c N_h/q0), subset-T tables for intervals): eps→.03, -2032.64.
- Familiarity = rule count (no 1/|r|): -2032.44, RMSE .0075. **Submitted** as rule_count_familiar_strays. Check OK, 11 s smoke.
- Familiarity weighted by 1/|r|^2: -2033.15.
The critique's familiar_outlier_excuse_contrast is driven by 31,19,49,100 (odd targets 0.27 observed vs 0.5 predicted). None of the variants moved it.
Memory gotcha: never keep per-set (5050 x 880) interval term arrays across sets. Reduce to cells inside the set loop.
**How to apply:** the stray-shape, interval-prior and readout axes are exhausted on this incumbent. See [[exp2-iter2-incumbent-saturation]], [[equal-rule-prior-parsimony]].
