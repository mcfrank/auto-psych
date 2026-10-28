---
name: shared-final-digit-similarity
description: exp3 iter_3 c1: union "ends in any example's last digit" concept added to incumbent; MAP LL -0.9 vs incumbent, RMSE 0.003; check OK -3093.8
metadata:
  type: project
---
2026-10-22, exp3 iter_3 candidate_1 (`shared_final_digit_similarity`, similarity brief, exploratory). It is the incumbent sequence_opens_bottom_ends_top plus one concept per set: all numbers whose ones digit matches any example's digit. The concept applies only when the examples have 2 or more distinct ones digits, and it gets its own 4th Dirichlet share. It targets the critique shared_ones_digit_outside_gap (q=0.18). At MAP the share is small: LL -3090.5 vs the incumbent's -3089.7, and in-data p_yes RMSE 0.0031 (novelty risk). Check OK: smoke LL -3093.8, 238 s per run.
**Why:** Digit-union concepts don't improve the fit. **How to apply:** Check novelty/LOO before trying more digit-similarity concepts. Related: [[digit-sum-concepts]], [[concept-space-saturated]], [[exp3-iter3-critique-stats]]
