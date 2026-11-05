---
name: lone-example-digit-discount
description: exp3 iter1 c3: final-digit attention discounted for single-example sets on whole_set_round_category, +20 nats MAP (1 param)
metadata:
  type: project
---
Refined whole_set_round_category: c_unit × lone_digit (Beta, MAP 0.38) when n=1. MAP log-lik -4078.4 vs incumbent -4098.3; smoke -4085.6.
Residuals at MAP: single sets near-diff-digit obs .45 vs pred .32, far-same-digit .70 vs .79. Free single-example magnitude width (+0.2), n-scaled magnitude tightening (+0.1), exact-example recognition bonus (+2) all no help.
**Why:** single-example digit misfit was the largest per-category residual; it also targets near_target_set_size_contrast [[exp3-iter1-critique-stats]].
**How to apply:** for future rounds, build on this if it's admitted; don't retry the null variants. Gotcha: to get p_yes at MAP, use model.replace_rvs_by_values([p_yes]) then compile_fn(inputs=value_vars).
