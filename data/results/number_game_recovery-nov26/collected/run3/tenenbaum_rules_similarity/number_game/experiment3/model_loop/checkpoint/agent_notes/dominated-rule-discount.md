---
name: dominated-rule-discount
description: exp2 iter_2 c4 repair: discount a fitting rule when a strictly narrower rule also fits the examples — log-weight −0.29±0.34, LOO −0.8, RMSE vs incumbent 0.0048
metadata:
  type: project
---
experiment2 iter_2/candidate_4_repair_1 (2026-10-30), dominated_rule_discount_concepts on beta_share incumbent. Replaced a within-family specificity tilt that was rejected (RMSE 0.0000 vs specificity_biased_rule_prior). Per-set indicator: rule fits all examples and a strictly-narrower non-exception rule also fits; add log-weight × indicator.
2x300 NUTS: weight −0.29±0.34 (null-ish), LOO −2274.3 vs incumbent −2273.6; posterior-mean p_yes RMSE vs incumbent 0.0048, vs specificity_biased 0.0075. Check OK, ~2.5 min.
**Why:** the 4th null in a row on the nested_multiple_specificity gap ([[specificity-tilted-rule-prior]], [[size-weighted-within-kind-rules]], [[multiple-base-specificity]]).
**How to apply:** treat the nested-multiples gap (q=0.28) as noise; stop using rule-prior reweighting to chase it.
