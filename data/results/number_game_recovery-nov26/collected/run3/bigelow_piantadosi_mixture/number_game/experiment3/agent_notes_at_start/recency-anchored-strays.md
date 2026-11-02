---
name: recency-anchored-strays
description: exp2 iter_4 c1: last-example anchor on familiar_strays_equal_rule_prior — MAP -1.4 ll vs incumbent, RMSE .010, smoke OK 20 s
metadata:
  type: project
---
exp2 iter_4/candidate_1 (2026-11-01) is recency_anchored_familiar_strays. It is the equal-rule incumbent with the first-example anchor code from familiar_strays_first_example_anchor, but the anchor is the last-shown example (s[-1]) and gets its own trust a_last. At MAP: ll -2033.7 against the incumbent's -2032.3; a_last .92 ≈ alpha .91. p_yes RMSE against the incumbent is 0.010, so it is distinct. The check passed, with a 20 s smoke fit and loglik -2035.7.
The data contain only one permuted pair (8,16 / 16,8), so order effects are barely identified.
**How to apply:** anchors at either end (first or last) do not help this incumbent. Don't retry position anchors. See [[familiar-strays-anchor]], [[first-example-anchored-reliability]].
