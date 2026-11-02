---
name: two-source-variable-rule-expectation
description: exp3 iter_2 c4 repair: two_source graft on the variable-lambda incumbent: −0.9 MAP vs incumbent, pool RMSE .0028 (thin); 31,3,1,15→2,4,16 is likely the 2^k−1 pattern
metadata:
  type: project
---
exp3 iter_2/candidate_4_repair_1 (2026-11-02) is two_source_variable_rule_expectation. It is two_source_range_limited_rules with logit(lambda) ~ N(mu, sd) and 9 GH nodes. Since lambda enters only through the class weights, it reduces to (sets,3)@(3,K), so it is cheap: the smoke fit took 17 s.
- MAP ll: −3320.13, vs −3319.24 for the incumbent and −3321.13 for two_source_range_limited. lambda_sd = .60, mix = .036.
- Novelty-pool RMSE: .0028 vs the incumbent (only just over the .002 gate) and ≥.0058 vs the other models.
- The first attempt (tightness cue on the lambda mean) was rejected as a near-dup (RMSE .0005) of sibling c3; see [[check-sibling-candidates-first]].
Data hints from narrow-span in-hull non-members:
- 31,3,1,15 → 2, 4, 16 are all yes (1, 3, 15, 31 = 2^k − 1).
- 73,35,53,59 → 36 is yes 5/5, and 63,23,43 → 34 is yes 3/5. Both targets contain a digit shared by the examples, so digit-sharing may drive the fill-in.
See [[variable-rule-range-expectation]], [[two-source-range-limited-rules]].
