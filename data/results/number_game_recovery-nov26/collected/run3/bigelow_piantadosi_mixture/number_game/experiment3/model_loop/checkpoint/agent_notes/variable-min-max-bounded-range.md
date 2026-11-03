---
name: variable-min-max-bounded-range
description: exp3 iter_3 c5: variable-lambda graft on min_max_bounded_range +1.35 MAP vs incumbent, RMSE .008; dense-rules-only restriction −1.1
metadata:
  type: project
---
exp3 iter_3/candidate_5 (2026-11-02) is variable_min_max_bounded_range: min_max_bounded_range with logit(lambda) ~ N(mu, sd) and 9 GH nodes. The hull counts as a range kind, with weight (1-lam).
- MAP ll: −3317.89, vs −3319.24 for the incumbent and −3319.10 for min_max. boost = 6.8, lambda_sd = .68.
- Pool RMSE vs the incumbent: .0083.
- Allowing range-limited versions only for rules with more than 25 members (aimed at the outside_rule_member_setsize_contrast critique) loses: ll −3320.18.
- Sibling incumbent slots c3 and c4 could make the same hull graft (see [[check-sibling-candidates-first]]).
See [[min-max-bounded-range]], [[variable-rule-range-expectation]].
