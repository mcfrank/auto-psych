---
name: two-source-cluster-cued-ranges
description: exp3 iter_2 c5: cluster cue on two_source_range_limited_rules −3319.14 MAP (+2.0 over parent, ties incumbent); siblings c3/c4 both took incumbent+cluster graft
metadata:
  type: project
---
exp3 iter_2/candidate_5 (2026-11-02), two_source_cluster_cued_ranges. It is two_source_range_limited_rules with the range class weight multiplied by exp(beta·tight) per set, so w_class becomes (sets, class).
- MAP ll: −3319.14 (beta .50, mix .04). The parent gets −3321.13 and the incumbent variable_rule_range_expectation −3319.24.
- RMSE: .014 vs the incumbent, .008 vs incumbent+cluster. Check OK.
- First tried incumbent + cluster cue: −3317.61, the best MAP. But siblings c3 AND c4 both submitted that exact graft, c4 under the same name. So I switched.
- Also tried variable lambda on two_source: −3320.13 (+1.0 over parent), RMSE .006 vs the incumbent.
- Cluster cue and variable lambda gains roughly add, about +1.5 each.
**How to apply:** check sibling model_name.txt BEFORE building, every time ([[check-sibling-candidates-first]]). See [[cluster-cued-range-salience]], [[variable-rule-range-expectation]], [[two-source-range-limited-rules]].
