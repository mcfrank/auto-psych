---
name: exp3-iter4-critique
description: Exp3 iter-4 critique of position_decaying_example_evidence — 8 stats (rule-violating interior, rule-matching, up/down 1ex, Weber edge, prefix re-test, 2ex gap fill, 4ex-2ex interior, multiples of lone example); iter-3 got 1/8 (prefix)
metadata:
  type: project
---
Iter-3 critique (2026-10-31) found 1/8 significant: prefix_extension, obs 0.287 vs null 0.106, p=0.014. It rests almost entirely on 31,3,1 → 31,3,1,15. The new incumbent's decay rate rho is null and it drops the Weber/heading/upward terms, so for iter 4 I re-tested prefix extension and probed the dropped terms.
The stats come from /tmp/mk4.py, which shares a numpy header. Observed values:
- rule_violating_interior_yes 0.114 (n=385)
- rule_matching_target_yes 0.96 (n=125)
- single_example_above_minus_below -0.106
- near_extrapolation_large_minus_small_edge +0.145
- fourth_example_prefix_broadening 0.287
- two_example_interior_wide_minus_narrow_gap -0.183
- interior_yes_4ex_minus_2ex_matched_span -0.065
- single_example_multiple_target_yes +0.367

Each takes about 0.02 s per call. Data quirks: some shown examples get rejected (88 in 70,10,88,40 → 0.0; 29 in 29,77,37,17 → 0.2). The model has no decade concepts; "ends in 0" stands in for multiples of 10.
See [[exp3-iter3-critique]], [[critique-stat-no-imports]].
