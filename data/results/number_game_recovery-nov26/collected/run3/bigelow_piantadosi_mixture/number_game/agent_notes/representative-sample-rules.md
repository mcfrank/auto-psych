---
name: representative-sample-rules
description: exp3 iter_2 c1: rule likelihood × P(span this tight | rule)^beta on variable_rule_range_expectation: beta .15, +0.96 MAP, narrow fill-in .41→.42 only
metadata:
  type: project
---
exp3 iter_2/candidate_1 (2026-11-02), representative_sample_rules. Representativeness bias on whole-line rules only. The term is log P(span of contained examples ≤ observed | iid draws from the rule's members), computed exactly in prepare_observed (window count (c_all/k)^n − (c_rest/k)^n). It enters as beta·table, beta ~ HalfNormal(1).
- MAP ll: −3318.28 vs −3319.24 for the incumbent. beta = .15. Training p_yes RMSE vs the incumbent: .008.
- Narrow-span in-hull non-member yes: observed .58 (90 rows, my filter), prediction .407→.421. The critique gap is still open.
- Data hint: odd-only narrow sets (31,3,1,15 → .93; 73,35,53,59 → 1.0) drive the gap. The normative odd rule ties the [1,31] range (ratio ~1.7), so the fill-in prediction sits around .4.
**How to apply:** a likelihood-side clustering penalty does not collapse but only moves the gap a little, like [[cluster-cued-range-salience]]. See [[variable-rule-range-expectation]].
