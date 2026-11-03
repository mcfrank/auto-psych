---
name: exp3-iter3-equal-prior-incumbent-probes
description: Exp3 iter_3 cand_4 MAP probes on equal_prior_familiar_rules_digit_sums (LL -3514.87) — re-exposure evidence weight and rule-size prior dead; round edges +0.8 (RMSE 0.0027, cand_3 took it); graded ends-in +0.88 RMSE 0.0049 submitted
metadata:
  type: project
---
2026-11-01, exp3 iter_3 candidate_4 (incumbent refinement slot). Base MAP LL -3514.87.
- Evidence weight exp(-g*log1p(exposure to the same set within a participant), 5 bins): g = -0.001, dead, even though the critique stat repeated_set_exposure_residual_slope had z = 1.5.
- Rule prior proportional to |rule|^b over the familiar rules: b = 0.007, dead.
- Round-number range edges: -3514.06 (+0.8), bonus -0.5, MAP RMSE 0.0027 vs base (probably under the novelty gate). Sibling cand_3 submitted equal_prior_rules_round_edge_ranges anyway.
- Graded "ends in d" within the ends-in share (exp(-digit_decay*d)): -3513.99 (+0.88), decay -0.029, RMSE 0.0049 vs base and 0.0064 vs graded_final_digit_digit_sum_concepts. Submitted as graded_final_digit_equal_rule_prior. The smoke check took about 5 minutes.
Probe scripts are in /tmp/c4i3 (nov.py has trial_index fixed for the pool).
**How to apply:** this base is saturated. See [[exp3-iter2-equal-rule-prior-probes]], [[exp3-iter3-critique-participant-structure]].
