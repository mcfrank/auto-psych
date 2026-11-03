---
name: equal-rule-prior-parsimony
description: exp2 iter_3 c5: family levels of familiar_strays are ~0 at MAP; equal 45-rule prior loses 1.3 MAP ll, saves 6 params, RMSE .010; trial/participant effects are null
metadata:
  type: project
---
exp2 iter_3/candidate_5 (2026-11-01) is familiar_strays_equal_rule_prior. At MAP, familiar_strays_family_level_prior has all 7 family levels within ±0.27, so its rule prior is effectively flat. With an equal prior over the 45 rules it has 4 params and MAP ll -2032.3 (against -2031.0), with RMSE .010 from the parent. A 4x1000 NUTS fit gave per-row LOO -2035.8 (p_loo 3.5) against the parent's -2040.6 (p_loo 9.9), so the parsimony gain is real. The novelty gate threshold is RMSE 0.002 (model_zoo.DEFAULT_NOVELTY_RMSE_THRESHOLD).
Dead ends this round:
- Graded endorsement readout g^d on familiar strays: g→0.018, a dup.
- Strays from the learner's prior predictive, with the rule share set equal to lambda_0 plus interval-generated strays: −18 to −20 ll. The interval hump makes 1 and 100 hard to dismiss.
- Trial-position drift in yes-rate, and per-participant bias: the 10 participants are blocks of 450 rows in two 5-person sequence groups. Both are explained by the stimuli (+2.9 ll from 10 biases).
**How to apply:** for parsimony-driven ELPD gains, remove weakly identified params and check that the RMSE stays above 0.002. See [[exp2-iter2-incumbent-saturation]], [[familiar-strays-family-level]].
