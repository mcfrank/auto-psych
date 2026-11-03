---
name: variable-expectation-threshold-ranges
description: exp3 iter_3 c3: set-size critique not fixable globally; n-dependent lambda mean/sd, rho variability, WOR, run-uniform rr prior all collapse/lose; one-sided graft +0.16 submitted
metadata:
  type: project
---
exp3 iter_3/candidate_3 (2026-11-02) is variable_expectation_threshold_ranges: the one-sided threshold share tau grafted onto variable_rule_range_expectation (incumbent MAP −3319.24).
- Result: MAP −3319.08, tau .031. Pool RMSE .006 vs the incumbent, so it is distinct. Check OK; the smoke fit took 70 s.
- Critique outside_rule_member_setsize_contrast: the data are .52 vs .57 predicted at n=2 and .86 vs .82 at n≥3. The gap is item noise (4,16,12→32,64; 7,63→84,91,98; 81,25→100). Nothing below moves it.
MAP screens on the incumbent:
- lambda mean shifted by set size: beta −.07, +0.44.
- lambda sd scaled by set size: collapses.
- Per-judgement rho variability: rho_sd → 0.
- Without-replacement size principle: −5.0.
- Uniform prior over runs for restricted rules: −0.5.
- Gist graft: −3319.52, and lambda_sd shrinks to .31, so the two substitute for each other. Sibling c4 took this graft; c5 took min-max + variable lambda.
Speed trick: aggregate each concept kind first (logsumexp over rules plus membership share), then apply the lambda quadrature on a (cells, 3, K) tensor. It reproduces the incumbent's ll exactly and costs little.
**How to apply:** the incumbent's variants are saturated at about ±0.5 MAP. See [[variable-rule-range-expectation]], [[one-sided-threshold-ranges]], [[check-sibling-candidates-first]].
