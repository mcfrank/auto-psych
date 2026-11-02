---
name: exp3-iter3-decision-rule-probes
description: Exp3 iter_3 decision-stage probes on equal_prior_familiar_rules_digit_sums — coverage-relative criterion dead; Prelec weighting +0.93 (submitted); set-size gain +1.2 with 3 params
metadata:
  type: project
---
experiment3 iter_3 candidate_2 (2026-11-01). Base equal_prior_familiar_rules_digit_sums has MAP LL -3514.87.
- Set-relative criterion: logit b - w*(logit mean_b over 1..100 - c). w = -0.01, no gain. Dead.
- Prelec w(b)=exp(-beta(-ln b)^gamma): gamma 1.035, beta 1.008, LL -3513.94 (+0.93), data RMSE vs base 0.0054. Submitted as prelec_weighted_belief_response.
- Logit gain exp(g0+g1(n-2.5)) plus bias c0: g0 0.028, g1 0.028, LL -3513.63 (+1.2), RMSE 0.007.
Probe scripts are in /tmp/c2p (map.py takes a candidate and the responses CSV and saves the MAP p_yes).
**Why:** belief is already calibrated, so decision-stage tweaks gain about 1 nat at most.
**How to apply:** treat decision-rule slots as near the noise ceiling. See [[exp3-decision-rule-probes]] and [[exp3-iter2-novelty-gate-map-vs-posterior]].
