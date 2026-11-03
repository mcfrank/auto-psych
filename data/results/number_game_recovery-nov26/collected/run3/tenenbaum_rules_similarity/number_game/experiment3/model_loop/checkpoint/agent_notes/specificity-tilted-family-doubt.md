---
name: specificity-tilted-family-doubt
description: exp3 iter_3 c4: specificity_biased_rule_prior's size^-s within-family prior grafted onto rule_family_doubt_criterion; smoke OK ll −3653.1
metadata:
  type: project
---
exp3 iter_3/candidate_4 (2026-10-31), specificity_tilted_family_doubt: the incumbent rule_family_doubt_criterion plus the within-family rule prior ∝ size^-specificity (Normal(0,1)), built by text-patching the incumbent source. Smoke check OK in 18 s of sampling, ll −3653.1. The iter_3 critique had 0/8 significant stats, so this graft had no critique motivation. Posterior and novelty not checked. On exp2, s came out ≈0.1, so novelty RMSE vs the incumbent may be thin.
Raw exp3 fact: shown-example targets get 0.66 yes (190 rows), and the incumbent already predicts 0.66 at its initial point, so no lapse term is needed for that.
**How to apply:** if it is rejected as a duplicate, within-family size tilts are dead on exp3 too. See [[specificity-biased-rule-prior]], [[rule-family-doubt-criterion]].
