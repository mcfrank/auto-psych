---
name: anchored-rule-deviations
description: exp2 iter_1: incumbent first_example_anchored_reliability + per-rule logw (fixed spread 1.75) = +20.8 MAP ll; anchor alone adds only +1.6 over family_pooled_uncertain_reliability
metadata:
  type: project
---
experiment2 iter_1/candidate_3 (2026-11-01), named anchored_rules_individual_plausibility. It is the incumbent with rule_logw ~ N(family_level, 1.75) restored, built on the fast flat-table + sparse-dot layout from [[fast-incumbent-sampling]].
MAP log-likelihoods: incumbent -2031.5; family_pooled_uncertain_reliability -2012.3; new model -2010.7. Pool RMSE is 0.038 against the incumbent and 0.013 against family_pooled.
Speed: 2-3 ms/grad. The check passed, with smoke fit 15 s/200 iters and loglik -2034.2.
**Why:** the incumbent's ELPD lead comes from parsimony (11 params). The anchor (a1 ≈ alpha ≈ 0.9) explains almost nothing once rules carry their own weights.
**How to apply:** check this slot's admission/ELPD result before trying more anchor variants. See [[first-example-anchored-reliability]].
