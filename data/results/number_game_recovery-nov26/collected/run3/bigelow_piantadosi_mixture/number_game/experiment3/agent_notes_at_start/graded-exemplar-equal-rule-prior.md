---
name: graded-exemplar-equal-rule-prior
description: exp2 iter_4 c5: equal 45-rule prior on graded_exemplar_familiar_strays — family levels ≈0, −0.86 MAP ll, saves 6 params, RMSE .008 vs parent
metadata:
  type: project
---
exp2 iter_4/candidate_5 (2026-11-01) is graded_exemplar_equal_rule_prior: graded_exemplar_familiar_strays with family_level removed (log prior = log lambda_0 − log 45).
At the parent's MAP the family levels are all within ±0.15. MAP ll is −2030.05 against the parent's −2029.19. The p_yes RMSE against the parent is .008 (max .04), above the .002 novelty threshold. The smoke fit gave ll −2032.1 at 16 s per 200 iterations.
Expected ELPD is about −2043, which is a parsimony gain but probably still behind the incumbent (−2037.8).
The critique (familiar_outlier_excuse_contrast, q=.19) goes the random-looking-strays way, but that idea has already lost twice.
For a MAP comparison, prepare once and pm.set_data both models. Pass only value_vars to the compile_fn point.
See [[equal-rule-prior-parsimony]], [[graded-exemplar-familiar-strays]].
