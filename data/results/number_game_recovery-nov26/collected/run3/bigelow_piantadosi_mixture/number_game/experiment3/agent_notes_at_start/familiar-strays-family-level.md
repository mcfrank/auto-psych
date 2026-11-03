---
name: familiar-strays-family-level
description: exp2 iter_1 c5: familiar_number_strays with incumbent's family-only prior; MAP -2030.98 vs incumbent -2031.46; fixed-tau models lost >100 ELPD despite good MAP
metadata:
  type: project
---
experiment2 iter_1/candidate_5 (2026-11-01) is familiar_strays_family_level_prior. It is familiar_number_strays (strays are a uniform/equal-rule mix, mix≈.26) with rule_dev removed: family levels are a ZeroSumNormal, as in the incumbent. It has 10 params, MAP ll -2030.98 against the incumbent's -2031.46, and pool RMSE .016 from the incumbent. The smoke run took 14 s per 200 iters.
Key observation: in exp2 iter_0, every FPUR refinement with a fixed tau=1.6 plus 45 centred rule_dev scored badly at ELPD (familiar -2179, strong_weak -2189, kind_specific -2124), even though their MAP ll of about -2010 beat the incumbent's -2031. The incumbent's first-example anchor fits a1≈alpha, so it gains nothing at MAP; it wins through parsimony (no rule_dev).
**How to apply:** don't add per-rule deviations with a fixed spread; build on family-only weights. See [[fast-incumbent-sampling]], [[tau-funnel-speed-fix]], [[exp2-incumbent-refinements]].
