---
name: random-looking-strays
description: exp2 iter_2 c2: strays drawn ∝ 1/familiarity (opposite of familiar strays), motivated by critique familiar_vs_plain_outlier contrast; smoke OK 13 s, ll -2037
metadata:
  type: project
---
experiment2 iter_2/candidate_2 (2026-11-01), named random_looking_strays. It copies familiar_strays_family_level_prior, but the stray mix component is U(x) ∝ 1/(mean_r 1[x∈r]/|r|), which puts about 9x more mass on unremarkable numbers than on 64 or 8. It was motivated by a PPC critique: people excuse familiar outliers *less* than the model predicts (z=-2.4, q=0.16).
Smoke fit: 13 s for 100+100 iters, loglik -2037.3 (incumbent smoke about -2034).
**How to apply:** check the admission/ELPD outcome before trying stray-shape variants again. See [[familiar-strays-family-level]], [[stray-generator-variants]].
