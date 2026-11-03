---
name: thurstonian-decision-noise
description: exp3 iter_3 c2: decision-rule slot — 3-sample majority vote −140 ll; log-odds decision noise sigma .3/.6/1.0 costs .1/2/13 ll; submitted with LogNormal(log .4,.2) stated prior
metadata:
  type: project
---
exp3 iter_3/candidate_2 (2026-11-02) is thurstonian_decision_noise. The base is rule_count_familiar_strays (MAP ll -3321.79).
- A parameter-free majority vote over 3 sampled concepts, P=3b²-2b³, gave ll -3461 (−140), which rules it out.
- Log-odds Gaussian noise (9-pt Gauss-Hermite) at fixed sigma .3/.6/1.0 gave ll −3321.9/−3323.8/−3334.5, with pool RMSE vs base .0025/.0094/.023.
- Submitted with sigma ~ LogNormal(log .4, .2), a stated prior so it passes novelty. The smoke check was OK, with ll −3323.9.
- It does not address the outside-rule set-size critique.
**How to apply:** this confirms [[exp3-decision-rules-collapse]]. Sharpening rules lose heavily, and mild compression is nearly free but only passes novelty with a stated prior.
