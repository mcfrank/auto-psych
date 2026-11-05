---
name: set-round-kinship-averaging
description: exp3 iter0 c4: round route on round_number_averaging scaled by (fraction of 0/5 examples)^rho; smoke -4106.0
metadata:
  type: project
---
Exp3 iter0 candidate_4 refines [[round-number-averaging-exemplar]] to address the mixed_set_round_target_contrast critique ([[exp3-iter0-critique-stats]]). Round kinship route × frac_round^rho, rho~HalfNormal(2). Raw data: round far targets in mixed sets get a low yes-rate (2-ex 0.5 frac: 0.17; 4-ex 0.5 frac: 0.03; all-round 4-ex: 0.56). Smoke fit log-lik -4106.0, check OK.
**Why:** a lone 5-ending example in a mixed set over-boosts round targets.
**How to apply:** if this wins, set-level gating of the other arithmetic routes is the next thing to try.
