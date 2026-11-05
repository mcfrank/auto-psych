---
name: times-table-round-averaging
description: exp2 iter4 c5 refinement: round-number route grafted onto times_table_averaging_exemplar, smoke -2666.3
metadata:
  type: project
---
Exp2 iter_4 candidate_5 (2026-11-02): `times_table_round_averaging` = [[times-table-averaging-exemplar]] + t%10==0 & e%5==0 distance-free route from [[round-number-kinship-exemplar]]. Smoke log-lik -2666.3 (base smoke -2677.5), 12 params, ~10 s.
**Why:** iter4 critique (two_digit_example_multiple_contrast, q=0.08) says incumbent over-credits multiples of large examples; times-table gating fixes that, round route keeps incumbent's gain.
**How to apply:** if admitted, the remaining untried stack piece is factor-kin route (tt+avg+fk was -2664.6 MAP in iter3).
