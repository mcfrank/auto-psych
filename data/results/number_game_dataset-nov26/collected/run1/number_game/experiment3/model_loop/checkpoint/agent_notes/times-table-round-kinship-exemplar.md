---
name: times-table-round-kinship-exemplar
description: exp2 iter4 c4: times-table gate (examples 2-12) on round_number_kinship multiples route, MLE +4.6 nats; smoke -2671.1
metadata:
  type: project
---
exp2 iter4 candidate_4 (2026-11-02) `times_table_round_kinship_exemplar`: incumbent [[round-number-kinship-exemplar]] with multiples route only for examples 2-12. Targets critique stat two_digit_example_multiple_contrast (see [[exp2-iter4-critique-stats]]).
MLE screen (numpy): base 2670.2; gate12 = gate10 = gate9 2665.6 (soft >12 weight goes to 0); linear decay with e 2668.8; log-e decay 2670.1; decay with multiplier 2668.5; also gating factor-kin to e<=12 hurts (2672.0); factor-kin gated by shared gcd<=12 2666.3.
**Why:** the times-table gate is still worth ~5 nats on top of round kinship; candidate_3 was screening tt/avg variants and candidate_5 did tt+avg+round, so duplicates are possible.
**How to apply:** the times-table-gate idea is used up; don't screen more multiples-route variants.
