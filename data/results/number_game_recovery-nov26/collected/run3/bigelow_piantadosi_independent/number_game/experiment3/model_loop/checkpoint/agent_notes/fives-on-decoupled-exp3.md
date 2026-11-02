---
name: fives-on-decoupled-exp3
description: exp3 cand5 repair — fives bonus on prime_bonus_odd_cost_decoupled +18 MAP; set-size +12, div3 +3 on same base
metadata:
  type: project
---
2026-10-31, exp3 iter_0/candidate_5_repair_1 (`fives_privileged_decoupled_prime_odd`). The first attempt (repdigit graft on prime_salient_marked_odd) was rejected because it was the same model as prime_bonus_odd_cost_decoupled (RMSE 0). Grafting A onto B gives the same model as grafting B onto A ([[prime-plus-marked-odd]]).
MAP nll on the decoupled base is about 3454. Each one-parameter addition was screened on it:
- fives bonus on row 18: 3435.6 (bonus 2.6, pool RMSE 0.034)
- set-size (n/3)^g: 3442.0
- div3 cost: 3451.5
Screen script: candidate_5_repair_1/scratch/s2.py. The check printed OK, with smoke loglik -3443.2 and about 90 s per smoke chain; under load avg 45 the check overran the 10-min background limit once.

**Why:** the fives gain replicates exp2's ([[fives-bonus-finding]]) on the exp3 base.
**How to apply:** check whether a live model already has the same rule set before you graft.
