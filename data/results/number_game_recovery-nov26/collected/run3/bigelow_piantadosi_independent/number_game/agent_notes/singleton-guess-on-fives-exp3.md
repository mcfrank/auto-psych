---
name: singleton-guess-on-fives-exp3
description: exp3 iter1 cand4 repair — singleton guessing on fives incumbent +2.5 MAP; fixed log_m index, parity bonus, div3, rule-only size exponent all ≤2
metadata:
  type: project
---
2026-10-31, exp3 iter_1/candidate_4_repair_1 (`singleton_guessing_privileged_fives`). The first attempt was set-size on fives, and it was rejected because it duplicated candidate_3 (RMSE 0). Candidate_5 made the same graft from the other direction (see [[prime-plus-marked-odd]]).
MAP screens on fives_privileged_decoupled_prime_odd (loglik -3436.7), screen scripts in /tmp/r4:
- singleton guess at even odds, g≈0.18: -3434.2. This one was submitted.
- rule-only size-principle exponent w≈0.96: -3434.7, RMSE 0.010 vs incumbent.
- log_m index fixed to rows 16:25: -3436.2. div3 cost: -3436.0. Parity bonus: null.
Worst sets: 3,33,99 (39; 93 gets 5/5 yes); 8,64; 83,38,58,68; and the odd sets 31,3,1,15 and 63,27,99,33, where odd is under-extended (0.6 predicted vs 0.8 observed).
**Why:** the exp3 incumbent is at its ceiling for single-parameter changes ([[exp3-iter1-exploratory-screens]]).
**How to apply:** before you write a graft, check the other iter candidates' model_name.txt.
