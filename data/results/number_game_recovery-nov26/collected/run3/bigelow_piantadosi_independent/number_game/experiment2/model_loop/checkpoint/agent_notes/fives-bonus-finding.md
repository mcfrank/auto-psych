---
name: fives-bonus-finding
description: exp2 multiples-of-5 prior bonus on prime_over_odd_reading +15.8 MAP loglik; mult3 cost +6.6, uncertainty-gated guessing +2
metadata:
  type: project
---
2026-10-30, exp2 iter_3/candidate_4 (`fives_readable_multiples`). MAP negative log-likelihoods on the incumbent prime_over_odd_reading (2126.6):
- bonus on multiples of 5 (rule row 18): 2110.8 (bonus≈1.0), the best result. Smoke loglik -2114.1, check OK, 142 s.
- extra cost on multiples of 3: 2120.0. Extra cost on every multiples rule except 5: 2124.2. Contains-d bonus: 2126.0 (null).
- uncertainty-gated guessing: 2124.1-2125.0. On the incumbent the lapse goes to 0 and beta≈0.8, which already softens predictions.
Biggest residual sets: 8,64 (multiples of 8 get fewer yes than predicted); 83,38,58,68 (85/87/89 get yes, so contains-8 is underpredicted); 90,100,5.

**Why:** times-table-specific priors still have signal. The incumbent's misindexed LOG_FACTOR gives mult3/4/5 zero factor cost ([[incumbent-log-factor-misindexed]]).
**How to apply:** scripts are in that candidate's scratch/ (screen.py, resid.py). A multiples-of-3 cost may stack on top. Related: [[prime-over-odd-swap]], [[odd-mult3-set-pattern]].
