---
name: odd-mult3-set-pattern
description: exp2 odd multiple-of-3 sets get yes on odd non-multiples of 3, not on other multiples of 3
metadata:
  type: project
---
Raw data, exp2 (seen at iter_3 critique, 2026-10-30): after 69,9,39,21, targets 19/29/49 get 1.0 yes and multiples of 3 get about 0-0.4. After 63,27,99,33, odd primes 23/37/47/53 get 0.8-1.0 yes. Among odd multiple-of-3 sets, yes(odd non-mult3) minus yes(mult3) is +0.41. People seem to read these sets as "odd" or by final digit, not as "multiples of 3". The iter_3 critique statistic `odd_mult3_set_nonmult3_vs_mult3` tests this against the model.

**Why:** the prime_over_odd_reading incumbent penalises "odd", which may push it toward multiples of 3 on these sets.
**How to apply:** if this statistic comes back significant, consider weakening the odd penalty or adding a cost on multiples of 3. Related: [[prime-over-odd-swap]], [[final-digit-salience-finding]].
