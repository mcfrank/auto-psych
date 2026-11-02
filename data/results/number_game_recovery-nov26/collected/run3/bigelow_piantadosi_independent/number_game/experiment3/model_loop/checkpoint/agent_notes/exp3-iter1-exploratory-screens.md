---
name: exp3-iter1-exploratory-screens
description: exp3 iter1 cand1 MAP screens on the fives incumbent (loglik -3436.7); weak-sampling ranges, rote tables, production readout, noise all null or worse; hard 7/8 tables +2.2
metadata:
  type: project
---
2026-10-31, exp3 iter_1/candidate_1 (`hard_seven_eight_tables`). Base: fives_privileged_decoupled_prime_odd, MAP loglik -3436.7.
- Range sampling exponent (0 = weak, 1 = strong): fits 0.84, +0.4. Null.
- Rote times tables (m×1..10 or ×12): -340 to -450. People extend 4s, 6s and 3s beyond 10m.
- Posterior-predictive "production" readout (size-weighted, then a logistic): -124.
- Fixed NOISE 0.03 or 0.2: worse by 7 and 15. 0.1 is about optimal.
- Extra cost on multiples of 7 and 8 (rows 20, 21): +2.2, cost 0.65, RMSE vs base 0.012. This one was submitted; the check printed OK, smoke loglik -3442.0, 116 s for the smoke chain.
The worst-fit sets (excess nll over a saturated model): 3,33,99 (39); 8,64 (29: multiples of 8 predicted 0.7, observed 0.3); 83,38,58,68 (26: 85/87/89 observed 0.9, predicted 0.4); 14,98,91 (15: multiples of 7 predicted 0.7, observed 0.2). Aggregate calibration is fine.
Screen script: iter_1/candidate_1/scratch/rote2.py <mode> <K> <model.py>. Get p_yes with m.replace_rvs_by_values; m.compile_fn evaluates at random RV draws.

**Why:** the response stage, the sampling assumption and the noise rate are all exhausted on exp3.
**How to apply:** the remaining residuals are idiosyncratic per set. Related: [[concept-space-saturated-exp3]], [[exp3-decision-stage-screens]].
