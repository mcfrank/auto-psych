---
name: experiment3-iter4-critique-stats
description: exp3 iter_4 critique of rule_family_doubt_criterion — 8 new PPC stats, observed values, rough z from the iter_1 refit (iters 0-3 got 0/32 hits)
metadata:
  type: project
---
iter_4 critique (2026-10-31), incumbent rule_family_doubt_criterion. Iters 0-3: 0/32 significant.
Rough z values come from iter_1/critique/scratch/crit3/pred.csv, the older kind_uncertainty MAP, using 200 binomial replicates; copy of sim.py at iter_4/critique/scratch/sim.py. Stats, observed (z):
late_vs_early_discrimination 0.035 (z≈+3.1: answers agree more with cell consensus late in session); mersenne_fermat_target_contrast 0.028 (z≈+1.85, 2^k±1 sets); majority_rule_exception_contrast -0.18 (z≈-1.5); residual_lag1_autocorrelation 0.012 (z≈+1.1); span_scaled_outside_generalization -0.15 (noisy); offset_progression_contrast -0.04; participant_drift_sd 0.065; odd_rule_conjunction_contrast 0.03 (z≈0).
**Why:** avoid re-probing; see [[experiment3-iter3-critique-stats]].
**How to apply:** if late_vs_early_discrimination hits, the next fix is a session-time sharpening/noise parameter (trial_index is forbidden at model admission, see [[sample-size-shifted-criterion]]).
