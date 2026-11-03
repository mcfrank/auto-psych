---
name: experiment2-iter4-critique-stats
description: exp2 iter_4 critique of mixed_strong_weak_sampling_concepts — 8 new PPC stats (order, repeat exposure, reversals, endpoints, local consistency) + observed values
metadata:
  type: project
---
iter_4 critique (2026-10-31), incumbent mixed_strong_weak_sampling_concepts (≈ beta_share; iter_3 got 0/8 significant, closest round_number_target_bonus p=0.08).
New stats, observed: digit_reversal_target_yes 0.30 (260 trials); endpoint_target_salience (1/100 minus 2-5/96-99) -0.17; magnitude_outlier_neighbourhood +0.25; mixed_parity_even_target_bias -0.20; monotone_order_set_yes_contrast +0.055; near_target_same_set_agreement 0.68 (385 pairs); repeat_exposure_drift (late minus first 5 encounters per participant-set) +0.03; sequence_continuation_asymmetry (forward minus backward) +0.10.
Participants can be delimited by trial_index==0 (10 resets); sets are mostly interleaved, not blocked (run length 1 for 60% rows).
**Why:** avoid re-probing; see [[experiment2-iter3-critique-stats]].
**How to apply:** read iter_4/critique/ppc_results.json for hits before choosing iter_5 stats.
