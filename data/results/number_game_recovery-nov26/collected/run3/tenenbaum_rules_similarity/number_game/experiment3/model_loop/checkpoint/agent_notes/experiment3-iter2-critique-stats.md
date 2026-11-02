---
name: experiment3-iter2-critique-stats
description: exp3 iter_2 critique of kind_uncertainty_shifted_criterion — 8 new PPC stats + observed values (iter_1 got 0/8)
metadata:
  type: project
---
iter_2 critique (2026-10-31), incumbent still kind_uncertainty_shifted_criterion; iter_1 had 0/8 hits (closest decade_shared_contrast p=0.08, target_100_yes p=0.10).
New stats, observed: near_outspan_generalization (1-5 outside span, n>=2) 0.380; units_digit_shared_contrast -0.019; powers2_set_even_nonpower_yes 0.315; prime_set_prime_vs_odd_composite 0.195; set_yes_rate_sd (sets >=20 trials) 0.242; small_vs_large_target_contrast (<=10 vs >=91) 0.032; round_target_yes_nonround_sets -0.045; wide_sparse_interior_yes (n>=3, span>=30) 0.481.
Committed from reasoning only, with no null simulation run.
**Why:** so later rounds don't probe these again; see [[experiment3-iter1-critique-stats]].
**How to apply:** read iter_2/critique/ppc_results.json for hits before choosing iter_3 stats.
