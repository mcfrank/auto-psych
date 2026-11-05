---
name: exp2-iter2-critique-stats
description: exp2 iter2 critique of multiple_route_exemplar — 8 PPC stats on multiples-route scope, divisors, factor-kin, range/direction, log magnitude, item polarization
metadata:
  type: project
---
Wrote 8 stats in experiment2/model_loop/iter_2/critique/test_stats (2026-11-02): multiset_far_multiple_contrast, divisor_of_example_contrast, single_shared_factor_nonmultiple_contrast (gcd>=3 non-multiples after lone example), multiple_route_example_size_contrast (multiples of 2-digit vs 1-digit examples), interior_vs_exterior_contrast, above_vs_below_range_contrast, small_vs_large_magnitude_near_contrast, item_rate_sd.
Observed values: 0.168, -0.003, -0.137, -0.224, 0.024, 0.096, 0.014, 0.283. All run <0.03 s.
**Why:** follow-up to [[exp2-iter1-critique-stats]]; incumbent is [[multiple-route-exemplar]].
**How to apply:** check ppc_results.json next round; set-level breadth (background_set_rate_sd) was not re-tested here.
