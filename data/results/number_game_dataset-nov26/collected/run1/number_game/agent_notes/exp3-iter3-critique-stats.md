---
name: exp3-iter3-critique-stats
description: exp3 iter3 critique of parity_preserving_decisive_route — 8 PPC stats: polarization split single/incoherent-multi/coherent-by-n, coherent far violation floor, far-consistent set-size slope, parity outlier, odd-example multiple parity, first/last order
metadata:
  type: project
---
Wrote 8 stats in experiment3/model_loop/iter_3/critique/test_stats (2026-11-02). Observed values: single_example_cell_polarization 0.248 (113 cells), incoherent_multi_cell_polarization 0.187 (123 cells; this splits iter2's significant incoherent stat), coherent_polarization_set_size_contrast 0.025 (n4 vs n2; the model's coincidence grows with n-1), coherent_set_far_violation_rate 0.123 (1431 trials), coherent_far_consistent_set_size_slope 0.167, parity_outlier_neighborhood_contrast -0.061 (342 vs 72 trials), odd_example_multiple_parity_contrast 0.068 (tests the incumbent's parity-switch cut on the multiples route; 2x 0.51, other even multiples 0.51, odd multiples 0.58), first_vs_last_example_proximity -0.017 (a different definition from iter1's). Each runs in under 0.06 s.
Checked and dropped: squares (only 18 square-target trials) and Weber in single-example sets (no plain near targets).
**Why:** follows [[exp3-iter2-critique-stats]]; incumbent is [[parity-preserving-decisive-route]].
**How to apply:** check iter_3 ppc_results.json before proposing these ideas again.
