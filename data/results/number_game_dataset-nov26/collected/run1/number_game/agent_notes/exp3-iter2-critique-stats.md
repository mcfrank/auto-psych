---
name: exp3-iter2-critique-stats
description: exp3 iter2 critique of coincidence_decisive_exemplar — 8 PPC stats: polarization split coherent/incoherent, set-size near & far, powers of 2, near off-digit in digit sets, above/below range, odd/even parity
metadata:
  type: project
---
Wrote 8 stats in experiment3/model_loop/iter_2/critique/test_stats (2026-11-02). Observed values: incoherent_set_cell_polarization 0.216 (604 coherent / 236 incoherent cells), coherent_set_cell_polarization 0.434, near_target_set_size_contrast 0.178 (carried over), far_background_set_size_contrast -0.102, power_of_two_target_contrast 0.132 (n 36 vs 117), same_digit_set_near_offdigit_rate 0.144, above_vs_below_range_near_contrast 0.107 (n 126/72), odd_vs_even_set_parity_violation -0.009 (carried over to check the parity_asym fix). Each runs in about 0.03 s.
Checked and dropped: difference-lattice (no effect once gcd and parity are controlled), decade match (0.298 vs 0.280), Weber single-example (no large single examples with near targets).
Heredoc gotcha: `read -r -d ''` strips the indent from the first line, so the first line of the function body needed fixing with sed.
**Why:** follows [[exp3-iter1-critique-stats]]; incumbent is [[coincidence-decisive-exemplar]].
**How to apply:** check iter_2 ppc_results.json before proposing these ideas again.
