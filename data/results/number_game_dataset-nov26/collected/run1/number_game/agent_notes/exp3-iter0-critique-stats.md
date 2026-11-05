---
name: exp3-iter0-critique-stats
description: exp3 iter0 critique of round_number_averaging_exemplar — 8 PPC stats: set-level SD, large-example multiples, divisors, drift, odd/even parity, set-size near, rule gradient, mixed-set round targets
metadata:
  type: project
---
Wrote 8 stats in experiment3/model_loop/iter_0/critique/test_stats (2026-11-02). Observed values: background_set_rate_sd 0.150, large_example_multiple_contrast 0.240 (exp2 obs was 0.06; exp3 data differs), divisor_target_contrast 0.007, within_item_trial_drift 0.025, odd_vs_even_set_parity_violation -0.009, near_target_set_size_contrast 0.178, rule_consistent_distance_corr 0.001, mixed_set_round_target_contrast -0.047 (round targets rated BELOW background in sets like 15,11 / 6,25). All take ~0.05 s or less.
Exp3 raw data: 7560 trials, 144 sets, about 9 answers per (set,target), trial_index up to 449; 80,70,90,100 is ~0 off-rule; 15,11 → target 1 gets 0.78.
**Why:** first critique on the new exp3 (EIG-designed) data; follows [[exp2-iter4-critique-stats]]; incumbent is [[round-number-averaging-exemplar]].
**How to apply:** check ppc_results.json next round before re-proposing these ideas.
