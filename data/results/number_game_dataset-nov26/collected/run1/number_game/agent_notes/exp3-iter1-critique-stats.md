---
name: exp3-iter1-critique-stats
description: exp3 iter1 critique of whole_set_round_category — 8 PPC stats: parity asymmetry & set-size near (carried over), interpolation, primacy, digit-rule flatness, exact examples, mult-of-5 in round sets, primes, cell polarization
metadata:
  type: project
---
Wrote 8 stats in experiment3/model_loop/iter_1/critique/test_stats (2026-11-02). Carried over from iter0 (significant there, not targeted by the incumbent): odd_vs_even_set_parity_violation, near_target_set_size_contrast. Dropped within_item_trial_drift on purpose: compute_features never sees trial_index, so candidates can't fix drift.
New stats, with observed values: interpolation_vs_extrapolation_contrast -0.004, first_vs_last_example_proximity 0.086, same_digit_rule_distance_flatness -0.088, exact_example_target_rate 0.817, all_round_set_multiple_of_five_contrast 0.585, prime_target_in_prime_set_contrast 0.032 (n=54/99, small), cell_polarization_fraction 0.373. Each takes about 0.05 s or less.
**Why:** follows [[exp3-iter0-critique-stats]]; incumbent is [[whole-set-round-category]].
**How to apply:** check iter_1 ppc_results.json before proposing these ideas again.
