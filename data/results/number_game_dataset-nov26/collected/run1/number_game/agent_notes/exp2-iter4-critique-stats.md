---
name: exp2-iter4-critique-stats
description: exp2 iter4 critique of round_number_kinship_exemplar — 8 PPC stats: round-number scope, 5-endings, round-set strictness, gcd-3, far digit-run, parity slope, large multiples, target 100
metadata:
  type: project
---
Wrote 8 stats in experiment2/model_loop/iter_4/critique/test_stats (2026-11-02), observed values: round_target_in_nonround_even_set 0.091, five_ending_after_zero_examples 0.189, round_set_near_miss_rate 0.040, gcd3_divisibility_contrast 0.011, digit_run_far_match_gradient -0.077, parity_set_distance_slope -0.008, two_digit_example_multiple_contrast 0.061 (re-test of iter3's only hit), target_100_vs_other_round 0.059. All ~0.01-0.02 s.
Raw data: digit-reversal targets are rare/low (skip); "80,70,90,100" has 360 trials, ~0 yes off-rule.
**Why:** follow-up to [[exp2-iter3-critique-stats]]; incumbent is [[round-number-kinship-exemplar]].
**How to apply:** check ppc_results.json next round before re-proposing these ideas.
