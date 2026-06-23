# Holdout recovery — test-retest summary

- Source run root: `/scratch/users/benpry/auto-psych/impossible_holdout_test_retest_no_inner_loop`
- Recovery metric: `pearson_r`
- Repeats: 5 (run1, run2, run3, run4, run5)
- Ground-truth models: 4 (fewer_heads_more_random, longer_runs_more_random, more_heads_more_random, more_imbalance_more_random)

## Across-repeat reliability

- ICC(2,1): -0.014
- Mean pairwise correlation: -0.003

## Per ground-truth model (final-step pearson_r across repeats)

| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |
|---|---|---|---|---|---|---|---|---|
| fewer_heads_more_random | 5 | 0.558 | 0.431 | 0.772 | 0.079 | 1.000 | window_typicality | 0.20 |
| longer_runs_more_random | 5 | 0.592 | 0.137 | 0.232 | 0.485 | 0.789 | pure_alternation_rate | 0.20 |
| more_heads_more_random | 5 | 0.391 | 0.342 | 0.874 | 0.182 | 1.000 | bayesian_diagnosticity | 0.80 |
| more_imbalance_more_random | 5 | 0.795 | 0.459 | 0.578 | -0.026 | 1.000 | pure_imbalance | 0.40 |
