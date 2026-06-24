# Holdout recovery — test-retest summary

- Source run root: `/scratch/users/benpry/auto-psych/impossible_holdout_test_retest_v2`
- Recovery metric: `pearson_r`
- Repeats: 5 (run1, run2, run3, run4, run5)
- Ground-truth models: 4 (fewer_heads_more_random, longer_runs_more_random, more_heads_more_random, more_imbalance_more_random)

## Across-repeat reliability

- ICC(2,1): -0.128
- Mean pairwise correlation: -0.125

## Per ground-truth model (final-step pearson_r across repeats)

| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |
|---|---|---|---|---|---|---|---|---|
| fewer_heads_more_random | 5 | 1.000 | 0.000 | 0.000 | 0.999 | 1.000 | inner_loop_model | 0.60 |
| longer_runs_more_random | 5 | 1.000 | 0.000 | 0.000 | 0.999 | 1.000 | inner_loop_model | 0.60 |
| more_heads_more_random | 5 | 1.000 | 0.001 | 0.001 | 0.998 | 1.000 | iter1_candidate2 | 0.40 |
| more_imbalance_more_random | 5 | 1.000 | 0.001 | 0.001 | 0.998 | 1.000 | inner_loop_model | 0.40 |
