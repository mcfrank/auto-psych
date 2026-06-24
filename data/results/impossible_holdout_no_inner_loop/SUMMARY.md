# Holdout recovery — test-retest summary

- Source run root: `/scratch/users/benpry/auto-psych/impossible_holdout_test_retest_no_inner_loop_v2`
- Recovery metric: `pearson_r`
- Repeats: 5 (run1, run2, run3, run4, run5)
- Ground-truth models: 4 (fewer_heads_more_random, longer_runs_more_random, more_heads_more_random, more_imbalance_more_random)

## Across-repeat reliability

- ICC(2,1): 0.632
- Mean pairwise correlation: 0.604

## Per ground-truth model (final-step pearson_r across repeats)

| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |
|---|---|---|---|---|---|---|---|---|
| fewer_heads_more_random | 5 | 0.426 | 0.463 | 1.087 | 0.078 | 1.000 | length_sensitive_alternation | 0.20 |
| longer_runs_more_random | 5 | 0.568 | 0.132 | 0.232 | 0.498 | 0.804 | prototype_similarity | 0.20 |
| more_heads_more_random | 5 | 0.845 | 0.344 | 0.407 | 0.230 | 1.000 | head_count_bias | 0.40 |
| more_imbalance_more_random | 5 | -0.153 | 0.104 | -0.680 | -0.333 | -0.076 | length_sensitive_alternation | 0.60 |
