# Holdout recovery — test-retest summary

- Source run root: `/scratch/users/benpry/auto-psych/holdout_test_retest_full`
- Recovery metric: `pearson_r`
- Repeats: 5 (run1, run2, run3, run4, run5)
- Ground-truth models: 4 (bayesian_diagnosticity, encoding_compressibility, prototype_similarity, window_typicality)

## Across-repeat reliability

- ICC(2,1): 0.581
- Mean pairwise correlation: 0.704

## Per ground-truth model (final-step pearson_r across repeats)

| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |
|---|---|---|---|---|---|---|---|---|
| bayesian_diagnosticity | 5 | 0.957 | 0.018 | 0.019 | 0.934 | 0.977 | iter1_candidate1 | 0.40 |
| encoding_compressibility | 5 | 0.937 | 0.040 | 0.043 | 0.879 | 0.986 | inner_loop_model | 0.40 |
| prototype_similarity | 5 | 0.991 | 0.009 | 0.009 | 0.977 | 0.999 | inner_loop_model | 0.20 |
| window_typicality | 5 | 0.997 | 0.005 | 0.005 | 0.991 | 1.000 | inner_loop_model | 0.60 |
