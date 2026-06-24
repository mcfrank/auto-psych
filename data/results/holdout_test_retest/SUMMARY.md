# Holdout recovery — test-retest summary

- Source run root: `/scratch/users/benpry/auto-psych/holdout_test_retest_v2`
- Recovery metric: `pearson_r`
- Repeats: 5 (run1, run2, run3, run4, run5)
- Ground-truth models: 4 (bayesian_diagnosticity, encoding_compressibility, prototype_similarity, window_typicality)

## Across-repeat reliability

- ICC(2,1): 0.153
- Mean pairwise correlation: 0.270

## Per ground-truth model (final-step pearson_r across repeats)

| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |
|---|---|---|---|---|---|---|---|---|
| bayesian_diagnosticity | 5 | 0.974 | 0.020 | 0.021 | 0.939 | 0.992 | iter0_candidate1 | 0.40 |
| encoding_compressibility | 5 | 0.961 | 0.055 | 0.057 | 0.874 | 0.999 | iter0_candidate1 | 0.40 |
| prototype_similarity | 5 | 0.998 | 0.002 | 0.002 | 0.994 | 0.999 | inner_loop_model | 0.80 |
| window_typicality | 5 | 0.999 | 0.001 | 0.001 | 0.998 | 1.000 | inner_loop_model | 0.80 |
