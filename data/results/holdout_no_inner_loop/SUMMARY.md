# Holdout recovery — test-retest summary

- Source run root: `/scratch/users/benpry/auto-psych/holdout_test_retest_no_inner_loop`
- Recovery metric: `pearson_r`
- Repeats: 5 (run1, run2, run3, run4, run5)
- Ground-truth models: 4 (bayesian_diagnosticity, encoding_compressibility, prototype_similarity, window_typicality)

## Across-repeat reliability

- ICC(2,1): 0.240
- Mean pairwise correlation: 0.172

## Per ground-truth model (final-step pearson_r across repeats)

| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |
|---|---|---|---|---|---|---|---|---|
| bayesian_diagnosticity | 5 | 0.840 | 0.012 | 0.014 | 0.826 | 0.856 | encoding_compressibility | 0.80 |
| encoding_compressibility | 5 | 0.901 | 0.021 | 0.024 | 0.863 | 0.912 | prototype_similarity | 0.60 |
| prototype_similarity | 5 | 0.926 | 0.063 | 0.068 | 0.880 | 0.995 | pure_imbalance | 0.20 |
| window_typicality | 5 | 0.851 | 0.094 | 0.110 | 0.779 | 0.956 | length_sensitive_alternation | 0.40 |
