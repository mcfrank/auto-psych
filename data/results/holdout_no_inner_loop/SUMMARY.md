# Holdout recovery — test-retest summary

- Source run root: `/scratch/users/benpry/auto-psych/holdout_test_retest_no_inner_loop_v2`
- Recovery metric: `pearson_r`
- Repeats: 5 (run1, run2, run3, run4, run5)
- Ground-truth models: 4 (bayesian_diagnosticity, encoding_compressibility, prototype_similarity, window_typicality)

## Across-repeat reliability

- ICC(2,1): 0.374
- Mean pairwise correlation: 0.154

## Per ground-truth model (final-step pearson_r across repeats)

| Ground truth | n | mean | sd | cv | min | max | modal best model | agreement |
|---|---|---|---|---|---|---|---|---|
| bayesian_diagnosticity | 5 | 0.873 | 0.042 | 0.048 | 0.804 | 0.906 | prototype_similarity | 1.00 |
| encoding_compressibility | 5 | 0.883 | 0.000 | 0.001 | 0.883 | 0.884 | prototype_similarity | 1.00 |
| prototype_similarity | 5 | 0.932 | 0.021 | 0.023 | 0.915 | 0.968 | bayesian_diagnosticity | 0.60 |
| window_typicality | 5 | 0.824 | 0.086 | 0.105 | 0.762 | 0.951 | length_sensitive_alternation | 0.60 |
