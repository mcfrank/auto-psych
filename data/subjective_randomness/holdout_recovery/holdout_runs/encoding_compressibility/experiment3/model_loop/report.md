# Inner Model Loop Report

- Best model: **inner_loop_model** (posterior=0.283, elpd_loo=-816.27)
- Trials: 2400
- Models compared: 13

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.2826 | -816.27 |
| length_sensitive_compressibility | 0.1536 | -816.88 |
| iter0_candidate2 | 0.1274 | -817.07 |
| student_extended_compressibility | 0.1154 | -817.17 |
| iter1_candidate0 | 0.1154 | -817.17 |
| iter0_candidate0 | 0.1081 | -817.23 |
| extended_compressibility | 0.0872 | -817.45 |
| iter1_candidate1 | 0.0103 | -819.59 |
| prototype_similarity | 0.0000 | -899.04 |
| bayesian_diagnosticity | 0.0000 | -887.33 |
| alternation_and_run | 0.0000 | -863.41 |
| iter0_candidate1 | 0.0000 | -841.36 |
| iter1_candidate2 | 0.0000 | -876.79 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 0.694 |
| length_sensitive_compressibility | 0.61 | 1.00 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 0.80 | 0.60 | no (within ~2·dse) | 0.000 |
| student_extended_compressibility | 0.90 | 0.09 | yes | 0.000 |
| iter1_candidate0 | 0.90 | 0.09 | yes | 0.000 |
| iter0_candidate0 | 0.96 | 0.70 | no (within ~2·dse) | 0.000 |
| extended_compressibility | 1.18 | 0.19 | yes | 0.000 |
| iter1_candidate1 | 3.31 | 4.17 | no (within ~2·dse) | 0.306 |
| iter0_candidate1 | 25.09 | 7.61 | yes | 0.000 |
| alternation_and_run | 47.14 | 9.72 | yes | 0.000 |
| iter1_candidate2 | 60.52 | 11.22 | yes | 0.001 |
| bayesian_diagnosticity | 71.05 | 12.34 | yes | 0.000 |
| prototype_similarity | 82.77 | 13.38 | yes | 0.000 |
