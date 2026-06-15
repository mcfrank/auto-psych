# Inner Model Loop Report

- Best model: **iter0_candidate2** (posterior=0.224, elpd_loo=-680.91)
- Trials: 1800
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate2 | 0.2242 | -680.91 |
| iter0_candidate0 | 0.2201 | -680.93 |
| iter0_candidate1 | 0.2101 | -680.98 |
| inner_loop_model | 0.1697 | -681.19 |
| iter1_candidate2 | 0.1016 | -681.70 |
| extended_compressibility | 0.0743 | -682.02 |
| prototype_similarity | 0.0000 | -760.09 |
| bayesian_diagnosticity | 0.0000 | -749.50 |
| alternation_and_run | 0.0000 | -720.70 |
| iter1_candidate1 | 0.0000 | -6150.13 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate2 | 0.00 | 0.00 | — (best) | 1.000 |
| iter0_candidate0 | 0.02 | 0.15 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 0.07 | 0.32 | no (within ~2·dse) | 0.000 |
| inner_loop_model | 0.28 | 0.06 | yes | 0.000 |
| iter1_candidate2 | 0.79 | 0.42 | no (within ~2·dse) | 0.000 |
| extended_compressibility | 1.10 | 0.17 | yes | 0.000 |
| alternation_and_run | 39.78 | 8.88 | yes | 0.000 |
| bayesian_diagnosticity | 68.59 | 11.74 | yes | 0.000 |
| prototype_similarity | 79.18 | 12.68 | yes | 0.000 |
| iter1_candidate1 | 5469.22 | 356.15 | yes | 0.000 |
