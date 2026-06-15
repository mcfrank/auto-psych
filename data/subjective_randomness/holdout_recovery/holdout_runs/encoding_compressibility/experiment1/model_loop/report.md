# Inner Model Loop Report

- Best model: **iter1_candidate2** (posterior=0.548, elpd_loo=-407.78)
- Trials: 900
- Models compared: 6

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate2 | 0.5483 | -407.78 |
| iter1_candidate1 | 0.4517 | -407.97 |
| iter0_candidate0 | 0.0000 | -420.51 |
| bayesian_diagnosticity | 0.0000 | -420.83 |
| prototype_similarity | 0.0000 | -424.63 |
| iter1_candidate0 | 0.0000 | -4174.31 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate2 | 0.00 | 0.00 | — (best) | 0.974 |
| iter1_candidate1 | 0.19 | 0.30 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 12.73 | 5.05 | yes | 0.026 |
| bayesian_diagnosticity | 13.05 | 4.87 | yes | 0.000 |
| prototype_similarity | 16.85 | 5.62 | yes | 0.000 |
| iter1_candidate0 | 3766.53 | 327.21 | yes | 0.000 |
