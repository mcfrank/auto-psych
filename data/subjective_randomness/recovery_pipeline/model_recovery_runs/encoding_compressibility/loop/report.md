# Inner Model Loop Report

- Best model: **encoding_compressibility** (posterior=1.000, elpd_loo=-530.07)
- Trials: 800
- Models compared: 3

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| encoding_compressibility | 1.0000 | -530.07 |
| bayesian_diagnosticity | 0.0000 | -541.76 |
| prototype_similarity | 0.0000 | -545.58 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| encoding_compressibility | 0.00 | 0.00 | — (best) | 1.000 |
| bayesian_diagnosticity | 11.69 | 4.78 | yes | 0.000 |
| prototype_similarity | 15.51 | 5.30 | yes | 0.000 |
