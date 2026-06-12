# Inner Model Loop Report

- Best model: **prototype_similarity** (posterior=0.645, elpd_loo=-545.36)
- Trials: 800
- Models compared: 3

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| prototype_similarity | 0.6448 | -545.36 |
| bayesian_diagnosticity | 0.3551 | -545.95 |
| encoding_compressibility | 0.0001 | -554.39 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| prototype_similarity | 0.00 | 0.00 | — (best) | 0.866 |
| bayesian_diagnosticity | 0.60 | 1.27 | no (within ~2·dse) | 0.116 |
| encoding_compressibility | 9.03 | 4.33 | yes | 0.018 |
