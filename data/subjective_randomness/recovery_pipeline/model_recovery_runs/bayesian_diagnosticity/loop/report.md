# Inner Model Loop Report

- Best model: **prototype_similarity** (posterior=0.712, elpd_loo=-420.20)
- Trials: 800
- Models compared: 3

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| prototype_similarity | 0.7121 | -420.20 |
| bayesian_diagnosticity | 0.2879 | -421.10 |
| encoding_compressibility | 0.0000 | -457.88 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| prototype_similarity | 0.00 | 0.00 | — (best) | 0.925 |
| bayesian_diagnosticity | 0.91 | 1.42 | no (within ~2·dse) | 0.000 |
| encoding_compressibility | 37.68 | 9.11 | yes | 0.075 |
