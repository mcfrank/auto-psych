# Inner Model Loop Report

- Best model: **iter0_candidate1** (posterior=0.472, elpd_loo=-531.50)
- Trials: 900
- Models compared: 7

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate1 | 0.4715 | -531.50 |
| iter1_candidate2 | 0.4325 | -531.58 |
| iter1_candidate1 | 0.0948 | -533.10 |
| prototype_similarity | 0.0009 | -537.79 |
| iter0_candidate0 | 0.0003 | -538.71 |
| encoding_compressibility | 0.0000 | -606.79 |
| iter0_candidate2 | 0.0000 | -624.84 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate1 | 0.00 | 0.00 | — (best) | 0.553 |
| iter1_candidate2 | 0.09 | 0.43 | no (within ~2·dse) | 0.211 |
| iter1_candidate1 | 1.60 | 2.44 | no (within ~2·dse) | 0.236 |
| prototype_similarity | 6.30 | 3.02 | yes | 0.000 |
| iter0_candidate0 | 7.22 | 3.29 | yes | 0.000 |
| encoding_compressibility | 75.30 | 11.06 | yes | 0.000 |
| iter0_candidate2 | 93.34 | 12.31 | yes | 0.000 |
