# Inner Model Loop Report

- Best model: **iter0_candidate1** (posterior=0.194, elpd_loo=-166.77)
- Trials: 600
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate1 | 0.1935 | -166.77 |
| iter1_candidate0 | 0.1645 | -166.93 |
| iter1_candidate2 | 0.1506 | -167.02 |
| encoding_compressibility | 0.1137 | -167.30 |
| iter0_candidate0 | 0.1086 | -167.35 |
| bayesian_diagnosticity | 0.0981 | -167.45 |
| iter1_candidate1 | 0.0864 | -167.57 |
| iter0_candidate2 | 0.0845 | -167.60 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate1 | 0.00 | 0.00 | — (best) | 0.926 |
| iter1_candidate0 | 0.16 | 0.47 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 0.25 | 0.70 | no (within ~2·dse) | 0.000 |
| encoding_compressibility | 0.53 | 0.70 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 0.58 | 0.52 | no (within ~2·dse) | 0.000 |
| bayesian_diagnosticity | 0.68 | 1.26 | no (within ~2·dse) | 0.000 |
| iter1_candidate1 | 0.81 | 1.30 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 0.83 | 1.37 | no (within ~2·dse) | 0.074 |
