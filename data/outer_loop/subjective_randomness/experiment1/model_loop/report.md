# Inner Model Loop Report

- Best model: **iter0_candidate1** (posterior=0.277, elpd_loo=-60.87)
- Trials: 100
- Models compared: 9

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate1 | 0.2771 | -60.87 |
| iter1_candidate0 | 0.2224 | -61.09 |
| iter0_candidate2 | 0.1759 | -61.33 |
| alternation_bias | 0.1581 | -61.43 |
| iter1_candidate1 | 0.0978 | -61.91 |
| iter0_candidate0 | 0.0663 | -62.30 |
| iter1_candidate2 | 0.0019 | -65.85 |
| runs_penalty | 0.0005 | -67.24 |
| bayesian_fair_coin | 0.0001 | -68.95 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate1 | 0.00 | 0.00 | — (best) | 1.000 |
| iter1_candidate0 | 0.22 | 0.16 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 0.45 | 0.71 | no (within ~2·dse) | 0.000 |
| alternation_bias | 0.56 | 0.32 | no (within ~2·dse) | 0.000 |
| iter1_candidate1 | 1.04 | 1.14 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 1.43 | 0.69 | yes | 0.000 |
| iter1_candidate2 | 4.98 | 2.41 | yes | 0.000 |
| runs_penalty | 6.36 | 2.80 | yes | 0.000 |
| bayesian_fair_coin | 8.08 | 3.14 | yes | 0.000 |
