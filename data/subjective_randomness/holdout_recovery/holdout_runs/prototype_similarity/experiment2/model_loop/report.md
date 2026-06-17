# Inner Model Loop Report

- Best model: **iter1_candidate1** (posterior=0.161, elpd_loo=-326.00)
- Trials: 1200
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate1 | 0.1611 | -326.00 |
| inner_loop_model | 0.1554 | -326.04 |
| asymmetric_alternation_prototype | 0.1542 | -326.05 |
| iter1_candidate2 | 0.1442 | -326.11 |
| iter0_candidate1 | 0.1001 | -326.48 |
| iter0_candidate0 | 0.0946 | -326.54 |
| iter0_candidate2 | 0.0634 | -326.94 |
| encoding_compressibility | 0.0557 | -327.07 |
| length_sensitive_prototype | 0.0490 | -327.19 |
| bayesian_diagnosticity | 0.0223 | -327.98 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate1 | 0.00 | 0.00 | — (best) | 0.542 |
| inner_loop_model | 0.04 | 0.68 | no (within ~2·dse) | 0.000 |
| asymmetric_alternation_prototype | 0.04 | 0.66 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 0.11 | 0.55 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 0.48 | 1.43 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 0.53 | 1.31 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 0.93 | 1.51 | no (within ~2·dse) | 0.000 |
| encoding_compressibility | 1.06 | 1.25 | no (within ~2·dse) | 0.000 |
| length_sensitive_prototype | 1.19 | 3.03 | no (within ~2·dse) | 0.319 |
| bayesian_diagnosticity | 1.98 | 2.86 | no (within ~2·dse) | 0.139 |
