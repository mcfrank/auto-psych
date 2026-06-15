# Inner Model Loop Report

- Best model: **iter0_candidate0** (posterior=0.183, elpd_loo=-484.21)
- Trials: 1800
- Models compared: 13

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate0 | 0.1831 | -484.21 |
| asymmetric_alternation_prototype | 0.1512 | -484.40 |
| periodicity_salience | 0.1461 | -484.44 |
| iter1_candidate1 | 0.1146 | -484.68 |
| iter1_candidate0 | 0.1107 | -484.72 |
| length_sensitive_prototype | 0.1014 | -484.80 |
| encoding_compressibility | 0.0936 | -484.88 |
| inner_loop_model | 0.0319 | -485.96 |
| iter0_candidate1 | 0.0278 | -486.10 |
| runs_test_model | 0.0232 | -486.28 |
| bayesian_diagnosticity | 0.0077 | -487.39 |
| iter1_candidate2 | 0.0054 | -487.74 |
| iter0_candidate2 | 0.0035 | -488.17 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate0 | 0.00 | 0.00 | — (best) | 0.000 |
| asymmetric_alternation_prototype | 0.19 | 1.04 | no (within ~2·dse) | 0.000 |
| periodicity_salience | 0.23 | 0.85 | no (within ~2·dse) | 0.000 |
| iter1_candidate1 | 0.47 | 1.13 | no (within ~2·dse) | 0.388 |
| iter1_candidate0 | 0.50 | 0.53 | no (within ~2·dse) | 0.000 |
| length_sensitive_prototype | 0.59 | 3.03 | no (within ~2·dse) | 0.387 |
| encoding_compressibility | 0.67 | 0.64 | no (within ~2·dse) | 0.000 |
| inner_loop_model | 1.75 | 2.30 | no (within ~2·dse) | 0.074 |
| iter0_candidate1 | 1.89 | 1.82 | no (within ~2·dse) | 0.000 |
| runs_test_model | 2.07 | 2.37 | no (within ~2·dse) | 0.000 |
| bayesian_diagnosticity | 3.17 | 2.96 | no (within ~2·dse) | 0.151 |
| iter1_candidate2 | 3.53 | 2.62 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 3.96 | 2.63 | no (within ~2·dse) | 0.000 |
