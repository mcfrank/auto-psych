# Inner Model Loop Report

- Best model: **inner_loop_model** (posterior=0.440, elpd_loo=-1127.57)
- Trials: 1800
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.4401 | -1127.57 |
| iter1_candidate0 | 0.2253 | -1128.24 |
| iter0_candidate0 | 0.1678 | -1128.53 |
| inferred_mixture_model | 0.1666 | -1128.54 |
| iter0_candidate1 | 0.0001 | -1135.76 |
| prototype_similarity | 0.0000 | -1136.90 |
| iter0_candidate2 | 0.0000 | -1138.69 |
| encoding_compressibility | 0.0000 | -1228.07 |
| unnormalized_mixture_model | 0.0000 | -8065.33 |
| iter1_candidate1 | 0.0000 | -1168.35 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 1.000 |
| iter1_candidate0 | 0.67 | 0.51 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 0.96 | 0.54 | no (within ~2·dse) | 0.000 |
| inferred_mixture_model | 0.97 | 0.37 | yes | 0.000 |
| iter0_candidate1 | 8.20 | 3.52 | yes | 0.000 |
| prototype_similarity | 9.33 | 4.19 | yes | 0.000 |
| iter0_candidate2 | 11.12 | 4.46 | yes | 0.000 |
| iter1_candidate1 | 40.78 | 6.97 | yes | 0.000 |
| encoding_compressibility | 100.50 | 13.31 | yes | 0.000 |
| unnormalized_mixture_model | 6937.76 | 280.95 | yes | 0.000 |
