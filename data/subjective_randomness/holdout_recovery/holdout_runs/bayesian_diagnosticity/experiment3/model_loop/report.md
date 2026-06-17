# Inner Model Loop Report

- Best model: **inner_loop_model** (posterior=0.379, elpd_loo=-1702.64)
- Trials: 2700
- Models compared: 12

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.3790 | -1702.64 |
| inferred_mixture_model | 0.3417 | -1702.75 |
| iter0_candidate0 | 0.2743 | -1702.96 |
| iter0_candidate2 | 0.0050 | -1706.97 |
| prototype_similarity | 0.0000 | -1718.86 |
| encoding_compressibility | 0.0000 | -1833.07 |
| unnormalized_mixture_model | 0.0000 | -13831.60 |
| feature_heuristic_model | 0.0000 | -1780.45 |
| fair_vs_alternating | 0.0000 | -1870.17 |
| iter0_candidate1 | 0.0000 | -1725.33 |
| iter1_candidate0 | 0.0000 | -1747.07 |
| iter1_candidate2 | 0.0000 | -1743.74 |

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 0.994 |
| inferred_mixture_model | 0.10 | 0.37 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 0.32 | 0.30 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 4.32 | 2.57 | no (within ~2·dse) | 0.000 |
| prototype_similarity | 16.22 | 5.27 | yes | 0.000 |
| iter0_candidate1 | 22.68 | 5.01 | yes | 0.000 |
| iter1_candidate2 | 41.09 | 8.63 | yes | 0.000 |
| iter1_candidate0 | 44.43 | 7.57 | yes | 0.000 |
| feature_heuristic_model | 77.81 | 12.22 | yes | 0.006 |
| encoding_compressibility | 130.43 | 15.27 | yes | 0.000 |
| fair_vs_alternating | 167.53 | 17.12 | yes | 0.000 |
| unnormalized_mixture_model | 12128.96 | 399.96 | yes | 0.000 |
