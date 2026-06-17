# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate1** (posterior=0.717, elpd_loo=-47.76)
- Trials: 2400
- Models compared: 12

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate1 | 0.7168 | -47.76 |
| absolute_heads_lapse | 0.1940 | -48.92 |
| iter0_candidate0 | 0.0847 | -49.74 |
| squared_heads_heuristic | 0.0022 | -53.48 |
| inner_loop_model | 0.0022 | -53.48 |
| length_scaled_head_difference | 0.0000 | -59.04 |
| prototype_similarity | 0.0000 | -1586.78 |
| encoding_compressibility | 0.0000 | -1716.39 |
| bayesian_diagnosticity | 0.0000 | -1296.25 |
| power_law_heads | 0.0000 | -163912.14 |
| iter0_candidate1 | 0.0000 | -142.39 |
| iter1_candidate2 | 0.0000 | -71.34 |

## Hypotheses

- **iter1_candidate1**: People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains. Their choice between sequences is driven by a cumulative normal (probit) discrimination process on the difference in head counts, reflecting Gaussian internal noise with no baseline rate of random guessing.
- **absolute_heads_lapse**: People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains, but their choices are subject to a constant lapse rate representing random guessing.
- **iter0_candidate0**: People evaluate the randomness of a sequence based on the logarithm of the absolute number of heads it contains, reflecting a diminishing sensitivity to head count differences as the total number of heads increases, and their choices are subject to a constant lapse rate for random guessing.
- **squared_heads_heuristic**: People evaluate the randomness of a sequence strictly based on the squared number of heads it contains, amplifying the perception of randomness for sequences with very high head counts.
- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **length_scaled_head_difference**: People evaluate randomness primarily by the absolute number of heads, but their sensitivity to the difference in head counts between two sequences is diminished when the overall length of the sequences being compared is larger.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **power_law_heads**: People evaluate the randomness of a sequence based on the number of heads it contains, but their perception of randomness scales as an inferred power-law function of the head count.
- **iter0_candidate1**: People evaluate the randomness of a sequence primarily based on the proportion of heads it contains relative to its total length, judging sequences with a higher fraction of heads to be more random, and their choices are subject to a constant lapse rate representing occasional random guessing.
- **iter1_candidate2**: People evaluate the randomness of a sequence based on the square root of the number of heads it contains, reflecting a diminishing sensitivity to each additional head, and their choices are subject to a constant lapse rate representing occasional random guessing.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate1 | 0.00 | 0.00 | — (best) | 0.880 |
| absolute_heads_lapse | 1.16 | 0.13 | yes | 0.000 |
| iter0_candidate0 | 1.99 | 1.99 | no (within ~2·dse) | 0.120 |
| squared_heads_heuristic | 5.72 | 3.58 | no (within ~2·dse) | 0.000 |
| inner_loop_model | 5.72 | 3.58 | no (within ~2·dse) | 0.000 |
| length_scaled_head_difference | 11.28 | 3.32 | yes | 0.000 |
| iter1_candidate2 | 23.59 | 4.69 | yes | 0.000 |
| iter0_candidate1 | 94.64 | 17.30 | yes | 0.000 |
| bayesian_diagnosticity | 1248.49 | 26.64 | yes | 0.000 |
| prototype_similarity | 1539.02 | 15.40 | yes | 0.000 |
| encoding_compressibility | 1668.63 | 11.49 | yes | 0.000 |
| power_law_heads | 163864.38 | 116489.57 | no (within ~2·dse) | 0.000 |
