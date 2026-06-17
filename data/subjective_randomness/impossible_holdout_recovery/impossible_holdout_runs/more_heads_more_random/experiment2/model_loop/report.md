# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **squared_heads_heuristic** (posterior=0.387, elpd_loo=-28.28)
- Trials: 1500
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| squared_heads_heuristic | 0.3867 | -28.28 |
| iter1_candidate0 | 0.2959 | -28.44 |
| iter1_candidate2 | 0.1720 | -29.04 |
| inner_loop_model | 0.1404 | -29.29 |
| iter0_candidate2 | 0.0050 | -32.63 |
| length_scaled_head_difference | 0.0000 | -41.12 |
| prototype_similarity | 0.0000 | -1018.50 |
| encoding_compressibility | 0.0000 | -1069.79 |
| bayesian_diagnosticity | 0.0000 | -899.91 |
| iter0_candidate1 | 0.0000 | -129.07 |

## Hypotheses

- **squared_heads_heuristic**: People evaluate the randomness of a sequence strictly based on the squared number of heads it contains, amplifying the perception of randomness for sequences with very high head counts.
- **iter1_candidate0**: People evaluate the randomness of a sequence based solely on the number of heads it contains, but their perception of randomness scales as a power-law function of the head count rather than strictly linearly or quadratically. The model infers this exponent to capture exactly how marginal increases in head counts shape judgments.
- **iter1_candidate2**: People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains. However, their choices are subject to a constant lapse rate, meaning they occasionally guess randomly regardless of the head counts, rather than perfectly following a logistic choice curve.
- **inner_loop_model**: People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains.
- **iter0_candidate2**: People evaluate the randomness of a sequence strictly based on the cubed number of heads it contains. This mechanism creates an extreme, accelerating non-linear preference where sequences with high head counts are overwhelmingly perceived as more random.
- **length_scaled_head_difference**: People evaluate randomness primarily by the absolute number of heads, but their sensitivity to the difference in head counts between two sequences is diminished when the overall length of the sequences being compared is larger.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate1**: People evaluate the randomness of a sequence strictly based on the proportion of heads it contains, perceiving sequences with a higher proportion of heads as more random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| squared_heads_heuristic | 0.00 | 0.00 | — (best) | 0.802 |
| iter1_candidate0 | 0.17 | 0.03 | yes | 0.000 |
| iter1_candidate2 | 0.76 | 0.45 | no (within ~2·dse) | 0.000 |
| inner_loop_model | 1.01 | 2.60 | no (within ~2·dse) | 0.198 |
| iter0_candidate2 | 4.35 | 4.00 | no (within ~2·dse) | 0.000 |
| length_scaled_head_difference | 12.85 | 3.15 | yes | 0.000 |
| iter0_candidate1 | 100.79 | 16.73 | yes | 0.000 |
| bayesian_diagnosticity | 871.63 | 17.42 | yes | 0.000 |
| prototype_similarity | 990.23 | 9.59 | yes | 0.000 |
| encoding_compressibility | 1041.52 | 8.51 | yes | 0.000 |
