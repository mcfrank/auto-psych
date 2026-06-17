# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.684, elpd_loo=-517.16)
- Trials: 1800
- Models compared: 12

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.6843 | -517.16 |
| iter1_candidate2 | 0.3157 | -517.83 |
| prototype_similarity | 0.0000 | -917.46 |
| encoding_compressibility | 0.0000 | -1273.89 |
| bayesian_diagnosticity | 0.0000 | -594.69 |
| raw_alternation_count | 0.0000 | -1248.66 |
| high_alternation_rate | 0.0000 | -1248.67 |
| ideal_alternation_rate | 0.0000 | -630.83 |
| iter0_candidate0 | 0.0000 | -553.45 |
| iter0_candidate1 | 0.0000 | -595.26 |
| iter0_candidate2 | 0.0000 | -1248.64 |
| iter1_candidate1 | 0.0000 | -607.17 |

## Hypotheses

- **inner_loop_model**: People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving sequences with a greater imbalance between the number of heads and tails as more random.
- **iter1_candidate2**: People judge sequence randomness based on outcome frequencies, paradoxically perceiving sequences with a greater proportion imbalance as more random, but their judgments are moderated by a constant lapse rate representing random guessing.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **raw_alternation_count**: People perceive sequences with a higher raw count of alternations as more random, regardless of sequence length.
- **high_alternation_rate**: People judge sequence randomness based on the proportion of alternations, perceiving sequences with a higher alternation rate as strictly more random.
- **ideal_alternation_rate**: People judge sequence randomness by comparing the proportion of alternating outcomes to a subjective ideal alternation rate, perceiving sequences closer to this ideal as more random.
- **iter0_candidate0**: People judge sequence randomness based on outcome frequencies, with perceived randomness growing quadratically as the proportion of heads and tails deviates further from a balanced distribution.
- **iter0_candidate1**: People judge sequence randomness based on the statistical evidence for outcome bias, perceiving sequences as more random when the log-likelihood ratio of a biased coin model versus a fair coin model is higher.
- **iter0_candidate2**: People judge sequence randomness by looking for the longest streak of identical outcomes, perceiving sequences where the longest streak takes up a smaller proportion of the sequence length as more random.
- **iter1_candidate1**: People judge sequence randomness by evaluating the raw numerical difference between the counts of the two outcomes, perceiving sequences with a larger absolute difference between the number of heads and tails as more random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 0.875 |
| iter1_candidate2 | 0.67 | 0.59 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 36.29 | 9.53 | yes | 0.091 |
| bayesian_diagnosticity | 77.53 | 12.80 | yes | 0.000 |
| iter0_candidate1 | 78.10 | 15.08 | yes | 0.000 |
| iter1_candidate1 | 90.01 | 11.32 | yes | 0.000 |
| ideal_alternation_rate | 113.66 | 15.05 | yes | 0.034 |
| prototype_similarity | 400.30 | 22.51 | yes | 0.000 |
| iter0_candidate2 | 731.48 | 26.38 | yes | 0.000 |
| raw_alternation_count | 731.50 | 26.38 | yes | 0.000 |
| high_alternation_rate | 731.51 | 26.38 | yes | 0.000 |
| encoding_compressibility | 756.73 | 26.75 | yes | 0.000 |
