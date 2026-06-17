# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.394, elpd_loo=-629.24)
- Trials: 2400
- Models compared: 14

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.3942 | -629.24 |
| iter1_candidate0 | 0.2806 | -629.58 |
| imbalance_with_lapse | 0.1847 | -629.99 |
| iter1_candidate1 | 0.1404 | -630.22 |
| prototype_similarity | 0.0000 | -1408.92 |
| encoding_compressibility | 0.0000 | -1716.28 |
| bayesian_diagnosticity | 0.0000 | -746.07 |
| raw_alternation_count | 0.0000 | -1664.59 |
| high_alternation_rate | 0.0000 | -1664.57 |
| ideal_alternation_rate | 0.0000 | -815.47 |
| quadratic_imbalance | 0.0000 | -669.96 |
| iter0_candidate0 | 0.0000 | -732.32 |
| iter0_candidate2 | 0.0000 | -712.46 |
| iter1_candidate2 | 0.0000 | -704.98 |

## Hypotheses

- **inner_loop_model**: People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving sequences with a greater imbalance between the number of heads and tails as more random.
- **iter1_candidate0**: People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving greater proportion imbalance as more random, with the psychophysical scaling of this imbalance governed by a free power-law exponent.
- **imbalance_with_lapse**: People judge sequence randomness based purely on outcome frequencies, perceiving sequences with a greater proportion imbalance as more random, but their judgments are moderated by a constant lapse rate representing random guessing.
- **iter1_candidate1**: People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving sequences with a greater imbalance between the number of heads and tails as more random. However, their choices are also influenced by a constant baseline spatial or presentation order preference (a side bias) that shifts their probability of choosing the first sequence.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient non-random alternatives: alternating, biased, and streaky generators.
- **raw_alternation_count**: People perceive sequences with a higher raw count of alternations as more random, regardless of sequence length.
- **high_alternation_rate**: People judge sequence randomness based on the proportion of alternations, perceiving sequences with a higher alternation rate as strictly more random.
- **ideal_alternation_rate**: People judge sequence randomness by comparing the proportion of alternating outcomes to a subjective ideal alternation rate, perceiving sequences closer to this ideal as more random.
- **quadratic_imbalance**: People judge sequence randomness based on outcome frequencies, with perceived randomness growing quadratically as the proportion of heads and tails deviates further from a balanced distribution.
- **iter0_candidate0**: People judge sequence randomness based on outcome frequencies, perceiving sequences with a larger absolute difference between the raw count of heads and tails as more random.
- **iter0_candidate2**: People judge sequence randomness based purely on the relative length of the longest streak of identical outcomes, perceiving sequences with a larger normalized maximum run as less random.
- **iter1_candidate2**: People judge sequence randomness based purely on the Shannon entropy of the sequence's outcome frequencies, paradoxically perceiving sequences with lower entropy (a more biased distribution of heads and tails) as more random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 0.879 |
| iter1_candidate0 | 0.34 | 1.15 | no (within ~2·dse) | 0.000 |
| imbalance_with_lapse | 0.76 | 0.59 | no (within ~2·dse) | 0.000 |
| iter1_candidate1 | 0.98 | 0.14 | yes | 0.000 |
| quadratic_imbalance | 40.72 | 10.18 | yes | 0.103 |
| iter1_candidate2 | 75.75 | 13.37 | yes | 0.000 |
| iter0_candidate2 | 83.23 | 13.49 | yes | 0.009 |
| iter0_candidate0 | 103.08 | 12.64 | yes | 0.000 |
| bayesian_diagnosticity | 116.84 | 14.81 | yes | 0.000 |
| ideal_alternation_rate | 186.24 | 18.30 | yes | 0.010 |
| prototype_similarity | 779.68 | 28.72 | yes | 0.000 |
| high_alternation_rate | 1035.33 | 30.49 | yes | 0.000 |
| raw_alternation_count | 1035.35 | 30.49 | yes | 0.000 |
| encoding_compressibility | 1087.04 | 31.11 | yes | 0.000 |
