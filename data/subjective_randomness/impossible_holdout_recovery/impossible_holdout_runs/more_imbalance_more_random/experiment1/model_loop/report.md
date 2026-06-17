# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=1.000, elpd_loo=-287.89)
- Trials: 900
- Models compared: 9

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 1.0000 | -287.89 |
| prototype_similarity | 0.0000 | -473.21 |
| encoding_compressibility | 0.0000 | -639.57 |
| bayesian_diagnosticity | 0.0000 | -340.71 |
| iter0_candidate0 | 0.0000 | -339.65 |
| iter0_candidate1 | 0.0000 | -340.77 |
| iter0_candidate2 | 0.0000 | -624.83 |
| iter1_candidate1 | 0.0000 | -320.14 |
| iter1_candidate2 | 0.0000 | -624.82 |

## Hypotheses

- **iter1_candidate0**: People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving sequences with a greater imbalance between the number of heads and tails as more random.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate0**: People judge sequence randomness based on the Bayesian diagnosticity of a fair coin compared to a single salient alternative: a "streaky" Markov generator that has an innate tendency to repeat previous outcomes. Sequences that are more likely under the fair coin than the streaky alternative are perceived as more random.
- **iter0_candidate1**: People judge sequence randomness based purely on the alternation rate heuristic: they compare the proportion of alternating outcomes in the sequence to a subjective ideal alternation rate. Sequences whose alternation rate is closer to this subjective ideal are perceived as more random.
- **iter0_candidate2**: People judge sequence randomness purely based on the relative frequencies of outcomes: sequences with a smaller imbalance between the number of heads and tails are perceived as more random.
- **iter1_candidate1**: People judge sequence randomness based on the Bayesian diagnosticity of a biased coin relative to a fair coin. Sequences that provide stronger evidence for a biased coin (which can favor either heads or tails) over a fair coin are perceived as more random.
- **iter1_candidate2**: People judge sequence randomness based purely on the maximum run length heuristic. They assess the length of the longest continuous streak of identical outcomes in each sequence, perceiving the sequence with the shorter maximum streak as more random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate0 | 0.00 | 0.00 | — (best) | 0.995 |
| iter1_candidate1 | 32.25 | 10.32 | yes | 0.000 |
| iter0_candidate0 | 51.76 | 8.84 | yes | 0.000 |
| bayesian_diagnosticity | 52.82 | 9.45 | yes | 0.000 |
| iter0_candidate1 | 52.88 | 10.00 | yes | 0.000 |
| prototype_similarity | 185.32 | 14.97 | yes | 0.000 |
| iter1_candidate2 | 336.93 | 18.50 | yes | 0.000 |
| iter0_candidate2 | 336.94 | 18.50 | yes | 0.000 |
| encoding_compressibility | 351.68 | 18.86 | yes | 0.005 |
