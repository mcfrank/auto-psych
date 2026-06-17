# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter0_candidate2** (posterior=1.000, elpd_loo=-29.03)
- Trials: 900
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate2 | 1.0000 | -29.03 |
| prototype_similarity | 0.0000 | -619.22 |
| encoding_compressibility | 0.0000 | -628.79 |
| bayesian_diagnosticity | 0.0000 | -608.24 |
| iter0_candidate0 | 0.0000 | -120.42 |
| iter0_candidate1 | 0.0000 | -119.99 |
| iter1_candidate0 | 0.0000 | -58.03 |
| iter1_candidate2 | 0.0000 | -129.11 |

## Hypotheses

- **iter0_candidate2**: People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains. They apply a simple additive heuristic where each additional head linearly increases the perceived randomness score, disregarding the sequence length and outcome order. When comparing two sequences, they are more likely to choose the sequence with the higher total head count as the more random one.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate0**: People judge the randomness of a sequence by its similarity to a prototype, but rather than expecting balanced outcomes, their prototype expects a sequence to be dominated by heads. Sequences with a higher proportion of heads are perceived as closer to the prototype and therefore more random.
- **iter0_candidate1**: People rely on a simple "heads-equal-randomness" heuristic, where heads are viewed as independent random events and tails are viewed as non-random deterministic events. Consequently, the perceived randomness of a sequence strictly and monotonically increases with its proportion of heads.
- **iter1_candidate0**: People evaluate the randomness of a sequence based on the absolute number of heads it contains, but following the Weber-Fechner law for numerosity perception, their sensitivity to additional heads exhibits diminishing returns. When comparing two sequences, they compute a randomness score that grows logarithmically with the absolute head count plus a constant.
- **iter1_candidate2**: People evaluate the randomness of a sequence by calculating the excess of heads over tails, expecting random sequences to be heavily biased. They compute a randomness score equal to the number of heads minus the number of tails, perceiving sequences with a greater excess of heads as more random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate2 | 0.00 | 0.00 | — (best) | 1.000 |
| iter1_candidate0 | 29.01 | 2.44 | yes | 0.000 |
| iter0_candidate1 | 90.96 | 13.80 | yes | 0.000 |
| iter0_candidate0 | 91.40 | 13.92 | yes | 0.000 |
| iter1_candidate2 | 100.09 | 12.27 | yes | 0.000 |
| bayesian_diagnosticity | 579.21 | 7.43 | yes | 0.000 |
| prototype_similarity | 590.19 | 6.12 | yes | 0.000 |
| encoding_compressibility | 599.76 | 6.11 | yes | 0.000 |
