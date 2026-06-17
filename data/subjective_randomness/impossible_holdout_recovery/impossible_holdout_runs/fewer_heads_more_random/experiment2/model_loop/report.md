# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.500, elpd_loo=-30.00)
- Trials: 1800
- Models compared: 11

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.4997 | -30.00 |
| iter0_candidate0 | 0.4997 | -30.00 |
| iter1_candidate2 | 0.0003 | -37.24 |
| iter1_candidate0 | 0.0003 | -37.45 |
| prototype_similarity | 0.0000 | -1223.04 |
| encoding_compressibility | 0.0000 | -1238.00 |
| bayesian_diagnosticity | 0.0000 | -1150.97 |
| fewer_heads_proportion | 0.0000 | -373.04 |
| short_streaks | 0.0000 | -1248.66 |
| iter0_candidate1 | 0.0000 | -302.72 |
| iter0_candidate2 | 0.0000 | -1248.65 |

## Hypotheses

- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **iter0_candidate0**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, rather than the proportion of heads. Sequences are penalized directly for their total count of heads, such that fewer absolute heads are judged as more representative of a random process, regardless of the sequence's length.
- **iter1_candidate2**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, but their penalty for heads grows quadratically. Each additional head decreases the perceived randomness more than the previous one, meaning the perceived difference in randomness between 8 and 9 heads is much larger than between 1 and 2.
- **iter1_candidate0**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, but their sensitivity to this count diminishes logarithmically. Sequences are penalized based on the logarithm of their head count, meaning the perceived difference in randomness between 1 and 2 heads is larger than the difference between 8 and 9.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **fewer_heads_proportion**: People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads are more representative of a random coin.
- **short_streaks**: People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes, judging sequences with a shorter maximum run length as more random.
- **iter0_candidate1**: People judge the randomness of a sequence by its Bayesian diagnosticity for a fair coin, specifically contrasting it against an alternative hypothesis that the generative process is biased towards heads. They evaluate the log-likelihood ratio of the sequence under a fair coin versus a heads-biased coin, judging sequences that provide stronger evidence against the heads bias as more random.
- **iter0_candidate2**: People evaluate the randomness of a sequence solely by checking whether its proportion of heads matches the expectation of a fair coin (0.5). Sequences whose proportion of heads is closer to 0.5 are consistently judged as being more representative of a random process.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate0 | 0.00 | 0.00 | — (best) | 0.487 |
| inner_loop_model | 0.00 | 0.00 | yes | 0.487 |
| iter1_candidate2 | 7.25 | 5.05 | no (within ~2·dse) | 0.026 |
| iter1_candidate0 | 7.46 | 3.74 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 272.72 | 16.94 | yes | 0.000 |
| fewer_heads_proportion | 343.04 | 26.56 | yes | 0.000 |
| bayesian_diagnosticity | 1120.98 | 16.85 | yes | 0.000 |
| prototype_similarity | 1193.04 | 11.97 | yes | 0.000 |
| encoding_compressibility | 1208.01 | 11.98 | yes | 0.000 |
| iter0_candidate2 | 1218.65 | 9.88 | yes | 0.000 |
| short_streaks | 1218.67 | 9.88 | yes | 0.000 |
