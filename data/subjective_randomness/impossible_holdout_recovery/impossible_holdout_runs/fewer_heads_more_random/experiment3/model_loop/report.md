# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter0_candidate0** (posterior=0.415, elpd_loo=-31.56)
- Trials: 2700
- Models compared: 13

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate0 | 0.4147 | -31.56 |
| inner_loop_model | 0.3388 | -31.76 |
| iter1_candidate2 | 0.1381 | -32.60 |
| iter0_candidate2 | 0.1081 | -32.80 |
| logarithmic_heads_penalty | 0.0002 | -39.10 |
| iter1_candidate1 | 0.0000 | -44.54 |
| prototype_similarity | 0.0000 | -1825.12 |
| encoding_compressibility | 0.0000 | -1875.15 |
| bayesian_diagnosticity | 0.0000 | -1501.62 |
| fewer_heads_proportion | 0.0000 | -637.45 |
| short_streaks | 0.0000 | -1872.50 |
| quadratic_heads_penalty | 0.0000 | -52.41 |
| iter0_candidate1 | 0.0000 | -51.27 |

## Hypotheses

- **iter0_candidate0**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. Their trial-by-trial comparison of these head counts follows a probit function, meaning their evaluation noise is normally rather than logistically distributed.
- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **iter1_candidate2**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. Their trial-by-trial comparison follows a probit function, reflecting normally distributed evaluation noise, but their final choices incorporate a baseline rate of random guessing due to occasional attentional lapses.
- **iter0_candidate2**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. However, their choices include a baseline rate of random guessing due to occasional attentional lapses.
- **logarithmic_heads_penalty**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, but their sensitivity to this count diminishes logarithmically, such that the difference between small head counts matters more than between large head counts.
- **iter1_candidate1**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. Their trial-by-trial comparison of these head counts follows a Cauchy CDF, meaning their evaluation noise is heavy-tailed and they are prone to occasional extreme deviations from their core preference.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **fewer_heads_proportion**: People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads are more representative of a random coin.
- **short_streaks**: People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes, judging sequences with a shorter maximum run length as more random.
- **quadratic_heads_penalty**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, but the penalty for heads grows quadratically, such that each additional head decreases perceived randomness more than the previous one.
- **iter0_candidate1**: People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. However, the perceived penalty for heads scales with the square root of the head count, meaning sensitivity diminishes gradually as the number of heads increases.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate0 | 0.00 | 0.00 | — (best) | 1.000 |
| inner_loop_model | 0.20 | 0.24 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 1.05 | 0.02 | yes | 0.000 |
| iter0_candidate2 | 1.24 | 0.25 | yes | 0.000 |
| logarithmic_heads_penalty | 7.54 | 3.96 | no (within ~2·dse) | 0.000 |
| iter1_candidate1 | 12.99 | 0.37 | yes | 0.000 |
| iter0_candidate1 | 19.71 | 4.02 | yes | 0.000 |
| quadratic_heads_penalty | 20.86 | 8.09 | yes | 0.000 |
| fewer_heads_proportion | 605.90 | 32.29 | yes | 0.000 |
| bayesian_diagnosticity | 1470.06 | 28.22 | yes | 0.000 |
| prototype_similarity | 1793.57 | 14.34 | yes | 0.000 |
| short_streaks | 1840.95 | 10.37 | yes | 0.000 |
| encoding_compressibility | 1843.60 | 14.75 | yes | 0.000 |
