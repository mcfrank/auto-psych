# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **bayesian_fair_coin** (posterior=0.705, elpd_loo=-108.28)
- Trials: 160
- Models compared: 9

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| bayesian_fair_coin | 0.7047 | -108.28 |
| iter0_candidate0 | 0.0499 | -111.48 |
| alternation_rate | 0.0430 | -111.43 |
| iter0_candidate2 | 0.0427 | -111.34 |
| iter0_candidate1 | 0.0403 | -111.39 |
| iter1_candidate1 | 0.0370 | -111.03 |
| equally_likely | 0.0361 | -111.65 |
| iter1_candidate2 | 0.0257 | -111.79 |
| iter1_candidate0 | 0.0205 | -111.42 |

## Hypotheses

- **bayesian_fair_coin**: Observers compare two binary sequences via the log Bayes factor between a fair-coin null and a biased-coin alternative.
- **iter0_candidate0**: People judge a sequence as more random the higher its proportion of alternations, evaluating randomness via a linear monotonic preference rather than calculating distance to a subjective ideal alternation rate.
- **alternation_rate**: People judge a sequence as more random if its proportion of alternations is closer to their subjective ideal alternation rate.
- **iter0_candidate2**: People judge a sequence as less random if it contains periodic, repeating patterns. When comparing two sequences, they prefer the one with a lower periodicity score as being more randomly generated.
- **iter0_candidate1**: People judge a sequence as less random the longer its longest continuous run of identical outcomes. When comparing two sequences, they prefer the one with the shorter maximum run length as being more random.
- **iter1_candidate1**: Observers evaluate a sequence's randomness by computing the log Bayes factor between an independent fair coin and an alternative first-order Markov process with a fixed, subjective transition probability. Rather than evaluating the overall proportion of heads, they assess the sequential dependence, preferring sequences that provide stronger evidence against the Markov alternative.
- **equally_likely**: People judge a sequence as more random the closer its proportion of heads is to 50%.
- **iter1_candidate2**: People evaluate a sequence's randomness by how statistically typical its number of alternations is. They compute the exact binomial probability of observing the sequence's number of alternations under a fair coin, and prefer the sequence whose alternation count is more mathematically probable to occur by chance.
- **iter1_candidate0**: People evaluate a sequence's randomness by comparing its likelihood under a fair coin against its probability under a biased coin, mathematically marginalizing over all possible alternative biases rather than assuming a single fixed bias. They hold a symmetric prior belief about the alternative coin's bias, with the concentration of this prior acting as a subjective parameter, and they prefer the sequence that provides stronger Bayesian evidence for the fair coin.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| bayesian_fair_coin | 0.00 | 0.00 | — (best) | 0.952 |
| iter1_candidate1 | 2.75 | 2.23 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 3.05 | 2.59 | no (within ~2·dse) | 0.048 |
| iter0_candidate1 | 3.11 | 2.36 | no (within ~2·dse) | 0.000 |
| iter1_candidate0 | 3.14 | 2.19 | no (within ~2·dse) | 0.000 |
| alternation_rate | 3.15 | 2.30 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 3.20 | 2.23 | no (within ~2·dse) | 0.000 |
| equally_likely | 3.37 | 2.15 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 3.51 | 2.60 | no (within ~2·dse) | 0.000 |
