# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **bayesian_fair_coin** (posterior=0.343, elpd_loo=-210.40)
- Trials: 310
- Models compared: 11

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| bayesian_fair_coin | 0.3432 | -210.40 |
| inner_loop_model | 0.3432 | -210.40 |
| iter1_candidate0 | 0.2813 | -211.00 |
| alternation_rate | 0.0130 | -214.02 |
| subjective_markov_evidence | 0.0047 | -214.68 |
| iter1_candidate1 | 0.0039 | -215.12 |
| absolute_alternation_deviation | 0.0027 | -215.40 |
| equally_likely | 0.0026 | -215.69 |
| iter0_candidate1 | 0.0025 | -215.56 |
| iter0_candidate2 | 0.0015 | -215.69 |
| iter0_candidate0 | 0.0013 | -215.66 |

## Hypotheses

- **bayesian_fair_coin**: Observers compare two binary sequences via the log Bayes factor between a fair-coin null and a biased-coin alternative.
- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **iter1_candidate0**: Refining the hypothesis that observers track the frequency of heads, we propose they evaluate subjective randomness using a simple directional heuristic. Rather than penalizing symmetric deviations from a 50% proportion, observers judge a sequence as more random simply if it contains a greater absolute number of heads.
- **alternation_rate**: People judge a sequence as more random if its proportion of alternations is closer to their subjective ideal alternation rate.
- **subjective_markov_evidence**: Observers evaluate randomness by computing the log Bayes factor of the sequence's transitions under a subjective ideal Markov process versus a purely independent fair coin.
- **iter1_candidate1**: Observers evaluate the randomness of a binary sequence using a simple directional heuristic based on alternations. Rather than penalizing deviations from a specific ideal rate, they judge a sequence as more random simply if it contains a higher proportion of alternations.
- **absolute_alternation_deviation**: People judge a sequence as less random the further its number of alternations deviates from the expected number under their subjective ideal rate, computing distance in absolute counts rather than proportions.
- **equally_likely**: People judge a sequence as more random the closer its proportion of heads is to 50%.
- **iter0_candidate1**: Observers evaluate the randomness of a sequence based on its longest unbroken streak of identical outcomes. They use the length of this maximum run as a heuristic for non-randomness, judging sequences with longer streaks as less random.
- **iter0_candidate2**: Observers evaluate the randomness of a sequence using a point-estimate log-likelihood ratio rather than full Bayesian integration. They estimate the sequence's bias by smoothing the empirical proportion of heads with subjective pseudo-counts, and judge the sequence as less random the more its likelihood under this estimated bias exceeds its likelihood under a fair coin.
- **iter0_candidate0**: Observers evaluate the randomness of a sequence by computing the true log Bayes factor between a fair-coin null and a biased-coin alternative, marginalizing over possible alternative biases using a subjective Beta prior instead of comparing to a single fixed bias.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| bayesian_fair_coin | 0.00 | 0.00 | — (best) | 0.352 |
| inner_loop_model | 0.00 | 0.00 | no (within ~2·dse) | 0.352 |
| iter1_candidate0 | 0.60 | 1.54 | no (within ~2·dse) | 0.157 |
| alternation_rate | 3.62 | 3.28 | no (within ~2·dse) | 0.139 |
| subjective_markov_evidence | 4.28 | 2.98 | no (within ~2·dse) | 0.000 |
| iter1_candidate1 | 4.72 | 2.92 | no (within ~2·dse) | 0.000 |
| absolute_alternation_deviation | 5.00 | 3.10 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 5.16 | 2.93 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 5.26 | 2.89 | no (within ~2·dse) | 0.000 |
| equally_likely | 5.29 | 2.87 | no (within ~2·dse) | 0.000 |
| iter0_candidate2 | 5.29 | 2.89 | no (within ~2·dse) | 0.000 |
