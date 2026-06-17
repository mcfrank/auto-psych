# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter1_candidate0** (posterior=1.000, elpd_loo=-29.95)
- Trials: 900
- Models compared: 7

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter1_candidate0 | 1.0000 | -29.95 |
| prototype_similarity | 0.0000 | -616.87 |
| encoding_compressibility | 0.0000 | -628.16 |
| bayesian_diagnosticity | 0.0000 | -600.16 |
| iter0_candidate0 | 0.0000 | -619.59 |
| iter0_candidate1 | 0.0000 | -624.80 |
| iter0_candidate2 | 0.0000 | -218.89 |

## Hypotheses

- **iter1_candidate0**: People judge the randomness of a sequence based purely on its absolute count of heads, rather than the proportion of heads. They hold a biased belief that sequences containing fewer total heads are more representative of a random coin, and thus penalize sequences based directly on their total head count independent of sequence length.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate0**: People judge randomness by comparing the likelihood of the sequence under a fair coin to its likelihood under a single Markov alternative model that generates alternations at a non-random rate. The transition probability of this alternative model is a free cognitive parameter.
- **iter0_candidate1**: People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes. Relying on a representativeness heuristic, they expect short runs and penalize sequences with longer streaks, judging sequences with a shorter maximum run length as more random.
- **iter0_candidate2**: People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads (and thus more tails) are more representative of a random coin.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate0 | 0.00 | 0.00 | — (best) | 1.000 |
| iter0_candidate2 | 188.95 | 17.97 | yes | 0.000 |
| bayesian_diagnosticity | 570.21 | 11.78 | yes | 0.000 |
| prototype_similarity | 586.92 | 10.61 | yes | 0.000 |
| iter0_candidate0 | 589.65 | 10.22 | yes | 0.000 |
| iter0_candidate1 | 594.85 | 9.83 | yes | 0.000 |
| encoding_compressibility | 598.22 | 9.98 | yes | 0.000 |
