# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **encoding_compressibility** (posterior=0.539, elpd_loo=-113.70)
- Trials: 240
- Models compared: 4

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| encoding_compressibility | 0.5392 | -113.70 |
| iter0_candidate1 | 0.2199 | -115.60 |
| iter0_candidate0 | 0.2049 | -115.77 |
| bayesian_diagnosticity | 0.0361 | -115.11 |

## Hypotheses

- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **iter0_candidate1**: People evaluate the randomness of a coin flip sequence based on its alternation rate. Sequences with a higher proportion of alternations between outcomes are perceived as more random due to an expectation that random processes switch frequently.
- **iter0_candidate0**: People judge the randomness of a sequence by the proportion of the sequence made up of its longest continuous run, perceiving sequences where a single run occupies a larger fraction of the total length as less random.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| encoding_compressibility | 0.00 | 0.00 | — (best) | 0.546 |
| bayesian_diagnosticity | 1.41 | 2.15 | no (within ~2·dse) | 0.163 |
| iter0_candidate1 | 1.90 | 2.73 | no (within ~2·dse) | 0.082 |
| iter0_candidate0 | 2.07 | 2.88 | no (within ~2·dse) | 0.210 |
