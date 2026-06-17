# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **bayesian_diagnosticity** (posterior=0.743, elpd_loo=-95.44)
- Trials: 150
- Models compared: 3

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| bayesian_diagnosticity | 0.7431 | -95.44 |
| prototype_similarity | 0.2336 | -98.35 |
| encoding_compressibility | 0.0234 | -100.20 |

## Hypotheses

- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| bayesian_diagnosticity | 0.00 | 0.00 | — (best) | 1.000 |
| prototype_similarity | 2.91 | 1.35 | yes | 0.000 |
| encoding_compressibility | 4.76 | 2.37 | yes | 0.000 |
