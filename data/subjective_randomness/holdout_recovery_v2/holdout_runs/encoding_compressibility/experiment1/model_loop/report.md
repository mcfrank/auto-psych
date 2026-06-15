# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **prototype_similarity** (posterior=0.557, elpd_loo=-220.45)
- Trials: 600
- Models compared: 4

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| prototype_similarity | 0.5574 | -220.45 |
| iter1_candidate0 | 0.3241 | -221.29 |
| bayesian_diagnosticity | 0.1185 | -220.25 |
| iter0_candidate0 | 0.0000 | -315.62 |

## Hypotheses

- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **iter1_candidate0**: People judge a sequence as more random when it has shorter maximum runs. A long run (e.g., HHHHH) can be compactly described via run-length encoding — just name the element and its length — making the sequence feel structured and non-random. The sequence whose longest run is shorter resists this compact description and therefore appears more random.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate0**: People choose the sequence that is harder to compress. A sequence with high periodicity can be described compactly as a repeating pattern, which makes it feel structured and non-random. The sequence with lower periodicity resists this kind of compact description and therefore appears more random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| bayesian_diagnosticity | 0.00 | 0.00 | — (best) | 0.369 |
| prototype_similarity | 0.20 | 1.20 | no (within ~2·dse) | 0.203 |
| iter1_candidate0 | 1.04 | 2.07 | no (within ~2·dse) | 0.397 |
| iter0_candidate0 | 95.37 | 13.78 | yes | 0.031 |
