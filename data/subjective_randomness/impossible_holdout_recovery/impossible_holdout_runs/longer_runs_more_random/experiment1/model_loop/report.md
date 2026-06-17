# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **iter0_candidate2** (posterior=0.871, elpd_loo=-247.13)
- Trials: 900
- Models compared: 9

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| iter0_candidate2 | 0.8705 | -247.13 |
| iter1_candidate1 | 0.1295 | -248.89 |
| iter0_candidate1 | 0.0000 | -256.96 |
| prototype_similarity | 0.0000 | -416.07 |
| encoding_compressibility | 0.0000 | -614.77 |
| bayesian_diagnosticity | 0.0000 | -259.47 |
| iter0_candidate0 | 0.0000 | -618.44 |
| iter1_candidate0 | 0.0000 | -270.52 |
| iter1_candidate2 | 0.0000 | -624.84 |

## Hypotheses

- **iter0_candidate2**: People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. Because they believe true randomness naturally produces long streaks, they perceive a sequence as more random the longer its maximum run is relative to the total sequence length.
- **iter1_candidate1**: People judge the randomness of a sequence by comparing its longest streak of identical outcomes to their subjective ideal streak length. They perceive a sequence as more random the closer its maximum run proportion is to this expected ideal length, penalizing streaks that are either suspiciously short or excessively long.
- **iter0_candidate1**: People judge the randomness of a sequence solely by evaluating its alternation rate. Influenced by the Gambler's Fallacy, they expect random sequences to self-correct and alternate more frequently than chance, so they perceive a sequence as more random the closer its alternation rate is to their subjective ideal rate.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate0**: People judge a sequence as random based on its similarity to an ideal prototype, but they evaluate the sequence by its maximum absolute deviation (L-infinity norm) from ideal head balance and expected alternation rate, penalizing only its most salient flaw.
- **iter1_candidate0**: People judge the randomness of a sequence by focusing solely on its absolute longest streak of identical outcomes. Instead of adjusting for the total sequence length, they perceive a sequence as more random simply by counting the raw number of consecutive identical outcomes in its longest run.
- **iter1_candidate2**: People judge the randomness of a sequence solely by evaluating the overall balance of its outcomes. They perceive a sequence as more random the closer its proportion of heads is to exactly fifty percent, strictly penalizing any deviation from perfect balance.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter0_candidate2 | 0.00 | 0.00 | — (best) | 0.784 |
| iter1_candidate1 | 1.76 | 2.14 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 9.82 | 6.02 | no (within ~2·dse) | 0.000 |
| bayesian_diagnosticity | 12.33 | 6.83 | no (within ~2·dse) | 0.216 |
| iter1_candidate0 | 23.38 | 7.36 | yes | 0.000 |
| prototype_similarity | 168.93 | 14.29 | yes | 0.000 |
| encoding_compressibility | 367.63 | 19.21 | yes | 0.000 |
| iter0_candidate0 | 371.31 | 19.24 | yes | 0.000 |
| iter1_candidate2 | 377.70 | 19.26 | yes | 0.000 |
