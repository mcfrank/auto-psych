# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.568, elpd_loo=-485.40)
- Trials: 1800
- Models compared: 12

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.5677 | -485.40 |
| iter1_candidate0 | 0.2871 | -486.03 |
| ideal_run_proportion | 0.1394 | -486.75 |
| iter0_candidate0 | 0.0058 | -489.99 |
| prototype_similarity | 0.0000 | -986.65 |
| encoding_compressibility | 0.0000 | -1204.01 |
| bayesian_diagnosticity | 0.0000 | -541.03 |
| absolute_ideal_run | 0.0000 | -28797.01 |
| pure_periodicity_penalty | 0.0000 | -1248.67 |
| iter0_candidate1 | 0.0000 | -528.10 |
| iter0_candidate2 | 0.0000 | -552.65 |
| iter1_candidate1 | 0.0000 | -581.53 |

## Hypotheses

- **inner_loop_model**: People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. Because they believe true randomness naturally produces long streaks, they perceive a sequence as more random the longer its maximum run is relative to the total sequence length.
- **iter1_candidate0**: People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes relative to the total sequence length. However, their perception of this proportion is non-linear; they subjectively evaluate the normalized maximum run length according to a power law before comparing the sequences, allowing for either diminishing or accelerating marginal returns for longer streaks.
- **ideal_run_proportion**: People judge the randomness of a sequence by comparing its longest streak of identical outcomes to their subjective ideal streak proportion. They perceive a sequence as more random the closer its maximum run proportion is to this expected ideal length, penalizing streaks that are either suspiciously short or excessively long.
- **iter0_candidate0**: People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. They evaluate this streak as a simple ratio of the maximum run length to the total sequence length, perceiving a sequence as more random the larger this absolute fraction is.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **absolute_ideal_run**: People judge the randomness of a sequence by comparing its absolute longest streak of identical outcomes to a subjective ideal absolute streak length, regardless of total sequence length.
- **pure_periodicity_penalty**: People judge the randomness of a sequence strictly by penalizing its periodicity, perceiving sequences with fewer short, repeating patterns as more random.
- **iter0_candidate1**: People judge the randomness of a sequence by comparing its alternation rate to their subjective ideal alternation rate. They perceive a sequence as more random the closer its alternation proportion is to this expected ideal, penalizing sequences that either alternate too rarely (streaky) or too frequently (perfectly alternating).
- **iter0_candidate2**: People judge the randomness of a sequence based solely on the overall balance of its outcomes. They perceive a sequence as less random the more its proportion of heads deviates from an even split, heavily penalizing any imbalance.
- **iter1_candidate1**: People judge the randomness of a sequence simply by the absolute length of its longest streak of identical outcomes. They perceive a sequence as more random the longer its absolute maximum run is, completely ignoring the total length of the sequence in their evaluation.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 0.666 |
| iter1_candidate0 | 0.63 | 0.85 | no (within ~2·dse) | 0.000 |
| ideal_run_proportion | 1.35 | 2.48 | no (within ~2·dse) | 0.290 |
| iter0_candidate0 | 4.59 | 2.36 | no (within ~2·dse) | 0.000 |
| iter0_candidate1 | 42.70 | 9.40 | yes | 0.000 |
| bayesian_diagnosticity | 55.63 | 10.41 | yes | 0.000 |
| iter0_candidate2 | 67.25 | 12.85 | yes | 0.041 |
| iter1_candidate1 | 96.13 | 13.05 | yes | 0.000 |
| prototype_similarity | 501.25 | 24.63 | yes | 0.000 |
| encoding_compressibility | 718.61 | 27.71 | yes | 0.000 |
| pure_periodicity_penalty | 763.27 | 27.36 | yes | 0.003 |
| absolute_ideal_run | 28311.61 | 605.49 | yes | 0.000 |
