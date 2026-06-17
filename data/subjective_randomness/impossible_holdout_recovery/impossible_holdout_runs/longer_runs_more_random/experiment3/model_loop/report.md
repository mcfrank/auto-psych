# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.434, elpd_loo=-781.78)
- Trials: 2700
- Models compared: 15

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.4338 | -781.78 |
| nonlinear_run_proportion | 0.3718 | -781.83 |
| ideal_run_proportion | 0.1041 | -783.15 |
| iter1_candidate2 | 0.0904 | -783.10 |
| prototype_similarity | 0.0000 | -1680.51 |
| encoding_compressibility | 0.0000 | -1895.51 |
| bayesian_diagnosticity | 0.0000 | -861.01 |
| absolute_ideal_run | 0.0000 | -50508.74 |
| pure_periodicity_penalty | 0.0000 | -1872.48 |
| surprising_run_length | 0.0000 | -837.64 |
| iter0_candidate0 | 0.0000 | -819.55 |
| iter0_candidate1 | 0.0000 | -846.84 |
| iter0_candidate2 | 0.0000 | -906.66 |
| iter1_candidate0 | 0.0000 | -1066.17 |
| iter1_candidate1 | 0.0000 | -1872.49 |

## Hypotheses

- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **nonlinear_run_proportion**: People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes relative to the total sequence length, but their perception of this proportion is non-linear, following a power law.
- **ideal_run_proportion**: People judge the randomness of a sequence by comparing its longest streak of identical outcomes to their subjective ideal streak proportion. They perceive a sequence as more random the closer its maximum run proportion is to this expected ideal length, penalizing streaks that are either suspiciously short or excessively long.
- **iter1_candidate2**: People judge the randomness of a sequence by comparing its longest streak proportion to a subjective ideal, but they evaluate deviations asymmetrically. Sequences that are excessively streaky (exceeding the ideal) are penalized at a different, typically steeper rate than sequences that are overly alternating (falling short of the ideal).
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **absolute_ideal_run**: People judge the randomness of a sequence by comparing its absolute longest streak of identical outcomes to a subjective ideal absolute streak length, regardless of total sequence length.
- **pure_periodicity_penalty**: People judge the randomness of a sequence strictly by penalizing its periodicity, perceiving sequences with fewer short, repeating patterns as more random.
- **surprising_run_length**: People judge the randomness of a sequence by assessing how much the absolute length of its longest streak of identical outcomes exceeds the natural logarithmic growth expected for a sequence of that total length.
- **iter0_candidate0**: People judge the randomness of a sequence by comparing its maximum run proportion to a subjective ideal proportion, but they penalize deviations from this ideal using a squared distance, causing extreme deviations to seem disproportionately less random than minor ones.
- **iter0_candidate1**: People judge the randomness of a sequence strictly by comparing its rate of alternations to a subjective ideal alternation rate. They perceive a sequence as more random the closer its alternation proportion is to this expected ideal, penalizing sequences that either alternate too much or too little.
- **iter0_candidate2**: People judge the randomness of a sequence by focusing solely on the absolute length of its longest streak of identical outcomes. Because they believe true randomness naturally produces long streaks, they perceive a sequence as more random the longer its absolute maximum run is, completely ignoring the total sequence length.
- **iter1_candidate0**: People judge the randomness of a sequence by comparing its maximum run proportion to a subjective ideal proportion, but following the Weber-Fechner law, their perception of this difference is logarithmic, penalizing the absolute log-ratio of the observed run proportion to the ideal proportion.
- **iter1_candidate1**: People judge the randomness of a sequence by comparing its maximum run proportion against a subjective tolerance limit. Sequences with a longest streak below this limit are perceived as perfectly acceptable, but any maximum run proportion exceeding the tolerance incurs a linearly increasing penalty for appearing too streaky.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| inner_loop_model | 0.00 | 0.00 | — (best) | 0.892 |
| nonlinear_run_proportion | 0.05 | 1.37 | no (within ~2·dse) | 0.000 |
| iter1_candidate2 | 1.32 | 2.52 | no (within ~2·dse) | 0.000 |
| ideal_run_proportion | 1.38 | 2.53 | no (within ~2·dse) | 0.000 |
| iter0_candidate0 | 37.77 | 9.75 | yes | 0.000 |
| surprising_run_length | 55.87 | 9.93 | yes | 0.000 |
| iter0_candidate1 | 65.06 | 11.20 | yes | 0.000 |
| bayesian_diagnosticity | 79.24 | 12.19 | yes | 0.000 |
| iter0_candidate2 | 124.88 | 14.64 | yes | 0.000 |
| iter1_candidate0 | 284.40 | 57.77 | yes | 0.099 |
| prototype_similarity | 898.73 | 33.01 | yes | 0.000 |
| pure_periodicity_penalty | 1090.71 | 33.78 | yes | 0.000 |
| iter1_candidate1 | 1090.71 | 33.78 | yes | 0.009 |
| encoding_compressibility | 1113.74 | 34.58 | yes | 0.000 |
| absolute_ideal_run | 49726.96 | 739.55 | yes | 0.000 |
