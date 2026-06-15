# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **inner_loop_model** (posterior=0.271, elpd_loo=-859.30)
- Trials: 1800
- Models compared: 11

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| inner_loop_model | 0.2712 | -859.30 |
| iter1_candidate0 | 0.2201 | -859.26 |
| encoding_compressibility | 0.1856 | -861.23 |
| iter0_candidate0 | 0.1815 | -859.70 |
| bayesian_diagnosticity | 0.1409 | -860.20 |
| iter0_candidate2 | 0.0006 | -867.91 |
| alternation_prototype | 0.0000 | -899.33 |
| fair_coin_run_baseline | 0.0000 | -8153.29 |
| iter0_candidate1 | 0.0000 | -1215.99 |
| iter1_candidate1 | 0.0000 | -1220.92 |
| iter1_candidate2 | 0.0000 | -928.11 |

## Hypotheses

- **inner_loop_model**: Best PyMC model found by the inner model-improvement loop.
- **iter1_candidate0**: People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky generators — exactly as in the leading model. However, the characteristic bias level of the "biased" generator prototype (how extreme the head-tail imbalance must be to count as non-random) is not fixed at a canonical 0.85 but is instead learned from the data, testing whether people's internal representation of what makes a sequence look "biased" is at that conventional value or at some other level.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **iter0_candidate0**: People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky generators — exactly as in the leading model, but with the alternating generator's switch probability learned from data rather than fixed at the canonical 0.95. This tests whether people's internal representation of "how alternating" a sequence must look to seem non-random is flexible, while the streaky generator's switch probability remains fixed at its canonical value of 0.15.
- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **iter0_candidate2**: # Hypothesis: Head/Tail Balance

People judge a sequence as more random when the proportion of heads is closer to 0.5 — a perfectly balanced sequence looks like the output of a fair coin, while a heavily skewed sequence looks biased. When choosing which of two sequences is more random, people simply pick the one whose proportion of heads is nearer to perfect balance, with a single sensitivity parameter governing how sharply they discriminate between imbalance levels.
- **alternation_prototype**: People judge a sequence as more random when its alternation rate is closer
(L1 distance) to an internalized prototype value that is biased above 0.5,
reflecting the well-documented human tendency to overestimate alternation in
random sequences.
- **fair_coin_run_baseline**: People judge a sequence as more random when its maximum run length is short relative
to the expected maximum run length for a fair-coin sequence of the same length, where
the fair-coin baseline scales as kappa * log2(n) and kappa is a learned cognitive parameter.
- **iter0_candidate1**: # Hypothesis: Periodicity Aversion

People judge one sequence as more random than another based solely on how much periodic structure each sequence contains. A sequence with a pronounced repeating beat pattern looks non-random; a sequence with weak or absent periodic structure looks random. When choosing which of two sequences is more random, people pick the one with lower periodicity, and their sensitivity to periodicity differences is a single learned parameter.
- **iter1_candidate1**: # Hypothesis: Learned Streaky Generator Diagnosticity

People judge which sequence looks more random by computing how much better a fair coin explains each sequence compared with the most compelling non-random alternative — an alternating, biased-coin, or streaky (run-generating) generator. The alternating generator's switch probability and the biased generator's bias level are fixed at canonical values, but the streaky generator's characteristic switch probability — how likely it is to stay on the same outcome — is a flexible cognitive parameter that people learn from experience rather than inheriting from a fixed prototype.
- **iter1_candidate2**: # Hypothesis: Absolute Maximum-Run Aversion

People judge a sequence as more random when its longest unbroken streak of the same outcome is shorter. They treat any long run as a direct signal of non-randomness, comparing the two sequences' maximum run lengths and picking the one with the shorter streak, with a single sensitivity parameter governing how sharply they discriminate.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| iter1_candidate0 | 0.00 | 0.00 | — (best) | 0.000 |
| inner_loop_model | 0.04 | 0.19 | no (within ~2·dse) | 0.505 |
| iter0_candidate0 | 0.44 | 0.43 | no (within ~2·dse) | 0.000 |
| bayesian_diagnosticity | 0.95 | 0.42 | yes | 0.000 |
| encoding_compressibility | 1.97 | 4.56 | no (within ~2·dse) | 0.265 |
| iter0_candidate2 | 8.65 | 6.16 | no (within ~2·dse) | 0.168 |
| alternation_prototype | 40.07 | 9.14 | yes | 0.062 |
| iter1_candidate2 | 68.86 | 10.62 | yes | 0.000 |
| iter0_candidate1 | 356.73 | 22.74 | yes | 0.000 |
| iter1_candidate1 | 361.66 | 22.95 | yes | 0.000 |
| fair_coin_run_baseline | 7294.03 | 367.27 | yes | 0.000 |
