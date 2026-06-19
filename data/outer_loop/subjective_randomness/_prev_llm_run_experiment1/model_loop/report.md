# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **bayesian_diagnosticity** (posterior=0.663, elpd_loo=-70.08)
- Trials: 320
- Models compared: 10

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| bayesian_diagnosticity | 0.6631 | -70.08 |
| prototype_similarity | 0.3369 | -72.51 |
| encoding_compressibility | 0.0000 | -101.81 |
| statistical_inference | 0.0000 | -130.66 |
| iter0_candidate0 | 0.0000 | -222.67 |
| iter0_candidate1 | 0.0000 | -118.28 |
| iter0_candidate2 | 0.0000 | -169.49 |
| iter1_candidate0 | 0.0000 | -222.79 |
| iter1_candidate1 | 0.0000 | -157.38 |
| iter1_candidate2 | 0.0000 | -196.76 |

## Hypotheses

- **bayesian_diagnosticity**: Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.
- **prototype_similarity**: Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.
- **encoding_compressibility**: Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.
- **statistical_inference**: Randomness is the log-likelihood ratio of a fair coin versus a
complexity-penalized motif process (Griffiths et al. 2018): sequences
with no short motif description are evidence for a random generator.
- **iter0_candidate0**: People judge a sequence as random based solely on how close its alternation rate (the proportion of adjacent pairs that differ) is to 50% — the rate expected from a fair coin. The sequence whose alternation rate is nearest to 50% looks more random, regardless of whether its heads-to-tails ratio is balanced.
- **iter0_candidate1**: People judge a sequence as more random when it contains a shorter maximum run — the longest unbroken streak of the same outcome. Long streaks feel like a pattern or a non-random generator at work, so the sequence whose worst streak is smallest looks most random, regardless of its overall balance or alternation rate.
- **iter0_candidate2**: People judge a sequence as more random when its proportion of heads is closer to 50% — the rate expected from a fair coin. The sequence whose head proportion is nearest to 0.5 looks most random, regardless of run structure or alternation patterns.
- **iter1_candidate0**: People judge a sequence as more random when it simultaneously satisfies both hallmarks of a fair-coin prototype: the alternation rate is close to 50% and the head proportion is close to 50%. The key refinement over an additive prototype model is that these two criteria are conjunctive — a sequence that is ideal on one dimension but poor on the other looks less random than one that is moderately close to the prototype on both. The mechanism is a multiplicative proximity score: each cue's proximity to its prototype value is computed separately, then the two are multiplied, so violating either criterion sharply degrades the randomness impression.
- **iter1_candidate1**: People judge a sequence as more random when it contains fewer repetitive sub-patterns — runs of identical outcomes of any length (e.g., "HH", "HHH", "HHHH"). Each such repetitive motif is salient evidence that the sequence came from a non-random process; the sequence with the lower total count of repetitive motifs looks more random, regardless of overall balance or how often the sequence alternates.
- **iter1_candidate2**: People judge a sequence as more random when it has lower periodicity — fewer regular, repeating cycles in the pattern of outcomes. A sequence that follows a repeating temporal template (e.g., HTHTHT or HHTTHHTT) feels obviously non-random; the sequence with less periodic structure looks more like genuine noise, regardless of its overall balance or run lengths.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable.

| model | elpd_diff | dse | distinguishable from best | weight |
| --- | --- | --- | --- | --- |
| bayesian_diagnosticity | 0.00 | 0.00 | — (best) | 1.000 |
| prototype_similarity | 2.43 | 1.44 | no (within ~2·dse) | 0.000 |
| encoding_compressibility | 31.73 | 6.40 | yes | 0.000 |
| iter0_candidate1 | 48.20 | 6.88 | yes | 0.000 |
| statistical_inference | 60.58 | 6.50 | yes | 0.000 |
| iter1_candidate1 | 87.30 | 7.89 | yes | 0.000 |
| iter0_candidate2 | 99.41 | 9.25 | yes | 0.000 |
| iter1_candidate2 | 126.68 | 8.19 | yes | 0.000 |
| iter0_candidate0 | 152.59 | 7.59 | yes | 0.000 |
| iter1_candidate0 | 152.71 | 7.60 | yes | 0.000 |
