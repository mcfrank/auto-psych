# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -473.21

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -639.57

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.035, ELPD-LOO -340.71

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## iter0_candidate0  — posterior 0.649, ELPD-LOO -339.65

People judge sequence randomness based on the Bayesian diagnosticity of a fair coin compared to a single salient alternative: a "streaky" Markov generator that has an innate tendency to repeat previous outcomes. Sequences that are more likely under the fair coin than the streaky alternative are perceived as more random.

## iter0_candidate1  — posterior 0.315, ELPD-LOO -340.77

People judge sequence randomness based purely on the alternation rate heuristic: they compare the proportion of alternating outcomes in the sequence to a subjective ideal alternation rate. Sequences whose alternation rate is closer to this subjective ideal are perceived as more random.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -624.83

People judge sequence randomness purely based on the relative frequencies of outcomes: sequences with a smaller imbalance between the number of heads and tails are perceived as more random.

## iter1_candidate0

People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving sequences with a greater imbalance between the number of heads and tails as more random.
