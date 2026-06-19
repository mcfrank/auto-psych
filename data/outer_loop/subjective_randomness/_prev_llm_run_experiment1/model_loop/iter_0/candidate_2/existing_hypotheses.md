# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.337, ELPD-LOO -72.51

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -101.81

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.663, ELPD-LOO -70.08

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## statistical_inference  — posterior 0.000, ELPD-LOO -130.66

Randomness is the log-likelihood ratio of a fair coin versus a
complexity-penalized motif process (Griffiths et al. 2018): sequences
with no short motif description are evidence for a random generator.

## iter0_candidate0

People judge a sequence as random based solely on how close its alternation rate (the proportion of adjacent pairs that differ) is to 50% — the rate expected from a fair coin. The sequence whose alternation rate is nearest to 50% looks more random, regardless of whether its heads-to-tails ratio is balanced.

## iter0_candidate1

People judge a sequence as more random when it contains a shorter maximum run — the longest unbroken streak of the same outcome. Long streaks feel like a pattern or a non-random generator at work, so the sequence whose worst streak is smallest looks most random, regardless of its overall balance or alternation rate.
