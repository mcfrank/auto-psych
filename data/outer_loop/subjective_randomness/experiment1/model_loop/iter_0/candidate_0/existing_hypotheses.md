# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.220, ELPD-LOO -74.87

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -98.86

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.780, ELPD-LOO -71.85

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## statistical_inference  — posterior 0.000, ELPD-LOO -115.94

Randomness is the log-likelihood ratio of a fair coin versus a
complexity-penalized motif process (Griffiths et al. 2018): sequences
with no short motif description are evidence for a random generator.
