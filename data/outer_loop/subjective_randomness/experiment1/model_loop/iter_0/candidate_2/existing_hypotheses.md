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

## iter0_candidate0

People judge which sequence looks more random by computing Bayesian diagnosticity against a single implicit alternative: a repetitive (streaky) generator that rarely alternates. The sequence whose log-likelihood ratio — fair coin versus repetitive generator — is higher is chosen as more random. The repetitive generator's alternation probability is a free parameter inferred from behavior, expected to sit well below 0.5.

## iter0_candidate1

People judge which sequence looks more random by focusing on the longest unbroken run of identical outcomes (e.g., five heads in a row). The sequence with the shorter maximum run length appears more random, because long streaks are the most perceptually salient violation of randomness expectations. All other structural features of the sequence are ignored.
