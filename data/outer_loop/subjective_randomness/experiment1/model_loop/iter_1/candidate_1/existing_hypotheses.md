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

## iter0_candidate0  — posterior 0.000, ELPD-LOO -167.84

People judge which sequence looks more random by computing Bayesian diagnosticity against a single implicit alternative: a repetitive (streaky) generator that rarely alternates. The sequence whose log-likelihood ratio — fair coin versus repetitive generator — is higher is chosen as more random. The repetitive generator's alternation probability is a free parameter inferred from behavior, expected to sit well below 0.5.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -116.49

People judge which sequence looks more random by focusing on the longest unbroken run of identical outcomes (e.g., five heads in a row). The sequence with the shorter maximum run length appears more random, because long streaks are the most perceptually salient violation of randomness expectations. All other structural features of the sequence are ignored.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -171.91

People judge a sequence as more random when its proportion of heads is closer to 0.5 — that is, when its head-count is more balanced. The sequence with lower imbalance (smaller deviation from equal heads and tails) is chosen as more random. No other feature of the sequence — alternation patterns, run lengths, or motif structure — matters; only proximity to a 50/50 split drives the choice.

## iter1_candidate0

People judge sequences as random by comparing them to a mental prototype defined by two dimensions: alternation rate and head-balance. Crucially, they apply a conjunctive criterion — a sequence must be near the ideal on *both* dimensions to look random, so the dimension with the larger deviation from the prototype governs the judgment (Chebyshev / worst-case distance). A sequence that is perfectly balanced but wildly alternating looks just as non-random as one that alternates ideally but is heavily biased; only meeting both criteria simultaneously yields a high randomness rating.
