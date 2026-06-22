# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -793.68

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -841.68

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -794.24

Randomness is the log-likelihood ratio of a fair coin versus a regular
process: a mixture of a complexity-penalized motif process (Griffiths et
al. 2018) and a biased coin. Merges the former diagnosticity and
statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -855.68

Random-looking sequences have a longest run typical of a fair coin seen
through a limited memory window; over-long streaks look non-random (Hahn &
Warren 2009).

## iter0_candidate0  — posterior 1.000, ELPD-LOO -784.88

People judge randomness by comparing a sequence to a mental prototype, but this prototype is subjectively biased: it possesses an ideal proportion of heads and an ideal alternation rate that may deviate from objective fairness. Sequences are perceived as more random when their proportion of heads and alternations have a smaller squared deviation from these subjective ideals.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -887.75

People judge randomness by searching for local "clumps" of identical outcomes, specifically associating moderate-length runs (pairs and triplets) with the natural clumpiness of a stochastic process. Sequences are perceived as more random when a higher proportion of their outcomes belong to these moderate-length clusters, as this simultaneously avoids the artificial regularity of strict alternation and the perceived non-randomness of overly long streaks.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -886.66

People evaluate the randomness of a sequence based on its proportion of tails, exhibiting a cognitive bias where tails are perceived as inherently more random than heads. Thus, when comparing two sequences, they are more likely to judge the sequence with a higher proportion of tails as the more random one.
