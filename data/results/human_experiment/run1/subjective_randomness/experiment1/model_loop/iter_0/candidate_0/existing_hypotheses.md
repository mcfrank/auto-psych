# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.959, ELPD-LOO -793.68

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -841.68

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.041, ELPD-LOO -794.24

Randomness is the log-likelihood ratio of a fair coin versus a regular
process: a mixture of a complexity-penalized motif process (Griffiths et
al. 2018) and a biased coin. Merges the former diagnosticity and
statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -855.68

Random-looking sequences have a longest run typical of a fair coin seen
through a limited memory window; over-long streaks look non-random (Hahn &
Warren 2009).
