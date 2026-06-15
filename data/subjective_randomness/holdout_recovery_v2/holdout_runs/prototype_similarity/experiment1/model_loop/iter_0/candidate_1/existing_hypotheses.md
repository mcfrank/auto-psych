# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## encoding_compressibility  — posterior 0.228, ELPD-LOO -306.29

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.772, ELPD-LOO -303.77

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## iter0_candidate0

People judge a sequence as more random when its alternation rate is closest to the prototype value of 0.5 — the expected alternation rate for a fair coin. Randomness perception follows a Gaussian similarity function: the closer a sequence's p_alts is to 0.5, the more random it looks, with similarity falling off symmetrically as a function of squared deviation from the prototype. On each trial, people choose whichever of the two sequences is nearer to this internal prototype.
