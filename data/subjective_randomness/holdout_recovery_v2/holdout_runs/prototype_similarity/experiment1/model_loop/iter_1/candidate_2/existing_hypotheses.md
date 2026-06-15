# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## encoding_compressibility  — posterior 0.219, ELPD-LOO -306.29

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.743, ELPD-LOO -303.77

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## iter0_candidate0  — posterior 0.000, ELPD-LOO -416.85

People judge a sequence as more random when its alternation rate is closest to the prototype value of 0.5 — the expected alternation rate for a fair coin. Randomness perception follows a Gaussian similarity function: the closer a sequence's p_alts is to 0.5, the more random it looks, with similarity falling off symmetrically as a function of squared deviation from the prototype. On each trial, people choose whichever of the two sequences is nearer to this internal prototype.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -317.79

People judge a sequence as less random the longer its longest unbroken streak of identical outcomes. The maximum run length is the single most salient cue of non-randomness: when comparing two sequences, people choose the one whose longest run is shorter as the more random one. Sensitivity to this cue varies across individuals but is captured by a single inverse-temperature parameter.

## iter0_candidate2  — posterior 0.038, ELPD-LOO -308.84

People judge a sequence as more random when its proportion of heads is closer to 50%. When comparing two sequences, they choose whichever has better head/tail balance — the smaller absolute deviation from equal proportions — as the more random one. A single inverse-temperature parameter governs how sensitively this balance difference drives the choice.

## iter1_candidate0

People judge a sequence as more random when it is more diagnostic of a fair-coin generator relative to salient non-random alternatives (alternating, biased, or streaky generators). This Bayesian diagnostic comparison is the same mechanism as the leading model, but the switch probability that defines the "alternating" prototype is inferred from participants' choices rather than fixed — the model learns how strongly alternating the non-random alternative must be to explain the data, rather than assuming a perfectly regular alternating generator.

## iter1_candidate1

People judge a sequence as non-random when it exhibits rhythmic, periodic structure — a repeating temporal pattern that stands out as too regular to be the product of a fair coin. When choosing which of two sequences looks more random, people pick the one with lower periodicity, regardless of its overall balance or run lengths. A single sensitivity parameter captures how strongly this detected periodicity difference drives the choice.
