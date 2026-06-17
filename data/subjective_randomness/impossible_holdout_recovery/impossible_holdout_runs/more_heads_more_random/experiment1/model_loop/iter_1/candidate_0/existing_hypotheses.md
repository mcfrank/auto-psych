# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -619.22

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -628.79

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -608.24

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## iter0_candidate0  — posterior 0.000, ELPD-LOO -120.42

People judge the randomness of a sequence by its similarity to a prototype, but rather than expecting balanced outcomes, their prototype expects a sequence to be dominated by heads. Sequences with a higher proportion of heads are perceived as closer to the prototype and therefore more random.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -119.99

People rely on a simple "heads-equal-randomness" heuristic, where heads are viewed as independent random events and tails are viewed as non-random deterministic events. Consequently, the perceived randomness of a sequence strictly and monotonically increases with its proportion of heads.

## iter0_candidate2  — posterior 1.000, ELPD-LOO -29.03

People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains. They apply a simple additive heuristic where each additional head linearly increases the perceived randomness score, disregarding the sequence length and outcome order. When comparing two sequences, they are more likely to choose the sequence with the higher total head count as the more random one.
