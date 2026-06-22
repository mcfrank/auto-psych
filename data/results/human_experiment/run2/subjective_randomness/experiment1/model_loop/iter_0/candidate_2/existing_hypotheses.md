# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.999, ELPD-LOO -809.56

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -842.50

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.001, ELPD-LOO -813.69

Randomness is the log-likelihood ratio of a fair coin versus a regular
process: a mixture of a complexity-penalized motif process (Griffiths et
al. 2018) and a biased coin. Merges the former diagnosticity and
statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -838.52

Random-looking sequences have a longest run typical of a fair coin seen
through a limited memory window; over-long streaks look non-random (Hahn &
Warren 2009).

## iter0_candidate0

Random-looking sequences are judged by their similarity to a prototype with balanced counts and an ideal alternation rate, but the penalty for deviating from the ideal alternation rate is asymmetric: sequences that alternate less than the ideal are penalized more harshly than those that alternate more.

## iter0_candidate1

People judge a sequence's randomness by evaluating the joint typicality of its macroscopic features — specifically, the binomial probability of its head count given a fair coin, and the binomial probability of its alternation count given an ideal alternation rate. Rather than using length-invariant proportional heuristics, this mechanism naturally scales deviation penalties with sequence length and provides a statistically grounded, asymmetric tolerance for over-alternating (highly periodic) sequences.
