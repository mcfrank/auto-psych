# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -1223.04

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1238.00

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -1150.97

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## inner_loop_model  — posterior 0.500, ELPD-LOO -30.00

Best PyMC model found by the inner model-improvement loop.

## fewer_heads_proportion  — posterior 0.000, ELPD-LOO -373.04

People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads are more representative of a random coin.

## short_streaks  — posterior 0.000, ELPD-LOO -1248.66

People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes, judging sequences with a shorter maximum run length as more random.

## iter0_candidate0  — posterior 0.500, ELPD-LOO -30.00

People judge the randomness of a sequence strictly by the absolute number of heads it contains, rather than the proportion of heads. Sequences are penalized directly for their total count of heads, such that fewer absolute heads are judged as more representative of a random process, regardless of the sequence's length.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -302.72

People judge the randomness of a sequence by its Bayesian diagnosticity for a fair coin, specifically contrasting it against an alternative hypothesis that the generative process is biased towards heads. They evaluate the log-likelihood ratio of the sequence under a fair coin versus a heads-biased coin, judging sequences that provide stronger evidence against the heads bias as more random.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -1248.65

People evaluate the randomness of a sequence solely by checking whether its proportion of heads matches the expectation of a fair coin (0.5). Sequences whose proportion of heads is closer to 0.5 are consistently judged as being more representative of a random process.
