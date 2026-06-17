# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -1825.12

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1875.15

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -1501.62

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## fewer_heads_proportion  — posterior 0.000, ELPD-LOO -637.45

People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads are more representative of a random coin.

## short_streaks  — posterior 0.000, ELPD-LOO -1872.50

People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes, judging sequences with a shorter maximum run length as more random.

## inner_loop_model  — posterior 0.999, ELPD-LOO -31.76

Best PyMC model found by the inner model-improvement loop.

## logarithmic_heads_penalty  — posterior 0.001, ELPD-LOO -39.10

People judge the randomness of a sequence strictly by the absolute number of heads it contains, but their sensitivity to this count diminishes logarithmically, such that the difference between small head counts matters more than between large head counts.

## quadratic_heads_penalty  — posterior 0.000, ELPD-LOO -52.41

People judge the randomness of a sequence strictly by the absolute number of heads it contains, but the penalty for heads grows quadratically, such that each additional head decreases perceived randomness more than the previous one.

## iter0_candidate0

People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. Their trial-by-trial comparison of these head counts follows a probit function, meaning their evaluation noise is normally rather than logistically distributed.

## iter0_candidate1

People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. However, the perceived penalty for heads scales with the square root of the head count, meaning sensitivity diminishes gradually as the number of heads increases.
