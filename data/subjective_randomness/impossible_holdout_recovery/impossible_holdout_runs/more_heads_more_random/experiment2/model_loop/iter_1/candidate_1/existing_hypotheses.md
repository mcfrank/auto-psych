# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -1018.50

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1069.79

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -899.91

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## inner_loop_model  — posterior 0.264, ELPD-LOO -29.29

People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains.

## length_scaled_head_difference  — posterior 0.000, ELPD-LOO -41.12

People evaluate randomness primarily by the absolute number of heads, but their sensitivity to the difference in head counts between two sequences is diminished when the overall length of the sequences being compared is larger.

## squared_heads_heuristic  — posterior 0.727, ELPD-LOO -28.28

People evaluate the randomness of a sequence strictly based on the squared number of heads it contains, amplifying the perception of randomness for sequences with very high head counts.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -129.07

People evaluate the randomness of a sequence strictly based on the proportion of heads it contains, perceiving sequences with a higher proportion of heads as more random.

## iter0_candidate2  — posterior 0.009, ELPD-LOO -32.63

People evaluate the randomness of a sequence strictly based on the cubed number of heads it contains. This mechanism creates an extreme, accelerating non-linear preference where sequences with high head counts are overwhelmingly perceived as more random.

## iter1_candidate0

People evaluate the randomness of a sequence based solely on the number of heads it contains, but their perception of randomness scales as a power-law function of the head count rather than strictly linearly or quadratically. The model infers this exponent to capture exactly how marginal increases in head counts shape judgments.
