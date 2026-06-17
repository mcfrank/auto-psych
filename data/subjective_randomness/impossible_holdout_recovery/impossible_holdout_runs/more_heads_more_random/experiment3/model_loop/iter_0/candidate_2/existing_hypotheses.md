# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -1586.78

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1716.39

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -1296.25

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## length_scaled_head_difference  — posterior 0.000, ELPD-LOO -59.04

People evaluate randomness primarily by the absolute number of heads, but their sensitivity to the difference in head counts between two sequences is diminished when the overall length of the sequences being compared is larger.

## squared_heads_heuristic  — posterior 0.011, ELPD-LOO -53.48

People evaluate the randomness of a sequence strictly based on the squared number of heads it contains, amplifying the perception of randomness for sequences with very high head counts.

## inner_loop_model  — posterior 0.011, ELPD-LOO -53.48

Best PyMC model found by the inner model-improvement loop.

## power_law_heads  — posterior 0.000, ELPD-LOO -163912.14

People evaluate the randomness of a sequence based on the number of heads it contains, but their perception of randomness scales as an inferred power-law function of the head count.

## absolute_heads_lapse  — posterior 0.977, ELPD-LOO -48.92

People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains, but their choices are subject to a constant lapse rate representing random guessing.

## iter0_candidate0

People evaluate the randomness of a sequence based on the logarithm of the absolute number of heads it contains, reflecting a diminishing sensitivity to head count differences as the total number of heads increases, and their choices are subject to a constant lapse rate for random guessing.

## iter0_candidate1

People evaluate the randomness of a sequence primarily based on the proportion of heads it contains relative to its total length, judging sequences with a higher fraction of heads to be more random, and their choices are subject to a constant lapse rate representing occasional random guessing.
