# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.460, ELPD-LOO -622.08

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## bayesian_diagnosticity  — posterior 0.080, ELPD-LOO -622.08

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## inner_loop_model  — posterior 0.460, ELPD-LOO -622.08

Best PyMC model found by the inner model-improvement loop.

## max_run_length  — posterior 0.000, ELPD-LOO -662.49

People judge a sequence as more random when its longest consecutive run is
shorter, because a long run is the most compact single-symbol run-length
encoding unit and is the strongest single cue that the sequence is structured.

## rle_description_length  — posterior 0.000, ELPD-LOO -725.77

People judge a sequence as more random when its run-length encoding requires
more blocks (alternations + 1 normalized by length), predicting a monotonic
relationship between alternation rate and perceived randomness with no ideal rate.

## iter0_candidate0

People judge a sequence as more random when it is close to a prototype with balanced H/T counts and an ideal alternation rate, where closeness is measured by squared deviation rather than absolute deviation. This means the randomness gradient is steepest exactly at the prototype — even modest departures are penalized disproportionately — rather than declining at a constant rate regardless of how far the sequence already is from ideal.
