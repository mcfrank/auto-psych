# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.283, ELPD-LOO -969.42

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## bayesian_diagnosticity  — posterior 0.013, ELPD-LOO -970.79

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## max_run_length  — posterior 0.000, ELPD-LOO -1089.75

People judge a sequence as more random when its longest consecutive run is
shorter, because a long run is the most compact single-symbol run-length
encoding unit and is the strongest single cue that the sequence is structured.

## rle_description_length  — posterior 0.000, ELPD-LOO -1200.32

People judge a sequence as more random when its run-length encoding requires
more blocks (alternations + 1 normalized by length), predicting a monotonic
relationship between alternation rate and perceived randomness with no ideal rate.

## inner_loop_model  — posterior 0.044, ELPD-LOO -970.99

Best PyMC model found by the inner model-improvement loop.

## head_balance  — posterior 0.661, ELPD-LOO -968.93

People judge a sequence as more random when its head proportion is closer to 0.5,
assessing perceived randomness solely by whether the sequence looks like it came
from a fair (unbiased) coin and ignoring alternation patterns and run structure.
