# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.246, ELPD-LOO -622.08

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## bayesian_diagnosticity  — posterior 0.043, ELPD-LOO -622.08

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## inner_loop_model  — posterior 0.246, ELPD-LOO -622.08

Best PyMC model found by the inner model-improvement loop.

## max_run_length  — posterior 0.000, ELPD-LOO -662.49

People judge a sequence as more random when its longest consecutive run is
shorter, because a long run is the most compact single-symbol run-length
encoding unit and is the strongest single cue that the sequence is structured.

## rle_description_length  — posterior 0.000, ELPD-LOO -725.77

People judge a sequence as more random when its run-length encoding requires
more blocks (alternations + 1 normalized by length), predicting a monotonic
relationship between alternation rate and perceived randomness with no ideal rate.

## iter0_candidate0  — posterior 0.370, ELPD-LOO -621.37

People judge a sequence as more random when it is close to a prototype with balanced H/T counts and an ideal alternation rate, where closeness is measured by squared deviation rather than absolute deviation. This means the randomness gradient is steepest exactly at the prototype — even modest departures are penalized disproportionately — rather than declining at a constant rate regardless of how far the sequence already is from ideal.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -832.76

People judge a sequence as more random when it contains less periodic structure. Sequences that repeat a regular cycle (e.g., HTHT… or HHTTHHTT…) are perceived as non-random because the mind detects the underlying period. When comparing two sequences, people choose the one with lower periodicity as the more random-looking.

## iter0_candidate2  — posterior 0.096, ELPD-LOO -623.42

People judge a sequence as more random when its proportion of heads is closer to 0.5 — that is, when the sequence is more balanced. They ignore run structure, alternation patterns, and periodicity entirely; only the overall H/T ratio guides the judgment. The sequence with the less imbalanced coin-flip count looks more random.

## iter1_candidate0

People judge sequence randomness by comparing each sequence to a mental prototype with balanced heads-tails counts and an ideal alternation rate, penalizing deviations quadratically (as in the leading existing model). However, the penalty is asymmetric around the ideal alternation rate: sequences that are too streaky — fewer alternations than ideal — are treated as more non-random than sequences that are equally distant in the over-alternating direction. This asymmetry reflects the well-documented human tendency to find runs of the same symbol particularly diagnostic of non-randomness, stronger than the reverse surprise of excessive alternation.
