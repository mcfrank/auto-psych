# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.011, ELPD-LOO -1229.02

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1245.87

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## length_sensitive_alternation  — posterior 0.000, ELPD-LOO -21626.27

People evaluate alternation deviation on the count scale rather than the
proportion scale — a sequence with two extra transitions relative to the
ideal count looks equally deviant regardless of whether it is 4 or 8 flips
long, so the randomness signal scales with sequence length.

## bayesian_markov_fairness  — posterior 0.000, ELPD-LOO -1249.26

People implicitly compute the log-Bayes-factor comparing each sequence's
observed transitions against a fair Markov chain (p_transition = 0.5) versus
a biased one, and choose the sequence whose transitions are more consistent
with the fair-coin hypothesis.

## inner_loop_model  — posterior 0.759, ELPD-LOO -1224.93

People judge a sequence as more random-looking when it is close to an internal
2D prototype specifying both an ideal alternation rate and balanced heads and
tails, with Gaussian (quadratic) decay in each dimension independently.

## run_length_prototype  — posterior 0.000, ELPD-LOO -1239.19

People judge sequences by how close the maximum run length is to an internal
prototype for a random sequence — both too-long streaks and too-short maximum
runs look non-random, so sequences whose longest run matches the ideal are
judged most random.

## length_sensitive_2d_prototype  — posterior 0.230, ELPD-LOO -1226.02

People evaluate sequences using the two-dimensional prototype (alternation
rate and balance), but their sensitivity to deviations scales linearly with
sequence length because longer sequences provide stronger statistical evidence
of non-randomness for the same proportional deviation.
