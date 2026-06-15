# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.636, ELPD-LOO -824.97

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.018, ELPD-LOO -828.10

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## inner_loop_model  — posterior 0.346, ELPD-LOO -825.63

People judge a sequence as more random-looking when its alternation rate is
closer to an internal prototype ideal, computed quadratically (Gaussian decay),
so small departures are disproportionately forgiven relative to large ones.

## length_sensitive_alternation  — posterior 0.000, ELPD-LOO -12323.69

People evaluate alternation deviation on the count scale rather than the
proportion scale — a sequence with two extra transitions relative to the
ideal count looks equally deviant regardless of whether it is 4 or 8 flips
long, so the randomness signal scales with sequence length.

## bayesian_markov_fairness  — posterior 0.000, ELPD-LOO -831.98

People implicitly compute the log-Bayes-factor comparing each sequence's
observed transitions against a fair Markov chain (p_transition = 0.5) versus
a biased one, and choose the sequence whose transitions are more consistent
with the fair-coin hypothesis.

## iter0_candidate0

People judge a sequence as more random-looking when its alternation rate is close to an internal ideal, but they penalize under-alternation (too few changes, making the sequence look predictable and patterned) more harshly than over-alternation (too many changes, which still looks somewhat erratic). This asymmetric sensitivity — a steeper slope on the under-alternation side than the over-alternation side — is the single mechanism driving randomness judgments.

## iter0_candidate1

People judge a sequence as more random-looking when it contains a shorter maximum consecutive run of the same outcome. A long streak of identical flips is the most salient cue that a sequence is not random, so when comparing two sequences people choose the one whose longest run is shorter, and the strength of that preference scales with how different the two maximum runs are.
