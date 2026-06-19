# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.337, ELPD-LOO -72.51

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -101.81

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.663, ELPD-LOO -70.08

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## statistical_inference  — posterior 0.000, ELPD-LOO -130.66

Randomness is the log-likelihood ratio of a fair coin versus a
complexity-penalized motif process (Griffiths et al. 2018): sequences
with no short motif description are evidence for a random generator.

## iter0_candidate0  — posterior 0.000, ELPD-LOO -222.67

People judge a sequence as random based solely on how close its alternation rate (the proportion of adjacent pairs that differ) is to 50% — the rate expected from a fair coin. The sequence whose alternation rate is nearest to 50% looks more random, regardless of whether its heads-to-tails ratio is balanced.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -118.28

People judge a sequence as more random when it contains a shorter maximum run — the longest unbroken streak of the same outcome. Long streaks feel like a pattern or a non-random generator at work, so the sequence whose worst streak is smallest looks most random, regardless of its overall balance or alternation rate.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -169.49

People judge a sequence as more random when its proportion of heads is closer to 50% — the rate expected from a fair coin. The sequence whose head proportion is nearest to 0.5 looks most random, regardless of run structure or alternation patterns.

## iter1_candidate0

People judge a sequence as more random when it simultaneously satisfies both hallmarks of a fair-coin prototype: the alternation rate is close to 50% and the head proportion is close to 50%. The key refinement over an additive prototype model is that these two criteria are conjunctive — a sequence that is ideal on one dimension but poor on the other looks less random than one that is moderately close to the prototype on both. The mechanism is a multiplicative proximity score: each cue's proximity to its prototype value is computed separately, then the two are multiplied, so violating either criterion sharply degrades the randomness impression.

## iter1_candidate1

People judge a sequence as more random when it contains fewer repetitive sub-patterns — runs of identical outcomes of any length (e.g., "HH", "HHH", "HHHH"). Each such repetitive motif is salient evidence that the sequence came from a non-random process; the sequence with the lower total count of repetitive motifs looks more random, regardless of overall balance or how often the sequence alternates.
