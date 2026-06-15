# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## encoding_compressibility  — posterior 0.238, ELPD-LOO -861.23

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.181, ELPD-LOO -860.20

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## alternation_prototype  — posterior 0.000, ELPD-LOO -899.33

People judge a sequence as more random when its alternation rate is closer
(L1 distance) to an internalized prototype value that is biased above 0.5,
reflecting the well-documented human tendency to overestimate alternation in
random sequences.

## inner_loop_model  — posterior 0.348, ELPD-LOO -859.30

Best PyMC model found by the inner model-improvement loop.

## fair_coin_run_baseline  — posterior 0.000, ELPD-LOO -8153.29

People judge a sequence as more random when its maximum run length is short relative
to the expected maximum run length for a fair-coin sequence of the same length, where
the fair-coin baseline scales as kappa * log2(n) and kappa is a learned cognitive parameter.

## iter0_candidate0  — posterior 0.233, ELPD-LOO -859.70

People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky generators — exactly as in the leading model, but with the alternating generator's switch probability learned from data rather than fixed at the canonical 0.95. This tests whether people's internal representation of "how alternating" a sequence must look to seem non-random is flexible, while the streaky generator's switch probability remains fixed at its canonical value of 0.15.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -1215.99

# Hypothesis: Periodicity Aversion

People judge one sequence as more random than another based solely on how much periodic structure each sequence contains. A sequence with a pronounced repeating beat pattern looks non-random; a sequence with weak or absent periodic structure looks random. When choosing which of two sequences is more random, people pick the one with lower periodicity, and their sensitivity to periodicity differences is a single learned parameter.

## iter0_candidate2  — posterior 0.001, ELPD-LOO -867.91

# Hypothesis: Head/Tail Balance

People judge a sequence as more random when the proportion of heads is closer to 0.5 — a perfectly balanced sequence looks like the output of a fair coin, while a heavily skewed sequence looks biased. When choosing which of two sequences is more random, people simply pick the one whose proportion of heads is nearer to perfect balance, with a single sensitivity parameter governing how sharply they discriminate between imbalance levels.

## iter1_candidate0

People judge sequences by how diagnostic they are of a fair-coin process against three salient non-random alternatives — alternating, biased, and streaky generators — exactly as in the leading model. However, the characteristic bias level of the "biased" generator prototype (how extreme the head-tail imbalance must be to count as non-random) is not fixed at a canonical 0.85 but is instead learned from the data, testing whether people's internal representation of what makes a sequence look "biased" is at that conventional value or at some other level.
