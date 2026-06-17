# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -616.87

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -628.16

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -600.16

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## iter0_candidate0  — posterior 0.000, ELPD-LOO -619.59

People judge randomness by comparing the likelihood of the sequence under a fair coin to its likelihood under a single Markov alternative model that generates alternations at a non-random rate. The transition probability of this alternative model is a free cognitive parameter.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -624.80

People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes. Relying on a representativeness heuristic, they expect short runs and penalize sequences with longer streaks, judging sequences with a shorter maximum run length as more random.

## iter0_candidate2  — posterior 1.000, ELPD-LOO -218.89

People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads (and thus more tails) are more representative of a random coin.

## iter1_candidate0

People judge the randomness of a sequence based purely on its absolute count of heads, rather than the proportion of heads. They hold a biased belief that sequences containing fewer total heads are more representative of a random coin, and thus penalize sequences based directly on their total head count independent of sequence length.
