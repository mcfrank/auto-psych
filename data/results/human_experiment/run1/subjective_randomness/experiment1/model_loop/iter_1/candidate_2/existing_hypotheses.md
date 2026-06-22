# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -793.68

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -841.68

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -794.24

Randomness is the log-likelihood ratio of a fair coin versus a regular
process: a mixture of a complexity-penalized motif process (Griffiths et
al. 2018) and a biased coin. Merges the former diagnosticity and
statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -855.68

Random-looking sequences have a longest run typical of a fair coin seen
through a limited memory window; over-long streaks look non-random (Hahn &
Warren 2009).

## iter0_candidate0  — posterior 1.000, ELPD-LOO -784.88

People judge randomness by comparing a sequence to a mental prototype, but this prototype is subjectively biased: it possesses an ideal proportion of heads and an ideal alternation rate that may deviate from objective fairness. Sequences are perceived as more random when their proportion of heads and alternations have a smaller squared deviation from these subjective ideals.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -887.75

People judge randomness by searching for local "clumps" of identical outcomes, specifically associating moderate-length runs (pairs and triplets) with the natural clumpiness of a stochastic process. Sequences are perceived as more random when a higher proportion of their outcomes belong to these moderate-length clusters, as this simultaneously avoids the artificial regularity of strict alternation and the perceived non-randomness of overly long streaks.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -886.66

People evaluate the randomness of a sequence based on its proportion of tails, exhibiting a cognitive bias where tails are perceived as inherently more random than heads. Thus, when comparing two sequences, they are more likely to judge the sequence with a higher proportion of tails as the more random one.

## iter1_candidate0

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, rather than merely evaluating its average properties. They evaluate how well a sequence conforms to a mental prototype—represented by an ideal proportion of heads and alternations—and integrate this fit across all events. Because this typicality is accumulated, longer sequences that match the prototype accrue a higher total randomness score, while longer sequences that deviate strongly accumulate a heavier penalty, explaining why humans prefer longer sequences when average rates are equal.

## iter1_candidate1

People judge the randomness of a sequence by comparing it to a mental prototype that embodies the expected natural variability of a stochastic process, actively distrusting sequences that appear artificially "too perfect." Rather than expecting an exactly balanced sequence, they recognize that flawless balance is statistically rare and thus expect a moderate, typical degree of outcome imbalance alongside a specific alternation rate. Sequences are perceived as more random when their degree of imbalance and alternation rate have a smaller squared deviation from these subjective, non-zero ideal expectations.
