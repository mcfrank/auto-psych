# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -1535.68

Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1614.88

Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -1539.38

Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -1654.87

Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).

## inner_loop_model  — posterior 0.999, ELPD-LOO -1491.38

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, rather than merely evaluating its average properties. They evaluate how well a sequence conforms to a mental prototype—represented by an ideal proportion of heads and alternations—and integrate this fit across all events.

## accumulated_alternation_typicality  — posterior 0.000, ELPD-LOO -1512.62

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this typicality is based entirely on how closely the sequence's alternation rate matches a mental prototype, completely ignoring the overall proportion of heads and tails.

## linear_accumulated_typicality  — posterior 0.001, ELPD-LOO -1497.97

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal head proportion and alternation rate is linear (absolute difference) rather than quadratic, treating extreme deviations proportionally to small ones.

## iter0_candidate0

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this evidence integration exhibits diminishing returns. Rather than growing linearly, the accumulated typicality scales according to a power-law function of sequence length, which prevents extremely long sequences from disproportionately dominating judgments and explains why the preference for longer sequences saturates.

## iter0_candidate1

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but working memory for past events is leaky. The evidence contributed by each event decays exponentially over time, meaning the total accumulated randomness score saturates for longer sequences, which prevents large differences in sequence length from having an oversized effect on choice.
