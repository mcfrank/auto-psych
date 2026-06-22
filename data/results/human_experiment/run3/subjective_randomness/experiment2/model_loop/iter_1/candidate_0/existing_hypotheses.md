# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.004, ELPD-LOO -1720.11

Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1765.14

Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -1723.64

Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -1769.42

Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).

## inner_loop_model  — posterior 0.000, ELPD-LOO -1726.02

People judge the randomness of a sequence by comparing its features (head proportion and alternation rate) to a subjective ideal, but their psychological penalty for deviations is scaled by the square root of the sequence length, reflecting an intuitive sensitivity to the standard error of small samples.

## smoothed_prototype_distance  — posterior 0.811, ELPD-LOO -1714.40

People judge the randomness of a sequence by comparing its feature proportions to a subjective ideal, but they estimate these proportions using Bayesian smoothing with subjective pseudo-counts, naturally tolerating extreme imbalances in short sequences because their prior pulls the estimates toward the ideal.

## falk_konold_complexity  — posterior 0.000, ELPD-LOO -1774.59

People judge the randomness of a sequence by its structural complexity when parsed into continuous alternating and repeating sub-sequences (Falk & Konold's difficulty of encoding), perceiving sequences with a lower rate of sub-sequences as simpler and therefore less random.

## iter0_candidate0  — posterior 0.185, ELPD-LOO -1715.18

People judge the randomness of a sequence by its smoothed distance from a subjective prototype, but their ideal template includes an expectation for higher-order alternating motifs (like HTHT) in addition to basic heads/tails balance and simple bigram alternations.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -1769.01

People judge the randomness of a sequence by its smoothed distance from a subjective prototype, but this prototype tracks the proportion of 4-item alternating motifs (like HTHT) instead of simple bigram alternations, viewing these longer, more complex alternations as the primary structural signature of local randomness.
