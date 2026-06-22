# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -1632.53

Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1722.85

Random-looking sequences are those with low simple-description penalties like long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -1631.89

Randomness is the log-likelihood ratio of a fair coin versus a regular process (a mixture of a complexity-penalized motif process and a biased coin).

## window_typicality  — posterior 0.000, ELPD-LOO -1711.20

Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random.

## inner_loop_model  — posterior 0.719, ELPD-LOO -1618.54

Random-looking sequences are judged by their Euclidean distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with quadratic asymmetric penalties.

## falk_konold_difficulty  — posterior 0.000, ELPD-LOO -1756.79

Random-looking sequences are those that are cognitively harder to encode into chunks, as quantified by Falk and Konold's Difficulty Predictor (the number of repetition motifs plus twice the number of alternation motifs) normalized by sequence length.

## manhattan_messy_prototype  — posterior 0.281, ELPD-LOO -1619.58

Random-looking sequences are judged by their Manhattan (absolute) distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with absolute-value penalties that are asymmetric for alternation rate.
