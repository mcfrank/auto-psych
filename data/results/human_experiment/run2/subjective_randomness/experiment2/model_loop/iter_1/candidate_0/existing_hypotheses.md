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

## inner_loop_model  — posterior 0.000, ELPD-LOO -1618.54

Random-looking sequences are judged by their Euclidean distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with quadratic asymmetric penalties.

## falk_konold_difficulty  — posterior 0.000, ELPD-LOO -1756.79

Random-looking sequences are those that are cognitively harder to encode into chunks, as quantified by Falk and Konold's Difficulty Predictor (the number of repetition motifs plus twice the number of alternation motifs) normalized by sequence length.

## manhattan_messy_prototype  — posterior 0.000, ELPD-LOO -1619.58

Random-looking sequences are judged by their Manhattan (absolute) distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with absolute-value penalties that are asymmetric for alternation rate.

## iter0_candidate0  — posterior 0.000, ELPD-LOO -1663.65

Random-looking sequences are judged by their deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate), but the cognitive penalty grows as a quartic (power of 4) function of the deviation rather than a quadratic one. This different functional form creates a much wider, flatter tolerance for near-ideal sequences but imposes significantly harsher penalties on extreme deviations like exact balance or severe imbalance.

## iter0_candidate1  — posterior 0.978, ELPD-LOO -1598.07

Hypothesis: Random-looking sequences are judged by an evidence accumulation process where each item in the sequence provides a baseline "weight of evidence" for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate). This mechanism naturally creates a preference for longer sequences when they are near-ideal, but more heavily penalizes long sequences that clearly deviate from the prototype.

## iter0_candidate2  — posterior 0.022, ELPD-LOO -1600.77

People evaluate randomness by assessing the structural diversity of a sequence's run-lengths. They compute a positive cognitive score based on the combinatorial diversity of the sequence (the log number of ways to arrange the observed counts of short, medium, and long runs) combined with a weighted preference for the sheer number of those runs. Because this mechanism accumulates positive structural evidence, longer balanced sequences naturally achieve higher scores than shorter ones, while highly regular sequences (zero diversity) and extremely imbalanced sequences (very few runs) are inherently penalized.
