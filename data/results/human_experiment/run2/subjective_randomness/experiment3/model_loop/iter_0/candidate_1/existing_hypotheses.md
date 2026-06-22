# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -2453.26

Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -2573.96

Random-looking sequences are those with low simple-description penalties like long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -2449.28

Randomness is the log-likelihood ratio of a fair coin versus a regular process (a mixture of a complexity-penalized motif process and a biased coin).

## window_typicality  — posterior 0.000, ELPD-LOO -2561.23

Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random.

## falk_konold_difficulty  — posterior 0.000, ELPD-LOO -2610.91

Random-looking sequences are those that are cognitively harder to encode into chunks, as quantified by Falk and Konold's Difficulty Predictor (the number of repetition motifs plus twice the number of alternation motifs) normalized by sequence length.

## manhattan_messy_prototype  — posterior 0.000, ELPD-LOO -2445.50

Random-looking sequences are judged by their Manhattan (absolute) distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with absolute-value penalties that are asymmetric for alternation rate.

## inner_loop_model  — posterior 0.517, ELPD-LOO -2418.99

Random-looking sequences are judged by an evidence accumulation process where each item in the sequence provides a baseline "weight of evidence" for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate).

## evidence_accumulation_per_run  — posterior 0.483, ELPD-LOO -2419.06

Random-looking sequences are judged by an evidence accumulation process where each distinct run (streak of identical outcomes) provides a baseline weight of evidence for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype.

## evidence_accumulation_periodicity  — posterior 0.000, ELPD-LOO -2616.78

Random-looking sequences are judged by an evidence accumulation process where baseline evidence is discounted by the sequence's deviation from a messy prototype consisting of an ideal imbalance and an ideal (low) periodicity.

## iter0_candidate0

Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype defined by an ideal repeating-motif rate and an ideal alternating-motif rate.
