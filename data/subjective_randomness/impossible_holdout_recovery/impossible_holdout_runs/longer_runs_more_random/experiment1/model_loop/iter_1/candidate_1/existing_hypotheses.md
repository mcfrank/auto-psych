# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -416.07

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -614.77

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -259.47

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## iter0_candidate0  — posterior 0.000, ELPD-LOO -618.44

People judge a sequence as random based on its similarity to an ideal prototype, but they evaluate the sequence by its maximum absolute deviation (L-infinity norm) from ideal head balance and expected alternation rate, penalizing only its most salient flaw.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -256.96

People judge the randomness of a sequence solely by evaluating its alternation rate. Influenced by the Gambler's Fallacy, they expect random sequences to self-correct and alternate more frequently than chance, so they perceive a sequence as more random the closer its alternation rate is to their subjective ideal rate.

## iter0_candidate2  — posterior 1.000, ELPD-LOO -247.13

People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. Because they believe true randomness naturally produces long streaks, they perceive a sequence as more random the longer its maximum run is relative to the total sequence length.

## iter1_candidate0

People judge the randomness of a sequence by focusing solely on its absolute longest streak of identical outcomes. Instead of adjusting for the total sequence length, they perceive a sequence as more random simply by counting the raw number of consecutive identical outcomes in its longest run.
