# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -986.65

Random-looking sequences are close to a prototype with balanced H/T counts
and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -1204.01

Random-looking sequences are those with low simple-description penalties:
long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -541.03

Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

## inner_loop_model  — posterior 0.803, ELPD-LOO -485.40

People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. Because they believe true randomness naturally produces long streaks, they perceive a sequence as more random the longer its maximum run is relative to the total sequence length.

## ideal_run_proportion  — posterior 0.197, ELPD-LOO -486.75

People judge the randomness of a sequence by comparing its longest streak of identical outcomes to their subjective ideal streak proportion. They perceive a sequence as more random the closer its maximum run proportion is to this expected ideal length, penalizing streaks that are either suspiciously short or excessively long.

## absolute_ideal_run  — posterior 0.000, ELPD-LOO -28797.01

People judge the randomness of a sequence by comparing its absolute longest streak of identical outcomes to a subjective ideal absolute streak length, regardless of total sequence length.

## pure_periodicity_penalty  — posterior 0.000, ELPD-LOO -1248.67

People judge the randomness of a sequence strictly by penalizing its periodicity, perceiving sequences with fewer short, repeating patterns as more random.

## iter0_candidate0

People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. They evaluate this streak as a simple ratio of the maximum run length to the total sequence length, perceiving a sequence as more random the larger this absolute fraction is.
