# Theory Report — Experiment 3

## evidence_accumulation_per_run
**Hypothesis:** Random-looking sequences are judged by an evidence accumulation process where each distinct run (streak of identical outcomes) provides a baseline weight of evidence for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype.
**Motivation:** The inner-loop model comparison showed that `iter1_candidate0` (which proposed run-based accumulation) achieved an ELPD-LOO that was statistically indistinguishable from the top model (`iter0_candidate1` / `inner_loop_model`). This model penalizes highly periodic sequences naturally (they have fewer runs per length) while retaining the core evidence-accumulation framework.
**Mechanism:** The model multiplies a baseline positive evidence score by the number of runs (`alts + 1`), then subtracts quadratic penalties for deviation from an ideal imbalance and ideal alternation rate, contrasting with `inner_loop_model` which uses the sequence length `n` instead of the number of runs.

## evidence_accumulation_periodicity
**Hypothesis:** Random-looking sequences are judged by an evidence accumulation process where baseline evidence is discounted by the sequence's deviation from a messy prototype consisting of an ideal imbalance and an ideal (low) periodicity.
**Motivation:** In the previous model loop, `iter1_candidate1` explored replacing the alternation-rate penalty with a direct penalty on periodicity. Although it lost posterior mass compared to the alternation-based mechanism, this captures an explicit hypothesis that humans detect global templates (like HHTTHHTT) rather than local transition rates.
**Mechanism:** Instead of using the alternation rate (`p_alts`), the quadratic penalty function calculates distances between the sequence's template match score (`periodicity`) and an ideal low-periodicity threshold.
