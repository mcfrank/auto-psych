# Theory Report — Experiment 2

## raw_alternation_count
**Hypothesis:** People perceive sequences with a higher raw count of alternations as more random, regardless of sequence length.
**Motivation:** Experiment 1's best model relied purely on outcome frequencies (imbalance), but it was potentially fitting a spurious correlation induced by the holdout design. Given the true phenomenon likely involves alternations (as hinted by the ground-truth name), this model tests whether people use a simple count of alternations as a heuristic.
**Mechanism:** It uses the raw `alts` feature directly in a sigmoid comparison, proposing that a larger absolute number of alternations increases subjective randomness.

## high_alternation_rate
**Hypothesis:** People judge sequence randomness based on the proportion of alternations, perceiving sequences with a higher alternation rate as strictly more random.
**Motivation:** This refines the alternation heuristic to be length-sensitive. Since sequences of different lengths naturally have different expected alternation counts, humans might normalize by the maximum possible alternations.
**Mechanism:** It uses the `p_alts` feature, directly comparing the alternation proportions between the two sequences, with higher proportion leading to higher subjective randomness.

## ideal_alternation_rate
**Hypothesis:** People judge sequence randomness by comparing the proportion of alternating outcomes to a subjective ideal alternation rate, perceiving sequences closer to this ideal as more random.
**Motivation:** `iter0_candidate1` from Experiment 1 tested a similar hypothesis but lost posterior mass. However, given the failure modes of the models in the holdout regime, it is critical to keep a pure distance-to-ideal-rate hypothesis in the pool for Experiment 2, to distinguish it cleanly from the imbalance models.
**Mechanism:** It measures the absolute difference between `p_alts` and a free parameter `ideal_rate`, with a smaller difference resulting in a higher subjective randomness score.
