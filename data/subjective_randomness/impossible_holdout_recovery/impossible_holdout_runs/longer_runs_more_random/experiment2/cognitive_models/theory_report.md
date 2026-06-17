# Theory Report — Experiment 2

## ideal_run_proportion
**Hypothesis:** People judge the randomness of a sequence by comparing its longest streak of identical outcomes to their subjective ideal streak proportion. They perceive a sequence as more random the closer its maximum run proportion is to this expected ideal length, penalizing streaks that are either suspiciously short or excessively long.
**Motivation:** In experiment 1, `iter1_candidate1` (which implements this hypothesis) was statistically indistinguishable from the top model (`iter0_candidate2`), with an `elpd_diff` of only 1.76. This model explicitly tests if people have an *ideal* run proportion, contrasting with the top model's claim that *longer is always better*.
**Mechanism:** It computes the absolute difference between `max_run_norm` and an inferred subjective `ideal_run_norm`, penalizing sequences that deviate from this ideal. This is distinct from `inner_loop_model` which simply prefers larger `max_run_norm`.

## absolute_ideal_run
**Hypothesis:** People judge the randomness of a sequence by comparing its absolute longest streak of identical outcomes to a subjective ideal absolute streak length, regardless of total sequence length.
**Motivation:** Experiment 1 explored relative run lengths (`max_run_norm`) and simple absolute run maximization (`iter1_candidate0`). However, an *ideal* absolute run length hypothesis has not been tested. This complements `ideal_run_proportion` by positing that expectations of streaks are fixed counts, not proportions.
**Mechanism:** It computes the absolute difference between `max_run` (the raw count) and an inferred subjective `ideal_run` count, penalizing deviations.

## pure_periodicity_penalty
**Hypothesis:** People judge the randomness of a sequence strictly by penalizing its periodicity, perceiving sequences with fewer short, repeating patterns as more random.
**Motivation:** `encoding_compressibility` performed poorly in experiment 1 (elpd_diff > 360), but it mixed three different metrics (periodicity, imbalance, and runs). By isolating the periodicity penalty, we can cleanly test whether a rejection of simple repeating templates drives randomness judgments.
**Mechanism:** It uses a single sensitivity parameter `tau` to penalize the `periodicity` feature score, preferring whichever sequence has lower periodicity.
