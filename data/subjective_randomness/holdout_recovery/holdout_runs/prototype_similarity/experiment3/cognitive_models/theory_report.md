# Theory Report — Experiment 3

## Context

The experiment 2 model comparison (10 models, 1200 trials) produced a near-flat
ELPD-LOO posterior. The top four models (`iter1_candidate1`, `inner_loop_model`,
`asymmetric_alternation_prototype`, `iter1_candidate2`) are statistically
indistinguishable (all within 2·dse of best). All rely on some form of
alternation-rate deviation plus imbalance. `bayesian_diagnosticity` was the
clear loser (ELPD diff = 1.98 ≈ 2·dse = 2.86 — borderline distinguishable).

The two new models target cognitive mechanisms that have not been adequately
tested: (1) normalizing alternation deviation by the full theoretical variance
rather than by n, and (2) template-matching (periodicity) as an independent
non-randomness cue.

---

## runs_test_model

**Motivation:** The winning model (`inner_loop_model`/`iter1_candidate1`)
computes deviation from expected alternations E[alts] = 2ht/n and normalizes
by n. This normalization is arbitrary — it treats the magnitude of a deviation
as independent of the balance of the sequence. The Wald-Wolfowitz runs test
provides a principled alternative: normalize by sqrt(Var[R]) where
Var[R] = 2ht(2ht − n) / (n²(n−1)). For balanced sequences, Var[R] is larger,
so the same deviation is less surprising; for imbalanced sequences, Var[R]
collapses toward zero, making any alternation deviation very salient.
This could distinguish the two models on trials where h ≠ n/2.

**Mechanism:** Computes the Wald-Wolfowitz z-score per sequence — how many
standard deviations the observed run count R = alts + 1 lies from its
expectation under a fair coin. Applies an asymmetric penalty (streak_k)
that down-weights over-alternation relative to under-alternation, consistent
with gambler's-fallacy sensitivity. Combines with imbalance via a learned
imbalance_weight, identical to the inner_loop_model.

---

## periodicity_salience

**Motivation:** Every competitive model from experiment 2 is built around
alternation statistics. The `periodicity` feature — match to a short repeating
template — has not been the primary mechanism in any top model
(`encoding_compressibility` includes it as one of three compressed-description
terms but ranks 8th). Template-matching is a distinct representativeness cue:
a sequence can alternate frequently without matching any period-2 or period-3
template, and vice versa. If the winning model absorbs all variance in
alternation-based terms, we cannot tell whether periodicity salience is real
or redundant. A dedicated model isolates this question.

**Mechanism:** Non-randomness score is a weighted sum of the precomputed
periodicity feature and imbalance (complement weight). No alternation term.
A single free weight (periodicity_weight) lets the data determine whether
template salience or balance drives the penalty. The model makes predictions
that diverge from alternation-based models on pairs where one sequence has
high periodicity but moderate p_alts, while the other has low periodicity
but extreme p_alts.
