# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **costly_squares_final_digit_concepts** (posterior=1.000, elpd_loo=-810.62)
- Trials: 2250
- Models compared: 2

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| costly_squares_final_digit_concepts | 1.0000 | -810.62 |
| place_value_disjunctive_contains | 0.0000 | -821.41 |

## Hypotheses

- **costly_squares_final_digit_concepts**: Refinement of `final_digit_multiplicative_concepts`. People still build number concepts from mental primitives: a fitted cost per extra primitive, a fitted prior discount for salient "ends in d" rules, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept. The one change splits the single cost on repeated-multiplication concepts in two. "Perfect squares" (a number multiplied by itself) pays its own fitted extra cost, separate from the cost shared by powers of a base (and their exception variants) and cubes. The claim is that self-multiplication is the hardest of these to conceive, harder than repeatedly multiplying by a fixed base. So after square sets such as 100, 1, 16 or 4, 64, 25, 81, people rarely commit to "squares". Instead they fall back on other rules, ranges or scattered guesses. Meanwhile, powers of 2 or 3 stay more available than a single cost on all of these concepts allows.
- **place_value_disjunctive_contains**: Refinement of `multiplicative_not_exponential_concepts`: people still build number concepts from mental primitives (a fitted cost per extra primitive, a fitted extra cost on powers, squares and cubes, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change is how "contains the digit d" is represented: people read numerals by place value (tens digit, ones digit), so "contains d" is not an elementary property but a disjunction, "the ones digit is d OR the tens digit is d", and it costs one more primitive than "ends in d" or "multiples of m", priced by the same per-primitive cost with no new free parameter. So when every example contains a digit (e.g. 92, 82, 27, 42), people seldom commit to "contains 2" and say no to targets that contain the digit without sharing an example's ending (23, 29, 12), as the critique reports.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| costly_squares_final_digit_concepts ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
| place_value_disjunctive_contains | 10.79 | 7.68 | no (within ~2·dse) | 0.000 | yes |
