# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **seed_multiple_salience_overlap** (posterior=1.000, elpd_loo=-1475.40)
- Trials: 2520
- Models compared: 2

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| seed_multiple_salience_overlap | 1.0000 | -1475.40 |
| final_digit_overshadowing_overlap | 0.0000 | -1490.13 |

## Hypotheses

- **seed_multiple_salience_overlap**: People notice a relation between the examples themselves that the property-overlap models ignore: whether the smallest example divides every other example (7 and 63, 9 and 81, 8 and 64). When it does, they read that smallest number as the program's seed and take "multiples of the seed" as a salient property of the set, with its own salience. As in the overlap account, they still say "yes" in proportion to the salience-weighted share of the examples' common properties that the target also has, so a multiple of the seed is endorsed even when it breaks other shared properties such as parity (e.g. 14 and 28 for 7, 63).
- **final_digit_overshadowing_overlap**: Refinement of `one_sided_magnitude_bound_overlap` by one change: a final digit that is visibly shared by all of several examples (e.g. 28, 98) overshadows every other shared property. People still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but when the examples visibly share a final digit, all the other shared properties (parity, multiples, contained digits, magnitude bounds, and so on) lose an inferred fraction of their salience. This disagrees most sharply with the incumbent on shared-final-digit sets: targets that keep the digit but break the other shared properties (28, 98 -> 18, 38) are endorsed far more, and targets that fit the other properties but not the digit (28, 98 -> 14, 56) far less.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| seed_multiple_salience_overlap ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
| final_digit_overshadowing_overlap | 14.73 | 6.11 | yes | 0.000 | yes |
