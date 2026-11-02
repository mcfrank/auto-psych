# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **salient_decade_place_value_rules** (posterior=1.000, elpd_loo=-2585.84)
- Trials: 6750
- Models compared: 2

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| salient_decade_place_value_rules | 0.9999 | -2585.84 |
| individual_rule_vs_range_style | 0.0001 | -2595.47 |

## Hypotheses

- **salient_decade_place_value_rules**: Refinement of `salient_contained_digit_rules`: people do the same tempered, exception-tolerant Bayesian concept learning (salient final-digit and contained-digit rules, ranges on a compressed number line with a gamma-shaped width prior, and rule-plus-neighbourhood exceptions anchored on an off-pattern example and emitted through a side channel), but they also read the tens digit of a numeral as a place-value category: "numbers in the forties" (tens digit 4) is a rule they entertain directly, with its own fitted salience, not only as one range among the thousands of possible stretches of the number line. The one change is this set of salient decade rules: when the examples share a tens digit (e.g. 42, 47, 45), people fill in exactly that decade, endorsing its members, even those far from any example, and rejecting numbers just across the decade boundary (39, 50) more sharply than a learner whose only magnitude concepts are ranges with arbitrary edges.
- **individual_rule_vs_range_style**: Refinement of `salient_contained_digit_rules`: people do the same tempered, exception-tolerant Bayesian concept learning (salient final-digit and contained-digit rules, ranges on a compressed number line with a gamma-shaped width prior, and rule-plus-neighbourhood exceptions emitted through a side channel), but individuals differ stably in how readily they think of the hidden program as an arithmetic or digit rule rather than a range of magnitudes. The one change is this person-specific prior share for rules versus ranges, fitted for each person: some participants consistently extend shared rules to distant numbers, while others consistently stay near the examples' stretch of the number line across every set they judge.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| salient_decade_place_value_rules ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
| individual_rule_vs_range_style | 9.63 | 6.88 | no (within ~2·dse) | 0.000 | yes |
