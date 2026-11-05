# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **extrapolation_verification_failure** (posterior=1.000, elpd_loo=-1525.46)
- Trials: 2520
- Models compared: 2

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| extrapolation_verification_failure | 1.0000 | -1525.46 |
| range_bounded_parity_concepts | 0.0000 | -1546.85 |

## Hypotheses

- **extrapolation_verification_failure**: People infer the hidden concept by the same tempered, serial-position-weighted Bayesian concept learning as the best model (45 weighted rules and all intervals, with sorted lists shifting belief toward ranges), but to answer they must mentally verify that the target follows from the examples, and this extrapolation becomes unreliable beyond the span of numbers they have actually seen: the farther the target lies outside the examples' min–max range, the more likely the check fails and they answer "no", at an inferred rate per unit of distance. This predicts that members of a broad arithmetic rule (e.g. multiples of 3 after 63, 27, 99, 33) are endorsed less outside the examples' range than inside it, more steeply than belief in the concept alone implies, while inside-range judgments are unchanged.
- **range_bounded_parity_concepts**: Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is a single added component, taken from `bounded_range_rule_learning`: people can also restrict a rule to a range of numbers that covers the examples (e.g. "multiples of 3 between 20 and 100"), and these range-bounded rules get an inferred share of the prior for rules (share 0 recovers `parity_restricted_concepts`). A bounded rule is smaller than the full rule, so the size principle favours it. This predicts that people endorse members of the examples' rule that lie outside the examples' min-max range less than members inside it, and that this gap is larger than the rule/interval mixture allows (63,27,99,33 → 3 below 45).

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| extrapolation_verification_failure ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
| range_bounded_parity_concepts | 21.39 | 8.25 | yes | 0.000 | yes |
