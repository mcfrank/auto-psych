# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **exception_weighted_rule_rejection** (posterior=1.000, elpd_loo=-4092.86)
- Trials: 7560
- Models compared: 1

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| exception_weighted_rule_rejection | 1.0000 | -4092.86 |

## Hypotheses

- **exception_weighted_rule_rejection**: I refine the incumbent `magnitude_threshold_concepts`, keeping its tempered, sample-size-neglecting Bayesian concept learning over weighted rules and their odd/even versions, size-weighted ranges with their own evidence strength and one-sided "above x"/"below x" thresholds, fully consistent bounded rules, the lone-example discount and the cumulative explaining away of implied rules. The one change: people judge a mathematical or digit rule by its exceptions, so an example that breaks a rule counts against it by its own inferred amount rather than the fixed cost implied by treating 10% of examples as random strays (no extra cost recovers the incumbent), which separates how firmly a rule is rejected for an exception from how strongly the size principle favours small rules, while ranges keep the stray account. If exceptions weigh more than strays imply, a rule that one example breaks gives up belief to the ranges covering every example, so for sets that no rule fits people should generalize more inside the examples' span and less to rule members outside it, while the size principle among fitting rules can be gentler.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| exception_weighted_rule_rejection ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
