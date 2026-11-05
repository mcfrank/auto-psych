# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **consistent_examples_bounded_rules** (posterior=1.000, elpd_loo=-3088.24)
- Trials: 5040
- Models compared: 1

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| consistent_examples_bounded_rules | 1.0000 | -3088.24 |

## Hypotheses

- **consistent_examples_bounded_rules**: Refinement of `comparison_dependent_concepts` (the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, with digit rules and bounded rules discounted when only one example is shown). The one change takes that model's own premise further. A rule confined to the examples' span ("ends in 3 between 48 and 63") is built by comparing the examples and seeing that they share the rule, so people entertain it only when every example fits the rule. A bounded rule that one of the examples breaks is never considered, rather than kept with that example written off as a stray, and no parameter is added. People should then drop a rule that most examples share but one breaks, and fall back on the examples' range or on whole rules. In-range numbers lacking the majority pattern (48,63,53 → 49, 51) should be endorsed more than the incumbent predicts, while sets that a rule fits fully are unchanged.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| consistent_examples_bounded_rules ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
