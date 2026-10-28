# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **outlier_tolerant_rule_concepts** (posterior=1.000, elpd_loo=-2796.55)
- Trials: 5040
- Models compared: 1

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| outlier_tolerant_rule_concepts | 1.0000 | -2796.55 |

## Hypotheses

- **outlier_tolerant_rule_concepts**: Refinement of `evidence_scarce_guessing_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100 (evidence weight growing with set size, guessing that becomes rarer as more examples are shown), but how readily they accept that an example could be a stray number the program produced outside its rule is their own inferred belief, rather than a fixed one-in-ten chance. The one change is this inferred tolerance for exceptions, which matters most for sets where all examples but one share a rule (e.g. 92, 82, 27, 42 or 71, 61, 26, 81): a tolerant learner keeps the near-fitting rule and says "yes" to far targets that fit it (52, 62; 11, 41), while an intolerant learner abandons the rule for a range spanning every example and rejects them.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| outlier_tolerant_rule_concepts ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
