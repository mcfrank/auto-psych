# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **no_digit_containment_span_concepts** (posterior=1.000, elpd_loo=-1671.74)
- Trials: 4500
- Models compared: 1

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| no_digit_containment_span_concepts | 1.0000 | -1671.74 |

## Hypotheses

- **no_digit_containment_span_concepts**: People do the same Bayesian concept learning as the current best model (arithmetic and final-digit rules plus multiples of 13-50, rule runs confined to a stretch of the number line, ranges of a typical width, an inferred share of random examples, examples drawn without replacement), except that they never entertain "contains the digit d" concepts: a digit that merely appears somewhere in the written numbers (such as a shared tens digit) is not noticed as a pattern. This simpler concept space predicts that near targets get no extra "yes" for sharing an example's decade and that sets like 8, 80, 48 do not generalise to other numbers containing an 8.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| no_digit_containment_span_concepts ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
