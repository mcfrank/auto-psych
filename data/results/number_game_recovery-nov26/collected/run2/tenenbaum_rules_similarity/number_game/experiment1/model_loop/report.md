# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **without_replacement_example_sampling** (posterior=1.000, elpd_loo=-1206.51)
- Trials: 2250
- Models compared: 1

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| without_replacement_example_sampling | 1.0000 | -1206.51 |

## Hypotheses

- **without_replacement_example_sampling**: People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples), but they assume the program never produces the same number twice, so they treat the examples as drawn from the concept without replacement: each further example that fits a concept is taken as one of that concept's members not yet shown. This makes several examples far stronger evidence for a concept that has barely more members than were shown (cubes, powers of 3 or 4, squares, a tight range) than the best model's with-replacement size principle allows, so the sharpest disagreement lives on three- and four-example sets that nearly exhaust a small rule or range, where this model predicts fewer "yes" answers to targets outside that small concept.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| without_replacement_example_sampling ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
