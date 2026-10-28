# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **uneven_familiarity_conservative_updating** (posterior=1.000, elpd_loo=-3367.51)
- Trials: 6750
- Models compared: 1

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| uneven_familiarity_conservative_updating | 1.0000 | -3367.51 |

## Hypotheses

- **uneven_familiarity_conservative_updating**: Refinement of conservative_updating_recency_boost: people read the examples one at a time and update conservatively — after the first example (Bayesian concept learning over 42 rules with a size-dependent prior and all ranges of 1-100, perceived concept sizes inflated by a constant, a re-weighted first example, a decision gain and shifted criterion) each later example moves belief only an inferred fraction of the way toward its full Bayesian update, with the last example's update weighted by its own recency multiplier. The one change, taken from the incumbent uneven_rule_familiarity_ends, is how familiar each rule seems before any examples: on top of the size trend, each rule has an idiosyncratic familiarity drawn from a wide spread, instead of weights pulled toward one another, so a few rules ("even numbers", "multiples of 10", "powers of 2") are strongly entertained and many barely at all. This tests whether conservative, order-dependent updating beats the incumbent's order-blind accumulation once rule familiarity is allowed to be uneven.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| uneven_familiarity_conservative_updating ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
