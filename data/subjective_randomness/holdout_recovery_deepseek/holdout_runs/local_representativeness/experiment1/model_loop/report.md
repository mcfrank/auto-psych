# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **minkowski_accumulated_typicality** (posterior=0.912, elpd_loo=-816.26)
- Trials: 1280
- Models compared: 5

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| minkowski_accumulated_typicality | 0.9124 | -816.26 |
| recency_window_accumulated_typicality | 0.0841 | -817.34 |
| evidence_accumulation_per_run | 0.0030 | -821.76 |
| evidence_accumulation_messy_prototype | 0.0003 | -824.23 |
| artificial_balance_diagnosticity | 0.0002 | -822.92 |

## Hypotheses

- **minkowski_accumulated_typicality**: People evaluate the randomness of a sequence by accumulating a subjective
sense of typicality over its length, computing each event's typicality as
a penalty based on its distance from a mental prototype, with the distance
raised to a freely inferred Minkowski-like exponent so extreme feature
deviations are disproportionately punished.
- **recency_window_accumulated_typicality**: People judge the randomness of a sequence by accumulating a subjective sense of
typicality, but they only accumulate it over a small recent window of the
sequence — the most recent few tosses — rather than over the whole length. This
is a refinement of the accumulated-typicality mechanism: the same per-event
typicality penalty (imbalance and alternation distance from an ideal prototype)
is applied, but restricted to the recency window, so the judgment is driven by
what is in working memory rather than the entire pattern.
- **evidence_accumulation_per_run**: Random-looking sequences are judged by an evidence-accumulation process
where each distinct run (streak of identical outcomes) — rather than each
item — provides a baseline weight of evidence for randomness, discounted
by the sequence's quadratic deviation from a messy prototype, so periodic
patterns with artificially few runs are penalized without a separate
periodicity cue.
- **evidence_accumulation_messy_prototype**: Random-looking sequences are judged by an evidence-accumulation process
where each item provides a baseline weight of evidence for randomness,
discounted by the sequence's quadratic (and asymmetric, for alternation)
deviation from a messy prototype — an ideal positive imbalance and ideal
alternation rate — so near-ideal longer sequences are preferred while
clearly deviant long sequences are penalized more heavily.
- **artificial_balance_diagnosticity**: People judge randomness by Bayesian diagnosticity — the log-likelihood
ratio of a fair coin versus a subjective "regular" generative process —
where the regular process includes an "artificial balance" generator that
targets an exactly equal count of heads and tails, explaining why people
penalize suspiciously symmetric sequences as contrived rather than random.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| minkowski_accumulated_typicality ←selected | 0.00 | 0.00 | — (best) | 0.509 | yes |
| recency_window_accumulated_typicality | 1.08 | 6.08 | no (within ~2·dse) | 0.468 | no ⚠ |
| evidence_accumulation_per_run | 5.50 | 4.81 | no (within ~2·dse) | 0.000 | yes |
| artificial_balance_diagnosticity | 6.66 | 5.30 | no (within ~2·dse) | 0.023 | yes |
| evidence_accumulation_messy_prototype | 7.97 | 4.45 | no (within ~2·dse) | 0.000 | yes |
