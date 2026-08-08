# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **evidence_accumulation_per_run** (posterior=0.992, elpd_loo=-782.39)
- Trials: 1280
- Models compared: 6

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| evidence_accumulation_per_run | 0.9921 | -782.39 |
| linear_evidence_accumulation_per_run | 0.0079 | -787.23 |
| minkowski_accumulated_typicality | 0.0000 | -809.40 |
| evidence_accumulation_messy_prototype | 0.0000 | -835.54 |
| artificial_balance_diagnosticity | 0.0000 | -825.29 |
| local_representativeness | 0.0000 | -845.64 |

## Hypotheses

- **evidence_accumulation_per_run**: Random-looking sequences are judged by an evidence-accumulation process
where each distinct run (streak of identical outcomes) — rather than each
item — provides a baseline weight of evidence for randomness, discounted
by the sequence's quadratic deviation from a messy prototype, so periodic
patterns with artificially few runs are penalized without a separate
periodicity cue.
- **linear_evidence_accumulation_per_run**: Random-looking sequences are judged by an evidence-accumulation process where each distinct run provides a baseline weight of evidence for randomness, discounted by the sequence's absolute linear deviation (rather than quadratic) from an ideal positive imbalance and ideal alternation rate, creating a more robust and proportional penalty for sequences that deviate from the messy prototype.
- **minkowski_accumulated_typicality**: People evaluate the randomness of a sequence by accumulating a subjective
sense of typicality over its length, computing each event's typicality as
a penalty based on its distance from a mental prototype, with the distance
raised to a freely inferred Minkowski-like exponent so extreme feature
deviations are disproportionately punished.
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
- **local_representativeness**: People judge the randomness of a sequence according to the local representativeness heuristic, meaning they expect every short contiguous segment of the sequence to independently reflect the global properties of a fair coin (equal proportions of heads and tails, and a 50% alternation rate), and they perceive a sequence as less random based on its average deviation from these ideals across all local sliding windows.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| evidence_accumulation_per_run ←selected | 0.00 | 0.00 | — (best) | 0.923 | yes |
| linear_evidence_accumulation_per_run | 4.83 | 3.23 | no (within ~2·dse) | 0.000 | yes |
| minkowski_accumulated_typicality | 27.00 | 7.40 | yes | 0.000 | yes |
| artificial_balance_diagnosticity | 42.89 | 8.08 | yes | 0.000 | yes |
| evidence_accumulation_messy_prototype | 53.14 | 9.57 | yes | 0.000 | yes |
| local_representativeness | 63.25 | 11.84 | yes | 0.077 | no ⚠ |
