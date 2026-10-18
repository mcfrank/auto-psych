# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **even_rule_digit_size_principle_counting** (posterior=0.973, elpd_loo=-4677.17)
- Trials: 7560
- Models compared: 4

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| even_rule_digit_size_principle_counting | 0.9731 | -4677.17 |
| bayesian_digit_veto_counting_up | 0.0268 | -4680.76 |
| parity_and_digit_size_principle | 0.0000 | -4687.55 |
| digit_reversal_mirror_concept | 0.0000 | -4687.80 |

## Hypotheses

- **even_rule_digit_size_principle_counting**: A refinement of the incumbent `digit_rule_size_principle_counting` (shortfall from the examples' mutual final-digit similarity, rule-pre-empted parity credit, a privileged round common-factor rule, rule-gated decade-band familiarity, step-costed counting up from single examples, and the size principle on a shared final-digit rule, all kept). The one change, a component taken from `privileged_named_number_concepts`, is that "even numbers" is a privileged, named concept while "odd numbers" is not: when every example is even, people readily take "even numbers" as the program's rule and reject odd targets by an extra, inferred amount (e.g. 37 after 16, 8, 64 or after 68, 58), whereas an all-odd set evokes no "odd numbers" rule and its even targets pay nothing extra.
- **bayesian_digit_veto_counting_up**: A refinement of `bayesian_digit_rule_veto` (shortfall from the examples' mutual final-digit similarity, rule-pre-empted parity credit, the privileged "even numbers" rule, separate round and non-round common-factor boosts, rule-gated decade-band familiarity, and the Bayesian "ends in digit d" belief that vetoes targets ending in another digit, all kept). The one change, a component taken from `counting_up_from_single_example`, is that people also count up mentally from each single example of 3 or more (x, 2x, 3x, ...), so a target reached this way gets extra support toward "yes" even when it is not a multiple of what all the examples share (e.g. 66 after 33, 3; 48 after 4, 16, 12), strongest for double an example and fading by an inferred cost per further counting step. The claim is that the saturating digit-rule veto and step-costed counting together explain judgments better than the incumbent's linear per-example digit penalty.
- **parity_and_digit_size_principle**: A refinement of the incumbent `digit_rule_size_principle_counting` (final-digit similarity judged by its shortfall from the examples' mutual similarity, rule-pre-empted parity credit, a privileged round common-factor rule, rule-gated decade-band familiarity, step-costed counting up from single examples and the size principle on a shared final digit, all kept). The one change extends the size principle to parity: when every example is even (or every example is odd), each example beyond the first is another suspicious coincidence that firms up an "even numbers" (or "odd numbers") rule, so a target of the opposite parity is rejected by an inferred, additional amount per confirming example — e.g. 37 is rejected more after 4, 16, 12, 64 than after 4, 16 — whereas the incumbent's mutual-similarity shortfall is the same for two or four same-parity examples.
- **digit_reversal_mirror_concept**: Among the programs people consider is a digit-reversal concept: the program writes the examples' digits backwards, so a number that is one of the examples with its two digits swapped (27 after 72, 83 after 38, 4 after 40) feels like something the program would produce. Such mirror targets get extra support toward "yes", by an inferred amount, beyond what their final digit, parity, magnitude or arithmetic relation to the examples would predict. All other judgments follow the current best model (final-digit similarity judged by its shortfall from the examples' mutual similarity, rule-pre-empted parity credit, privileged round common-factor rule, rule-gated decade-band familiarity, step-costed counting up from single examples, and the size principle on a shared final-digit rule).

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| even_rule_digit_size_principle_counting ←selected | 0.00 | 0.00 | — (best) | 0.973 | yes |
| bayesian_digit_veto_counting_up | 3.59 | 1.78 | yes | 0.027 | yes |
| parity_and_digit_size_principle | 10.38 | 5.85 | no (within ~2·dse) | 0.000 | yes |
| digit_reversal_mirror_concept | 10.64 | 5.50 | no (within ~2·dse) | 0.000 | yes |
