# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **parity_veto_sparse_familiar_rules** (posterior=0.826, elpd_loo=-2025.39)
- Trials: 4500
- Models compared: 3

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| parity_veto_sparse_familiar_rules | 0.8264 | -2025.39 |
| rule_or_range_kind_commitment | 0.1555 | -2027.06 |
| sparse_rule_familiarity_within_kinds | 0.0181 | -2029.22 |

## Hypotheses

- **parity_veto_sparse_familiar_rules**: Refinement of `parity_conflict_decision_veto`: people infer the hidden concept by capacity-limited Bayesian learning over rules (multiples of 5 as its own base-ten kind) and numerical ranges, and a target whose parity matches none of the examples triggers an automatic parity conflict that, on a fixed fraction of such trials, overrides the inferred concept with a "no". The single change is grafting sparse rule familiarity (from `sparse_rule_familiarity_within_kinds`) onto the concept-learning stage: most rules share their kind's plausibility, and only a few stand out as exceptionally familiar (multiples of 11, powers of 10) or unfamiliar (multiples of 7, "contains the digit 1"). The claim is that the decision-stage parity veto survives once rule priors are calibrated this way: opposite-parity targets in all-odd or all-even sets (e.g. odd numbers inside the span of 20, 32, 92) get fewer yeses than the inferred concept implies, while same-parity targets are unaffected.
- **rule_or_range_kind_commitment**: People learn the hidden concept from the examples by the same capacity-limited Bayesian weighing of arithmetic rules and numerical ranges as the best model, but they do not blend the two kinds of explanation when they answer: they first commit to whether the program follows an arithmetic rule or produces a stretch of the number line, by comparing how well each kind as a whole explains the examples more decisively than an ideal learner would, and then judge the target only by the concepts of the kind they chose. This predicts that when rules and ranges both explain the examples moderately well, answers follow whichever kind is ahead more sharply than the blend implies, e.g. fewer yeses to far rule-members when ranges are slightly favored, and fewer yeses to in-span rule-violators when a rule is slightly favored.
- **sparse_rule_familiarity_within_kinds**: Refinement of `rule_familiarity_within_kinds`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with a prior plausibility for each kind of rule and rule-specific familiarities within each kind, but familiarity is sparse: most rules of a kind are about as familiar as one another and simply share the kind's plausibility, while only a few stand out as exceptionally familiar (e.g. multiples of 11, powers of 10) or exceptionally unfamiliar (e.g. multiples of 7, "contains the digit 1"). The single change is replacing the incumbent's evenly spread (normal) familiarities with heavy-tailed ones, which predicts that rules with no standout familiarity are generalized exactly as their kind's other rules are, instead of each being nudged up or down by a moderate amount, while the few exceptional rules keep their large deviations.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| parity_veto_sparse_familiar_rules ←selected | 0.00 | 0.00 | — (best) | 0.826 | yes |
| rule_or_range_kind_commitment | 1.67 | 3.34 | no (within ~2·dse) | 0.156 | yes |
| sparse_rule_familiarity_within_kinds | 3.82 | 2.84 | no (within ~2·dse) | 0.018 | yes |
