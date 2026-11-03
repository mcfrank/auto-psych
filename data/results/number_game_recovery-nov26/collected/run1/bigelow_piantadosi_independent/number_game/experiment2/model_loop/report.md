# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **rule_plus_exception_neighbourhood** (posterior=0.610, elpd_loo=-1760.33)
- Trials: 4500
- Models compared: 5

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| rule_plus_exception_neighbourhood | 0.6101 | -1760.33 |
| rule_plus_decade_exception | 0.2742 | -1761.13 |
| seen_final_digits_decade_exception | 0.1046 | -1762.10 |
| rule_with_exception_side_channel | 0.0057 | -1765.01 |
| example_anchored_exception_neighbourhood | 0.0054 | -1765.06 |

## Hypotheses

- **rule_plus_exception_neighbourhood**: Refinement of `rule_plus_exception_concepts`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus exception" concepts, but the exception they add to a rule is not a single isolated number — it is a small fuzzy region of the number line around it ("ends in 2, and also numbers around 27"), whose radius (zero, one, two … up to three steps) is a priori more likely the smaller it is, with the fall-off fitted. The one change is this neighbourhood-sized exception: when one example breaks the others' pattern (for instance, the one odd number among even ones), people still endorse it and still extend the rule to distant numbers, but they also endorse the off-pattern example's close neighbours more than the point-exception model predicts, as the critique reports.
- **rule_plus_decade_exception**: Refinement of `rule_plus_exception_concepts`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus one exception" concepts, but they can also code the exception at the place-value level, as the whole decade the off-pattern number falls in ("ends in 2, and also the twenties"), with a fitted share of the exception prior going to these rule-plus-decade concepts. The one change is this decade-level exception: when one example breaks the others' pattern, people endorse not only that example but also its same-decade neighbours (closing the critique that they generalise around the off-parity example more than the incumbent predicts), while still rejecting its neighbours just across a decade boundary and still extending the rule to distant numbers.
- **seen_final_digits_decade_exception**: Refinement of `rule_plus_decade_exception`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus exception" concepts (one extra number or its whole decade), but when the examples end in two to four different digits they also read a disjunctive final-digit rule straight off them — "ends in one of the digits I saw" (for 33, 17, 5, 9: ends in 3, 5, 7 or 9) — an example-cued concept as salient as a single final-digit rule, its prior scaled by a fitted factor. The one change is this seen-final-digits rule: because it is smaller than parity, the size principle favours it over "odd" or "even", so people extend the examples' endings to distant numbers while rejecting same-parity numbers whose final digit was never shown (e.g. 61, 71, 81 after 33, 17, 5, 9).
- **rule_with_exception_side_channel**: Refinement of `rule_plus_decade_exception`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus exception" concepts (a rule plus one extra number, or plus that number's whole decade), but they believe such a program works as a main rule with an occasional side-output: most of the time it emits a rule member, and now and then (at a fitted rate) a member of its exception, instead of drawing uniformly from rule and exception pooled together. The one change is this two-channel sampling of exception concepts: a larger exception (a whole decade) no longer weakens the evidence the rule-following examples give for the rule, so when several examples share a rule and one breaks it, people readily credit the off-pattern example's whole decade and endorse its neighbours far more than its mere proximity to other examples would warrant, while still extending the rule to distant numbers and rejecting rule-violating numbers next to the majority examples.
- **example_anchored_exception_neighbourhood**: Refinement of `rule_plus_exception_neighbourhood`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus a small fuzzy neighbourhood" concepts (radius zero to three, smaller radii a priori likelier), but they build those exceptions bottom-up from what they saw: a rule's exception region is only ever centred on an example that the rule fails to cover ("ends in 2, and also numbers around 27" because 27 was shown), never at an arbitrary place on the number line, and these anchored variants share a fitted part of the rule's prior. The one change is this example-anchored proposal of exceptions, replacing the incumbent's uniform prior over every possible exception location: when one example breaks the others' pattern, the rule-plus-neighbourhood account gets far more belief and is centred squarely on the outlier, so people endorse the outlier's close neighbours, falling off with distance from it, much more than mere neighbours of the majority examples that break the rule (closing the critique's outlier-versus-majority neighbour contrast).

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| rule_plus_exception_neighbourhood ←selected | 0.00 | 0.00 | — (best) | 0.610 | yes |
| rule_plus_decade_exception | 0.80 | 3.98 | no (within ~2·dse) | 0.274 | yes |
| seen_final_digits_decade_exception | 1.76 | 4.53 | no (within ~2·dse) | 0.105 | yes |
| rule_with_exception_side_channel | 4.68 | 3.69 | no (within ~2·dse) | 0.006 | yes |
| example_anchored_exception_neighbourhood | 4.73 | 2.58 | no (within ~2·dse) | 0.005 | yes |
