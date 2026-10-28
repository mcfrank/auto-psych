# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **scale_end_avoiding_ranges** (posterior=0.487, elpd_loo=-1246.17)
- Trials: 2250
- Models compared: 6

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| scale_end_avoiding_ranges | 0.4869 | -1246.17 |
| weber_blurred_ranges_kind_pooled_rules | 0.1683 | -1247.23 |
| family_pooled_rule_prior | 0.1661 | -1247.25 |
| range_restricted_rule_concepts | 0.0961 | -1247.79 |
| random_example_forgetting | 0.0488 | -1248.47 |
| parity_kind_rule_prior | 0.0337 | -1248.84 |

## Hypotheses

- **scale_end_avoiding_ranges**: People do Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 exactly as in `family_pooled_rule_prior` (rule-kind prior shares, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but they believe the program's range is unlikely to run all the way to the ends of the 1-100 scale. Each end of a candidate range that sits at 1 or at 100 changes its prior by an inferred factor, so people generalize less toward the extremes of the scale — fewer "yes" answers to numbers in the 90s above high examples (and near 1 below low ones) than a learner who treats the scale's ends like any other number.
- **weber_blurred_ranges_kind_pooled_rules**: Refinement of `weber_noisy_range_membership`: people do Bayesian concept learning over 42 arithmetic rules and every range of 1-100 (size principle, a prior favouring moderately wide ranges), and they check whether a target falls inside a range on a noisy mental number line whose noise grows with magnitude (Weber's law), so range edges are blurred, more among large numbers, while rule membership is checked exactly. The one change, grafted from `family_pooled_rule_prior`: people's prior over rules is set by the kind of rule (powers, multiples, final digit, contained digit, special sets), with all rules of one kind equally likely, instead of a separate free weight for each of the 42 rules — so the blurred-edge account stops being handicapped by an overfitted rule prior, and range generalization ends in a magnitude-scaled soft edge rather than at decade boundaries.
- **family_pooled_rule_prior**: Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (rule share, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but their prior over rules is set by the kind of rule, not rule by rule. The one change: people hold a belief about how likely each kind of rule is (powers, multiples, final digit, contained digit, and special sets such as primes, evens, odds, squares, cubes) and treat all rules of one kind as equally likely, so "multiples of 7" is as plausible a priori as "multiples of 4", instead of the incumbent's separate free weight for each of the 42 rules.
- **range_restricted_rule_concepts**: People do the same Bayesian concept learning as the best model (42 arithmetic rules and every interval of 1-100, a kind-level rule prior, an Erlang width prior favouring decade-bounded ranges, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their hypothesis space also contains range-restricted rules — conjunctions such as "even numbers in the 60s-70s" or "multiples of 4 below 20" — which receive an inferred share of prior belief and, being smaller, are strongly favoured by the size principle when the examples cluster. The sharpest disagreement with the best model lives on rule-consistent targets far from the examples' span (e.g. 12 or 30 after 66,78; 64 or 100 after 4,16,12): the best model says "yes" to every member of the rule wherever it lies, while this model predicts that "yes" to rule members falls off outside the region the examples occupy.
- **random_example_forgetting**: People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (as in the best model: kind-level rule prior, size principle, a width prior and a preference for decade-bounded ranges), but their working memory for the examples is unreliable: each example shown is independently lost from memory with an inferred probability (at least one is always retained), and people generalize only from the examples they still hold. So their judgment is an average of what the ideal learner would conclude from each remembered subset of the examples — larger sets are more often reduced to fewer examples, which weakens the size principle's tightening and lets broader concepts and single-example generalizations survive, irrespective of the order the examples were shown in.
- **parity_kind_rule_prior**: Refinement of `family_pooled_rule_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target, rules of one kind equally likely a priori), but parity is a kind of rule in its own right. The one change: "even numbers" and "odd numbers" form their own kind with their own inferred prior share, instead of being lumped with primes, squares and cubes as one "special sets" kind, so how readily people fall back on odd/even after examples like 3,7 or 66,78 is set apart from how readily they think of primes, squares or cubes.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| scale_end_avoiding_ranges ←selected | 0.00 | 0.00 | — (best) | 0.487 | yes |
| weber_blurred_ranges_kind_pooled_rules | 1.06 | 1.82 | no (within ~2·dse) | 0.168 | yes |
| family_pooled_rule_prior | 1.08 | 1.11 | no (within ~2·dse) | 0.166 | yes |
| range_restricted_rule_concepts | 1.62 | 1.62 | no (within ~2·dse) | 0.096 | yes |
| random_example_forgetting | 2.30 | 1.50 | no (within ~2·dse) | 0.049 | yes |
| parity_kind_rule_prior | 2.67 | 1.90 | no (within ~2·dse) | 0.034 | yes |
