# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **digit_reversal_pair_concepts** (posterior=0.391, elpd_loo=-3516.62)
- Trials: 6750
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| digit_reversal_pair_concepts | 0.3906 | -3516.62 |
| evidence_quantity_decisiveness | 0.2073 | -3517.26 |
| set_size_response_decisiveness | 0.1275 | -3517.74 |
| last_example_recency_weight | 0.1264 | -3517.75 |
| equal_prior_rules_round_edge_ranges | 0.0592 | -3518.51 |
| set_size_dependent_example_trust | 0.0579 | -3518.53 |
| equal_prior_familiar_rules_digit_sums | 0.0176 | -3519.72 |
| graded_final_digit_equal_rule_prior | 0.0136 | -3519.98 |

## Hypotheses

- **digit_reversal_pair_concepts**: People infer the hidden program as a Bayesian concept learner over equally plausible familiar number rules, digit-sum rules and numeric ranges of a typical width, with the size principle. Their space of programs also contains "digit-reversal pair" programs: a program that produces only a two-digit number and its mirror image (e.g. 38 and 83, or 47 and 74). These programs are a kind of their own, with a prior plausibility inferred from the data. Because each one contains just two numbers, a set that holds both a number and its reversal is a strong suspicious coincidence, so people endorse the pair and reject other numbers more than the current models predict. Seeing only one member of a pair barely changes how readily people endorse its reversal.
- **evidence_quantity_decisiveness**: People infer the hidden program as a Bayesian concept learner would (equally plausible familiar number rules, digit-sum rules and numeric ranges of a typical width, with the size principle), but how decisively they act on that belief depends on how much evidence they were given: their confidence grows (or shrinks) with the number of examples shown, so after one example they answer "yes" with the probability they believe the target belongs, while after each further example the same belief is pushed further toward a firm "yes" or "no". So a target believed to belong with, say, 70% is endorsed more often after four examples than after two, which no model that maps belief to answers in the same way for every set can produce.
- **set_size_response_decisiveness**: Refinement of `last_example_recency_weight`. People infer the program as that model says: Bayesian concept learning over equally plausible familiar rules, digit-sum rules and ranges, with the size principle, and the last example given its own evidential weight. The one change is in how they answer. How decisively belief becomes a yes or no depends on how many examples were shown. After more examples, people feel better informed and push their answers toward the extremes of their belief, so confident targets are endorsed or rejected more firmly. After a single example they hedge toward 50/50, so the same strength of belief gives a less extreme yes rate. The direction and strength of this effect are inferred from the data, and no effect recovers the parent.
- **last_example_recency_weight**: People infer the hidden program as a Bayesian concept learner over familiar number rules (all equally plausible), digit-sum rules and numeric ranges of a typical width, with the size principle, but they read the examples in the order shown and the most recently seen example stays most vivid: it carries its own evidential weight (more than an earlier example if recency dominates, less if it is discounted), while the earlier examples count normally. So for multi-example sets, which concepts survive depends on which example came last (16, 8, 64 is not read exactly like 64, 8, 16), while single-example sets behave as in the ideal learner.
- **equal_prior_rules_round_edge_ranges**: Refinement of `equal_prior_familiar_rules_digit_sums`, adding one component from `round_number_range_boundaries`: people do the same Bayesian concept learning over equally plausible familiar number rules, digit-sum rules (with a prior that changes smoothly with the sum) and numeric ranges (Erlang width prior, the size principle). The one change is that multiples of ten act as reference points when people judge how plausible a range is a priori: a range whose lower or upper edge falls exactly on a multiple of ten is judged more or less likely than an otherwise identical range, with the direction and strength inferred from the data. So the magnitude-based "yes" answers stop just before, or exactly at, a nearby multiple of ten (for example, whether 40 is endorsed after examples in the low thirties). Range outputs are taken as uniform within the range, because the parent's centre-clustering was inferred to be essentially zero.
- **set_size_dependent_example_trust**: Refinement of `last_example_recency_weight`: people do the same Bayesian concept learning over equally plausible familiar number rules, digit-sum rules and numeric ranges of a typical width, with the size principle and the last-seen example carrying its own evidential weight. The one change is to how far people trust the examples as genuine outputs of the program. The incumbent fixes a 10% chance that any example is a random slip. Here the chance that an example is a slip is inferred, and it changes smoothly with how many examples were shown, so an odd-one-out example in a four-example set is written off more (or less) readily than one in a two-example set, which changes how tightly people generalise from small versus large sets.
- **equal_prior_familiar_rules_digit_sums**: Refinement of `graded_prior_digit_sum_concepts`: people do the same Bayesian concept learning over number rules, digit-sum rules (with a prior that changes smoothly with the sum) and numeric ranges (Erlang width prior, outputs clustered toward the range's middle, the size principle), but they have no learned preferences among the familiar kinds of arithmetic rule. The one change is to the rule prior: every familiar rule — odd, even, primes, squares, cubes, powers of m, multiples of k, "ends in d" — is equally plausible a priori, with a single inferred share of belief for all of them together (instead of a separate prior mass per family), and ad-hoc rules such as "contains the digit d" or "powers of 2 except 32" are not entertained at all; so how readily a set is read as, say, "multiples of 7" versus "ends in 3" depends only on how well each rule explains the examples.
- **graded_final_digit_equal_rule_prior**: Refinement of `equal_prior_familiar_rules_digit_sums`, adding one component from `graded_final_digit_digit_sum_concepts`: people do the same Bayesian concept learning over equally plausible familiar number rules, digit-sum rules (prior changing smoothly with the sum) and numeric ranges (Erlang width prior, outputs clustered toward the middle, the size principle). The one change is that the "ends in d" rules are not equally natural to each other. They keep the same total share of belief as before, but within that share a rule's prior plausibility changes smoothly with its final digit, with the direction and strength inferred from the data. So after examples that share a final digit, how readily people go on to endorse other numbers ending in that digit, rather than a range around the examples, depends on which digit it is.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| digit_reversal_pair_concepts ←selected | 0.00 | 0.00 | — (best) | 0.391 | yes |
| evidence_quantity_decisiveness | 0.63 | 2.51 | no (within ~2·dse) | 0.207 | yes |
| set_size_response_decisiveness | 1.12 | 2.26 | no (within ~2·dse) | 0.128 | yes |
| last_example_recency_weight | 1.13 | 1.78 | no (within ~2·dse) | 0.126 | yes |
| equal_prior_rules_round_edge_ranges | 1.89 | 2.31 | no (within ~2·dse) | 0.059 | yes |
| set_size_dependent_example_trust | 1.91 | 2.87 | no (within ~2·dse) | 0.058 | yes |
| equal_prior_familiar_rules_digit_sums | 3.10 | 1.93 | no (within ~2·dse) | 0.018 | yes |
| graded_final_digit_equal_rule_prior | 3.36 | 2.75 | no (within ~2·dse) | 0.014 | yes |
