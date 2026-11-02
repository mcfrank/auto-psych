# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **without_replacement_free_outlier_rate** (posterior=0.208, elpd_loo=-906.93)
- Trials: 2250
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| without_replacement_free_outlier_rate | 0.2076 | -906.93 |
| shown_examples_not_repeated | 0.1599 | -907.19 |
| concepts_sampled_without_replacement | 0.1542 | -907.22 |
| without_replacement_size_principle | 0.1322 | -907.38 |
| round_number_salient_sampling | 0.1041 | -907.62 |
| rule_breadth_prior_concepts | 0.0882 | -907.78 |
| breadth_prior_concepts_free_outlier_rate | 0.0825 | -907.85 |
| counting_by_twos_run_concepts | 0.0713 | -908.00 |

## Hypotheses

- **without_replacement_free_outlier_rate**: Refinement of `concepts_sampled_without_replacement`: people do Bayesian concept learning over the same 45 arithmetic and digit rules (with the size-graded rule prior) and every interval of 1-100, reading each further example as a draw from the concept's members not yet shown (strong sampling without replacement), with occasional guessing at their own yes rate. The one change (the component taken from `breadth_prior_concepts_free_outlier_rate`) is that how often they believe the program shows a stray random number is a property of the learner, inferred from their answers, rather than fixed at one in ten — so a misfitting example (e.g. 14, 47 or 85, 19, 91) is discounted as a stray at the learner's own rate, letting tight, nearly exhausted concepts survive it instead of yielding to broad rules and wide ranges.
- **shown_examples_not_repeated**: Refinement of `concepts_sampled_without_replacement` (the incumbent): people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, with the size-graded rule prior, 10% of examples random, occasional guessing at their own yes rate, and the belief that the program shows distinct numbers, so each further example is a draw from the concept's members not yet shown. The one change follows from that same belief: when the number asked about is one of the examples already shown, people sometimes (at a rate inferred from their answers) read "would the program also produce it?" as asking whether it would come up again as a new example, and since the program does not repeat itself they answer "no". This predicts fewer "yes" answers to targets that are themselves examples than the concept's membership implies (about two-thirds rather than three-quarters), while leaving every other judgment governed by the incumbent's concept learning and making the guessing lapse smaller.
- **concepts_sampled_without_replacement**: Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, with a size-graded rule prior, 10% of examples random and occasional guessing at their own yes rate. They also take it that the program shows distinct numbers, so each further example is a draw from the concept's members not yet shown (strong sampling without replacement). The one change is this sampling assumption. It sharpens the size principle most for small concepts that are nearly used up by the examples (e.g. 8, 64 or 16, 8, 64 strongly favour "powers of 2", or 1, 8, 27 "cubes"), so people say "yes" less often to members of broader rules and wide ranges than a with-replacement learner predicts.
- **without_replacement_size_principle**: Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (equal rule prior, 10% of examples random, occasional guessing at their own yes rate), but they understand that the program shows distinct examples, so each further example is read as a draw from the concept's members not yet shown (sampling without replacement) rather than from all its members again. The one change is this without-replacement size principle, which adds no free parameter and predicts that several examples sharpen generalization toward small concepts more than the standard model does: a few examples that nearly exhaust a small rule (e.g. 16, 8, 64 for powers of 2) or a short range make that concept far more credible than broader alternatives.
- **round_number_salient_sampling**: People do Bayesian concept learning over the same arithmetic and digit rules (with a size-graded rule prior) and every interval of 1-100, with occasional guessing, treating the examples as distinct draws from the hidden concept (strong sampling without replacement, with some stray random examples). The one new claim is about how they believe the program picks numbers: not uniformly, but with round numbers (multiples of 5 and 10) more or less likely to be produced than other numbers, with the strength and direction of this round-number bias inferred from their answers. Because a concept's members are then weighted by how readily the program would emit them, this predicts that concepts made mostly of round numbers (e.g. "multiples of 10", "ends in 0" or "ends in 5") gain or lose credibility relative to other rules and ranges. It also predicts that a round example says less about the concept than a non-round one does, or more.
- **rule_breadth_prior_concepts**: Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random), with occasional guessing at their own yes rate — but before seeing any examples they already prefer rules by their breadth, giving each rule a prior weight that rises or falls as a power of how many numbers it contains (a preference for specific, "interesting" rules over broad ones, or the reverse). The one change is replacing the flat prior across rules with this single size-graded rule prior, which predicts that broad rules such as "odd" or "even" win less (or more) of the posterior than an equal-prior learner gives them, e.g. fewer "yes" answers to arbitrary odd numbers after 85, 19, 91.
- **breadth_prior_concepts_free_outlier_rate**: Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same 45 arithmetic and digit rules (with the same breadth-graded rule prior) and every interval of 1-100, with the size principle and occasional guessing at their own yes rate — but how often they believe the program emits a stray random example is a property of the learner, inferred from their answers, rather than fixed at one in ten. The one change is freeing this outlier rate (the component taken from `digit_concepts_inferred_outlier_rate`), which claims that part of what the lapse now absorbs is really people discounting a misfitting example as a stray, so tight rules survive an odd-one-out example (e.g. 85, 19, 91 or 14, 47) instead of being abandoned for broad ones.
- **counting_by_twos_run_concepts**: Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same arithmetic and digit rules (with their size-graded rule prior) and every interval of 1-100, with the size principle, 10% of examples random and occasional guessing. They also entertain one more kind of concept, "counting by twos" runs: a bounded stretch of every other number, such as 60, 62, 64, 66 or 31, 33, 35, i.e. the even or the odd numbers within a limited range. All such runs are equally likely a priori, and the family as a whole takes an inferred share of prior belief relative to the rules. This predicts that examples spaced by two (e.g. 60, 62, 64) draw "yes" to nearby same-parity numbers but not to same-parity numbers far away, a pattern that neither "even numbers" nor a numeric range gives.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| without_replacement_free_outlier_rate ←selected | 0.00 | 0.00 | — (best) | 0.208 | yes |
| shown_examples_not_repeated | 0.26 | 1.67 | no (within ~2·dse) | 0.160 | yes |
| concepts_sampled_without_replacement | 0.30 | 1.26 | no (within ~2·dse) | 0.154 | yes |
| without_replacement_size_principle | 0.45 | 1.65 | no (within ~2·dse) | 0.132 | yes |
| round_number_salient_sampling | 0.69 | 1.59 | no (within ~2·dse) | 0.104 | yes |
| rule_breadth_prior_concepts | 0.86 | 1.55 | no (within ~2·dse) | 0.088 | yes |
| breadth_prior_concepts_free_outlier_rate | 0.92 | 1.09 | no (within ~2·dse) | 0.083 | yes |
| counting_by_twos_run_concepts | 1.07 | 1.66 | no (within ~2·dse) | 0.071 | yes |
