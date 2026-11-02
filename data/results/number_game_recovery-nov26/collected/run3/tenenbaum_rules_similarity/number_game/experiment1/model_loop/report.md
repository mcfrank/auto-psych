# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **gamma_width_parity_range_concepts** (posterior=0.384, elpd_loo=-1072.55)
- Trials: 2250
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| gamma_width_parity_range_concepts | 0.3841 | -1072.55 |
| parity_within_range_concepts | 0.2695 | -1072.90 |
| decade_and_parity_range_concepts | 0.1247 | -1073.67 |
| example_recall_limited_concepts | 0.1111 | -1073.79 |
| multiples_within_range_concepts | 0.0370 | -1074.89 |
| rule_noticing_grows_with_examples | 0.0334 | -1074.99 |
| doubling_chain_concepts | 0.0229 | -1075.37 |
| gamma_width_exception_family_concepts | 0.0173 | -1075.65 |

## Hypotheses

- **gamma_width_parity_range_concepts**: Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the size principle. The one change, taken from `gamma_width_exception_family_concepts`, is the shape of their expectation about how wide a range (plain or parity-restricted) is: a gamma prior whose sharpness is inferred along with its typical width, instead of the fixed, loose Erlang form — so people may expect ranges of one sharply characteristic width, strongly disfavouring both very narrow and very wide ranges around the examples.
- **parity_within_range_concepts**: Refinement of the incumbent `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges with a preferred width and the size principle, but they can also entertain a conjunctive kind of concept, "odd (or even) numbers within a range" (e.g. "even numbers above 60"), with its own inferred plausibility and the same width preference. The one change is adding this parity-within-a-range kind of concept, so that after 4, 66, 78, 96 people say yes to large even numbers (70, 74) but no to small ones (8, 22), and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" evenly over all even (or all in-range) numbers.
- **decade_and_parity_range_concepts**: Refinement of `decade_concept_family_prior` (tied with the best): people do the same Bayesian concept learning over rules (family-organised prior, exception-laden rules as their own kind, and decades such as "numbers in the 60s" as their own kind of rule) and magnitude ranges with a preferred width and the size principle. The one change, taken from `parity_within_range_concepts`, is that people can also entertain "odd (or even) numbers within a range" as a kind of concept with its own inferred plausibility, so examples sharing a tens digit (61, 64, 67) confine "yes" to that decade while examples such as 4, 66, 78, 96 confine it to large even numbers — two ready-made ways of bounding generalisation that the incumbent and the decade model each have only one of.
- **example_recall_limited_concepts**: People do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, size principle), but their memory for the examples is limited: each example shown is retained, independently, only with an inferred probability, and they generalise from whichever examples they still hold. So a judgment is sometimes based on only part of the set, which broadens generalisation for longer sets (a forgotten example no longer rules out a broader or different concept) in proportion to how many examples could have been lost.
- **multiples_within_range_concepts**: Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the Erlang width preference and the size principle. The one change is that the conjunctive "rule within a range" idea extends beyond parity: people also entertain "multiples of m within a range" (e.g. "multiples of 3 between 10 and 60", "multiples of 9 from 27 to 81") as a further kind of concept with its own inferred plausibility and the same width preference. So after 39, 60, 12, 45 or 63, 27, 81 they accept multiples near the examples (e.g. 36, 54) but reject multiples far outside the examples' span (e.g. 90, 99, or 3), instead of spreading "yes" over all multiples of 3 or over every number in the range.
- **rule_noticing_grows_with_examples**: People do the same Bayesian concept learning over arithmetic rules, magnitude ranges and odd/even-within-a-range concepts as the best model, but whether a rule even comes to mind depends on how many examples they have seen: the prior odds of thinking of a rule rather than a magnitude range grow steadily, at an inferred rate, with each additional example. So after one or two examples people generalise mostly by magnitude, and after four examples they look for a rule more readily than the examples' likelihood alone would warrant.
- **doubling_chain_concepts**: People do the same Bayesian concept learning as the best model (family-organised rule prior with exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, and the size principle), but they also hold a kind of concept the best model lacks: multiplicative chains, in which each number is double (or triple) the one before, starting from some number other than 1 (e.g. 3, 6, 12, 24, 48, 96 or 5, 10, 20, 40, 80 or 2, 6, 18, 54). This kind has its own inferred plausibility. The disagreement with the best model shows up on example sets that form such a chain (e.g. 5, 10, 20 or 6, 12, 24): there this model predicts "yes" to the next links (40, 80 or 48, 96) and "no" to other multiples (15, 25, 30 or 18, 30, 36), whereas the best model spreads "yes" over all multiples of 5 (or 6) and nearby numbers.
- **gamma_width_exception_family_concepts**: Refinement of `gamma_width_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, family-organised rule prior, and a gamma prior over range width whose sharpness is inferred along with its typical width). The one change, taken from `arbitrary_exception_rule_prior`, is that rules with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") form their own kind of rule with their own, presumably very low, plausibility, instead of being ordinary power rules. So people expect ranges of a sharply characteristic width and also firmly reject 37 (and accept 32) after seeing powers of 2.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| gamma_width_parity_range_concepts ←selected | 0.00 | 0.00 | — (best) | 0.384 | yes |
| parity_within_range_concepts | 0.35 | 0.58 | no (within ~2·dse) | 0.269 | yes |
| decade_and_parity_range_concepts | 1.12 | 0.89 | no (within ~2·dse) | 0.125 | yes |
| example_recall_limited_concepts | 1.24 | 1.26 | no (within ~2·dse) | 0.111 | yes |
| multiples_within_range_concepts | 2.34 | 1.57 | no (within ~2·dse) | 0.037 | yes |
| rule_noticing_grows_with_examples | 2.44 | 1.32 | no (within ~2·dse) | 0.033 | yes |
| doubling_chain_concepts | 2.82 | 1.77 | no (within ~2·dse) | 0.023 | yes |
| gamma_width_exception_family_concepts | 3.10 | 3.59 | no (within ~2·dse) | 0.017 | yes |
