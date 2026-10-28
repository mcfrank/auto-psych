# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **criterion_shifted_redundant_examples** (posterior=0.426, elpd_loo=-777.66)
- Trials: 2250
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| criterion_shifted_redundant_examples | 0.4264 | -777.66 |
| range_bounded_rule_concepts | 0.1990 | -778.43 |
| round_landmark_interval_prior | 0.1820 | -778.51 |
| criterion_shifted_tempered_decision | 0.1318 | -778.84 |
| stochastic_example_forgetting | 0.0589 | -779.64 |
| redundant_examples_decision_gain | 0.0010 | -783.73 |
| win_stay_lose_shift_criterion_shift | 0.0005 | -784.43 |
| split_rule_range_tempering | 0.0004 | -784.64 |

## Hypotheses

- **criterion_shifted_redundant_examples**: Refinement of criterion_shifted_tempered_decision: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and turn their belief into a yes/no answer through the same decision gain and shifted (biased) criterion on its log-odds, but they treat additional examples as partly redundant rather than as independent draws. The one change, taken from redundant_examples_decision_gain, is that the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it, so with sublinear growth one example already supports a fairly tight concept while three or four sharpen generalization less than strong sampling predicts, while the criterion shift separately sets the overall lean toward "yes" (an exponent of one recovers the incumbent).
- **range_bounded_rule_concepts**: People do the same Bayesian concept learning as the best model (42 weighted rules and all intervals of 1-100, size principle with tempered evidence, a decision gain and response bias turning belief into a yes/no answer), but their rules can also be bounded to a range: besides "multiples of 10" they entertain "multiples of 10 from 70 to 100", i.e. any unbroken run of a rule's members, which takes an inferred share of each rule's prior belief. Because a range-bounded rule is smaller, the size principle favours it whenever the examples cluster, so this model disagrees most sharply with the best model on rule-consistent targets far outside the examples' range (e.g. 10 after 80, 70, 90, 100, or 6 after 96, 86, 36, 66), which it predicts far fewer yeses to, while keeping rule-violating targets inside the range at near-zero.
- **round_landmark_interval_prior**: People do the same Bayesian concept learning over weighted mathematical rules and numeric ranges as the best model (size principle with tempered evidence, a decision gain and criterion turning belief into a yes/no answer), but the ranges they entertain are not all equally natural: ranges whose boundaries fall on round decade landmarks (such as 1-50, 20-29, 61-100 or "the 80s") get extra prior belief, by an inferred factor, over ranges with arbitrary edges like 23-87. Generalization from examples therefore tends to stop at, or extend out to, the nearest round decade boundary rather than tracking the examples' exact span.
- **criterion_shifted_tempered_decision**: Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, examples' evidence tempered by an inferred weight) and pass their belief through the same inferred decision gain on its log-odds, but their yes/no criterion is not centred on even odds. The one change, taken from response_bias_concept_learning, is an inferred response bias that shifts the decision criterion by a constant on the log-odds scale, so the gain decides how sharply people separate strong from weak beliefs while the bias decides whether they lean toward "yes" or "no" — e.g. a liberal criterion gives more yeses to targets with middling or weak belief without making yeses to the examples themselves less likely.
- **stochastic_example_forgetting**: People do the same Bayesian concept learning as the best model (weighted rules and all intervals of 1-100, size principle with tempered evidence, and a decision gain and criterion turning belief into a yes/no answer), but their memory of the examples is unreliable: by the time they judge a target, each example has been retained only with an inferred probability, and they infer the concept from whichever examples they still hold. Their belief is therefore an average over the possible remembered subsets of the set, so larger sets are sometimes judged as if from fewer examples (broader generalization) and an odd-one-out example is sometimes forgotten, letting the tight rule the other examples share (e.g. "ends in 2" after 92, 82, 27, 42) survive without being explained away as noise.
- **redundant_examples_decision_gain**: Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and turn their belief into a yes/no answer through the same inferred decision gain on its log-odds, but they treat additional examples as partly redundant rather than as independent draws. The one change, taken from redundant_examples_tempered_concepts, is that the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it: with sublinear growth, a single example already supports a fairly tight concept while three or four examples sharpen generalization much less than strong sampling predicts (an exponent of one recovers the incumbent).
- **win_stay_lose_shift_criterion_shift**: Refinement of win_stay_lose_shift_decision_gain: people still hold a single running guess about the concept (drawn from the tempered Bayesian posterior over weighted rules and intervals after the first example, kept while later examples fit it, redrawn from the current posterior when an example contradicts it) and pass their belief that this tracked concept contains the target through an inferred decision gain on its log-odds. The one change, taken from the incumbent criterion_shifted_tempered_decision, is an inferred response bias that shifts the yes/no criterion away from even odds by a constant on the log-odds scale, so a liberal (or conservative) criterion raises (or lowers) yeses to targets with middling or weak belief under the order-dependent guess without much changing yeses to the examples themselves.
- **split_rule_range_tempering**: Refinement of criterion_shifted_tempered_decision: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and answer through the same decision gain and shifted criterion, but they do not weigh the examples' evidence equally strongly for the two kinds of concept. The one change is that the size principle is tempered separately, by its own inferred weight, for mathematical rules and for numeric ranges: people may treat examples as strong, informative samples of an arithmetic rule (locking onto "powers of 2" after 16, 8, 64) while treating them as only weak evidence about where a magnitude range starts and stops (so range-based generalization stays broad), or vice versa; equal weights recover the incumbent.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| criterion_shifted_redundant_examples ←selected | 0.00 | 0.00 | — (best) | 0.426 | yes |
| range_bounded_rule_concepts | 0.76 | 1.21 | no (within ~2·dse) | 0.199 | yes |
| round_landmark_interval_prior | 0.85 | 1.04 | no (within ~2·dse) | 0.182 | yes |
| criterion_shifted_tempered_decision | 1.17 | 1.03 | no (within ~2·dse) | 0.132 | yes |
| stochastic_example_forgetting | 1.98 | 1.59 | no (within ~2·dse) | 0.059 | yes |
| redundant_examples_decision_gain | 6.07 | 4.67 | no (within ~2·dse) | 0.001 | yes |
| win_stay_lose_shift_criterion_shift | 6.77 | 5.91 | no (within ~2·dse) | 0.000 | yes |
| split_rule_range_tempering | 6.97 | 4.00 | no (within ~2·dse) | 0.000 | yes |
