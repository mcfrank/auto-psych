# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## family_salience_feature_overlap  — rank 0, the best model on this data, ELPD-LOO -1531.5

Refinement of `shared_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to how much of that shared profile the target also has, with no size principle. The one change is that the properties are not equally salient: each family of property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, and lying within the examples' span) gets its own inferred salience weight, so a shared feature of a highly salient kind (e.g. a common final digit) dominates the judgment while shared features of less noticed kinds barely count.

## evidence_weighted_concept_learning  — rank 1, 35.7 ± 19.7 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1567.2

People do Bayesian concept learning over mathematical rules and numeric intervals with the size principle, as in the Independent model of Bigelow & Piantadosi, but they systematically mis-weight the evidence of the examples relative to their prior (Edwards-style conservatism or over-confidence): the examples' likelihood enters their belief update raised to a single inferred power. When that power is below one, people update too little from the examples, so tight rules and narrow ranges dominate less and generalization stays broader than the normative learner's; above one, they over-commit to the tightest fitting concept.

## conservative_size_principle_conjunction  — rank 2, 41.7 ± 20.2 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1573.3

Refinement of rule_within_range_conjunction: people still infer a conjunctive concept — a mathematical rule (or "any number") restricted to a range of magnitudes — choosing rule and range together by Bayesian size-principle reasoning, and say "yes" with the posterior probability that the concept contains the target. The one change is that people are conservative about the evidence: they weigh how well each candidate concept explains the examples with an inferred strength (the likelihood raised to a power below one if they are conservative), so the size principle pulls them less sharply toward the tightest rule and range, and they keep partial belief in broader concepts, such as any number within the examples' span, even when a narrow rule fits better.

## recency_weighted_example_evidence  — rank 3, 93.5 ± 23.0 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1625.0

People do Bayesian concept learning over mathematical rules and magnitude intervals (size principle), but they read the examples one at a time and memory for earlier examples fades: each example's evidential force is discounted by how many examples were read after it. Earlier examples therefore constrain the inferred concept less than the most recent one, so the same examples shown in a different order would be generalized differently, with targets resembling the last-shown example favoured.

## rule_within_range_conjunction  — rank 4, 94.6 ± 27.4 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1626.1

People infer a conjunctive concept: a mathematical rule (e.g. "even", "multiples of 7", or "any number") restricted to a range of magnitudes, and they infer both the rule and the range from the examples by Bayesian size-principle reasoning. Because the range can be tight around the examples, a target that fits the rule but lies far outside the span of the examples is judged less likely to be produced than one inside it.

## rule_range_width_power_prior  — rank 5, 97.8 ± 27.9 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1629.3

Refinement of `rule_within_range_conjunction`: people still infer a conjunctive concept, a mathematical rule (or "any number") restricted to a range of magnitudes, inferring rule and range from the examples by Bayesian size-principle reasoning and answering "yes" with the posterior probability that the concept contains the target. The one change is the prior over the restricting range: instead of every sub-range being equally likely, a range's prior weight scales as a power of its width (exponent inferred), so people may come in expecting narrow ranges hugging the examples (generalizing less beyond them) or broad ones (generalizing further) — the width preference, not just the examples, governs how far a rule is extended outside the examples' span.

## shared_feature_overlap  — rank 6, 101.0 ± 16.7 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1632.6

People do not weigh hypotheses by how tightly they fit the examples (no size principle); instead they note every salient mathematical property the examples all share (parity, prime, square, cube, powers, multiples, final digit, contained digit, and lying within the examples' span) and say "yes" in proportion to the fraction of those shared properties the target also has. This disagrees most sharply with the Bayesian rule-learning model where the examples share both a broad and a narrow property (e.g. 16,32,2,8 are all even and all powers of 2): the Bayesian model concentrates on the narrow rule and rejects merely-even targets, whereas feature-overlap gives such targets graded, intermediate endorsement, and it gives far out-of-range rule-matching targets less endorsement because they fail the shared "within the span" property.

## interval_width_power_prior  — rank 7, 114.9 ± 28.8 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1646.4

A refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 45 rules (each with its own inferred prior weight, lambda_0 for the rules as a group) and all intervals of 1-100, with the size principle and 10% uniform example noise, answering "yes" with the posterior probability that the concept contains the target. The one change is that intervals are no longer equally likely a priori: an interval's prior weight scales as a power of its width (exponent inferred), so people can prefer narrow, tight ranges around the examples (or broad ones), which governs how far they generalize beyond the range of the examples.

## erlang_interval_independent_rules  — rank 8, 126.0 ± 31.2 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1657.5

Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules, each with its own inferred prior weight, plus intervals of 1-100, sampling examples from the concept (size principle, 10% noise) and saying "yes" with the posterior probability that the concept contains the target. The one change is the interval prior: instead of every interval being equally likely a priori, the prior over intervals follows an Erlang density on interval size (scale inferred, as in tenenbaum_rules_similarity), so people favour intervals of a characteristic, typically modest, width and generalize less far beyond the range of the examples.

## decimal_exemplar_similarity  — rank 9, 138.5 ± 25.9 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1670.0

People do not induce rules or concepts at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization, as in the Generalized Context Model). Numbers are represented in their decimal written form, so similarity decays exponentially with a weighted city-block distance that combines how far apart the numbers are in magnitude and whether they share their ones digit, with a learned attention weight between the two; the higher the total similarity relative to a response criterion, the more likely a "yes".

## family_weighted_rule_prior  — rank 10, 140.3 ± 31.4 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1671.8

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, "yes" = posterior probability the concept contains the target), but their prior over rules is organised by kind of rule rather than flat. The one change: the rules are grouped into seven families (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each family gets its own inferred share of the rule prior, and the rules within a family are equally likely — so a few interpretable family preferences (e.g. arithmetic multiples and powers over digit-based rules) rather than one weight per rule or none explain which rules people readily entertain.

## bigelow_piantadosi_independent  — rank 11, 145.2 ± 30.4 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1676.7

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## most_specific_rule_winner_take_all  — rank 12, 207.4 ± 22.1 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1738.9

People do not weigh many hypotheses at once; they settle on a single concept, the most specific (smallest) mathematical rule that every example satisfies (e.g. "powers of 2" rather than "even" for 16, 32, 2, 8), and answer "yes" essentially only to targets that obey that one rule, with occasional noisy answers. Only when no mathematical rule fits all the examples do they fall back on the range spanned by the examples. This predicts all-or-none generalization: merely-even targets for a powers-of-2 set, and out-of-range rule-matching targets, get no graded endorsement beyond noise.

## bigelow_piantadosi_mixture  — rank 13, 283.5 ± 34.2 nats behind the best (8.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1815.1

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 14, 305.4 ± 42.6 nats behind the best (7.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1837.0

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## position_weighted_feature_tally  — no comparison row

People build up their sense of which mathematical properties matter by reading the examples in order and keeping a running tally of each property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), in which examples at certain positions count more than others — the most recent ones if memory fades, the first ones if the first example anchors the guess. They then say "yes" in proportion to how much of that order-weighted, salience-weighted property profile (plus lying within the examples' span) the target also has, so a property carried by the favoured examples but not by the rest still drives generalization, and the same examples in a different order would be generalized differently.

## asymmetric_decimal_contrast_similarity  — no comparison row

People judge a target by its asymmetric (Tversky-style) similarity to every example at once: similarity falls off with numerical distance and with mismatches in the decimal form — a different final digit, and the divisibility-by-2 and divisibility-by-5 properties visible in that digit — and the target must resemble all the examples (the distances to each example add up, so one badly mismatched example drags the judgment down). The comparison is directional: a property the examples have that the target lacks (e.g. examples even or multiples of 5, target not) costs a different, inferred amount than a property the target has that the examples lack, so violating an even or base-10 pattern is punished more (or less) than violating an odd pattern.

## posterior_confidence_criterion  — no comparison row

People infer the program's concept exactly as the simplest Bayesian rule-and-interval learner does (equally likely rules, size principle), but they do not answer "yes" with the probability they believe the target belongs (probability matching); instead they compare their confidence to an internal criterion and answer "yes" only when it is high enough, with a soft, inferred-steepness threshold. This sharpens fairly-likely targets toward "yes" and doubtful ones toward "no", so sets differ more starkly in overall endorsement and targets violating the favoured rule are rejected more firmly than graded posterior beliefs would imply.

## decimal_multiple_salience_overlap  — no comparison row

Refinement of `family_salience_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is that multiples of 5 and of 10 — the round, base-ten multiples — form their own salience family, separate from other multiples (of 3, 4, 6, 7, 8, 9, 11, 12), because decimal notation makes them far more noticeable; so when examples like 90 and 45 share both "multiple of 5" and "multiple of 9", people generalize along the multiple-of-5 property much more than along the other one.
