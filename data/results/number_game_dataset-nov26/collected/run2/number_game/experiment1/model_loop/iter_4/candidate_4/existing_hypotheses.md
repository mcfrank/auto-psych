# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## one_sided_magnitude_bound_overlap  — rank 0, the best model on this data, ELPD-LOO -1497.0

Refinement of `decimal_multiple_salience_overlap`: people still note every salient property all the examples share (each property family with its own salience, base-ten multiples their own family, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change is to the magnitude concepts people consider: instead of one two-sided "within the examples' span" property, they entertain two separate one-sided bounds — "at least as large as the smallest example" and "no larger than the largest example" — each with its own salience, so if the open-ended "this big or bigger" bound is the more salient, targets above all the examples are endorsed more readily than equally distant targets below them.

## parity_direction_salience_overlap  — rank 1, 5.6 ± 7.4 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -1502.6

Refinement of `family_salience_feature_overlap` (carried over in the form its successor `decimal_multiple_salience_overlap` established, with multiples of 5 and 10 as their own salience family): people note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one new change, grafted from the directional parity contrast of `asymmetric_decimal_contrast_similarity`, is that "all odd" and "all even" are not one parity family with a single salience: each has its own inferred salience, so a shared odd (or even) pattern can weigh far more in the judgment than the opposite parity pattern does, and breaking it costs a target correspondingly more.

## directional_span_salience_overlap_2  — rank 2, 10.0 ± 5.5 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1507.0

Refinement of `decimal_multiple_salience_overlap`: people still note every salient mathematical property the examples all share (with multiples of 5/10 as their own salience family) and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is to the "lies within the examples' span" property: instead of an all-or-none, symmetric in-span test, the span match fades gradually with how far the target lies beyond the examples, at separately inferred rates above the largest example and below the smallest one — people extend a set upward toward larger numbers more readily than downward toward smaller ones.

## directional_span_salience_overlap  — rank 3, 10.0 ± 5.4 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1507.0

Refinement of `decimal_multiple_salience_overlap`: people still note every salient mathematical property the examples all share (with multiples of 5/10 as their own salient family) and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is the "lies within the examples' range" property: instead of an all-or-none, symmetric in-span check, a target outside the span keeps partial credit that fades with its distance from the span, and it fades at a different, inferred rate above the largest example than below the smallest — people readily extend a pattern upward to larger numbers (as if the program counts up) but much less downward.

## decimal_multiple_salience_overlap  — rank 4, 11.1 ± 6.2 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1508.1

Refinement of `family_salience_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is that multiples of 5 and of 10 — the round, base-ten multiples — form their own salience family, separate from other multiples (of 3, 4, 6, 7, 8, 9, 11, 12), because decimal notation makes them far more noticeable; so when examples like 90 and 45 share both "multiple of 5" and "multiple of 9", people generalize along the multiple-of-5 property much more than along the other one.

## family_salience_feature_overlap  — rank 5, 34.5 ± 8.9 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1531.5

Refinement of `shared_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to how much of that shared profile the target also has, with no size principle. The one change is that the properties are not equally salient: each family of property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, and lying within the examples' span) gets its own inferred salience weight, so a shared feature of a highly salient kind (e.g. a common final digit) dominates the judgment while shared features of less noticed kinds barely count.

## position_weighted_feature_tally  — rank 6, 39.3 ± 9.7 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1536.3

People build up their sense of which mathematical properties matter by reading the examples in order and keeping a running tally of each property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), in which examples at certain positions count more than others — the most recent ones if memory fades, the first ones if the first example anchors the guess. They then say "yes" in proportion to how much of that order-weighted, salience-weighted property profile (plus lying within the examples' span) the target also has, so a property carried by the favoured examples but not by the rest still drives generalization, and the same examples in a different order would be generalized differently.

## violation_penalty_feature_test  — rank 7, 55.4 ± 16.6 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1552.4

Refinement of `family_salience_feature_overlap`: people still note the salient mathematical properties all the examples share (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, lying within the examples' span), each family with its own salience, and use no size principle. The one change is the combination rule: instead of a compensatory share of the profile the target matches (where matching parity or span can make up for breaking the shared final digit), each shared property the target violates independently cuts the chance of a "yes" by an amount set by its family's salience — a non-compensatory, noisy-AND test in which violating a highly salient shared property (e.g. a common final digit) is close to disqualifying no matter what else the target shares.

## evidence_weighted_concept_learning  — rank 8, 70.2 ± 18.4 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1567.2

People do Bayesian concept learning over mathematical rules and numeric intervals with the size principle, as in the Independent model of Bigelow & Piantadosi, but they systematically mis-weight the evidence of the examples relative to their prior (Edwards-style conservatism or over-confidence): the examples' likelihood enters their belief update raised to a single inferred power. When that power is below one, people update too little from the examples, so tight rules and narrow ranges dominate less and generalization stays broader than the normative learner's; above one, they over-commit to the tightest fitting concept.

## evidence_weighted_concepts_with_lapse  — rank 9, 71.9 ± 17.6 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1568.9

Refinement of `evidence_weighted_concept_learning`: people still do Bayesian concept learning over the same 45 mathematical rules and all intervals of 1-100 with the size principle, mis-weighting the examples' evidence by one inferred power, and usually answer "yes" with the posterior probability that the concept contains the target. The one change is that on an inferred fraction of trials people do not consult the inferred concept at all (inattention) and instead answer from a fixed, inferred bias toward "yes", so targets that clearly violate the concept still get a floor of endorsement and targets that clearly fit it are sometimes rejected, rather than near-certain all-or-none answers.

## conservative_size_principle_conjunction  — rank 10, 76.3 ± 18.6 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1573.3

Refinement of rule_within_range_conjunction: people still infer a conjunctive concept — a mathematical rule (or "any number") restricted to a range of magnitudes — choosing rule and range together by Bayesian size-principle reasoning, and say "yes" with the posterior probability that the concept contains the target. The one change is that people are conservative about the evidence: they weigh how well each candidate concept explains the examples with an inferred strength (the likelihood raised to a power below one if they are conservative), so the size principle pulls them less sharply toward the tightest rule and range, and they keep partial belief in broader concepts, such as any number within the examples' span, even when a narrow rule fits better.

## log_scale_feature_prototype  — rank 11, 121.7 ± 24.1 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1618.7

People do not infer a rule; they abstract a prototype from the examples — the average profile of mathematical properties the examples have (how many are even, odd, prime, squares, powers, multiples of each number, multiples of 5/10, end in each digit, contain each digit) together with a central magnitude on a compressed, logarithmic number line — and say "yes" in proportion to how similar the target is to that prototype, with similarity falling off with the attention-weighted mismatch in properties and the log-scale distance from the prototype's centre. Because the number line is logarithmic, a target the same numerical distance above the examples is closer to the prototype than one below them, so people extend upward more readily than downward.

## recency_weighted_example_evidence  — rank 12, 128.0 ± 24.9 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1625.0

People do Bayesian concept learning over mathematical rules and magnitude intervals (size principle), but they read the examples one at a time and memory for earlier examples fades: each example's evidential force is discounted by how many examples were read after it. Earlier examples therefore constrain the inferred concept less than the most recent one, so the same examples shown in a different order would be generalized differently, with targets resembling the last-shown example favoured.

## rule_within_range_conjunction  — rank 13, 129.1 ± 24.9 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1626.1

People infer a conjunctive concept: a mathematical rule (e.g. "even", "multiples of 7", or "any number") restricted to a range of magnitudes, and they infer both the rule and the range from the examples by Bayesian size-principle reasoning. Because the range can be tight around the examples, a target that fits the rule but lies far outside the span of the examples is judged less likely to be produced than one inside it.

## asymmetric_decimal_contrast_similarity  — rank 14, 130.9 ± 26.3 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1627.9

People judge a target by its asymmetric (Tversky-style) similarity to every example at once: similarity falls off with numerical distance and with mismatches in the decimal form — a different final digit, and the divisibility-by-2 and divisibility-by-5 properties visible in that digit — and the target must resemble all the examples (the distances to each example add up, so one badly mismatched example drags the judgment down). The comparison is directional: a property the examples have that the target lacks (e.g. examples even or multiples of 5, target not) costs a different, inferred amount than a property the target has that the examples lack, so violating an even or base-10 pattern is punished more (or less) than violating an odd pattern.

## rule_range_width_power_prior  — rank 15, 132.3 ± 25.5 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1629.3

Refinement of `rule_within_range_conjunction`: people still infer a conjunctive concept, a mathematical rule (or "any number") restricted to a range of magnitudes, inferring rule and range from the examples by Bayesian size-principle reasoning and answering "yes" with the posterior probability that the concept contains the target. The one change is the prior over the restricting range: instead of every sub-range being equally likely, a range's prior weight scales as a power of its width (exponent inferred), so people may come in expecting narrow ranges hugging the examples (generalizing less beyond them) or broad ones (generalizing further) — the width preference, not just the examples, governs how far a rule is extended outside the examples' span.

## shared_feature_overlap  — rank 16, 135.5 ± 19.3 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1632.6

People do not weigh hypotheses by how tightly they fit the examples (no size principle); instead they note every salient mathematical property the examples all share (parity, prime, square, cube, powers, multiples, final digit, contained digit, and lying within the examples' span) and say "yes" in proportion to the fraction of those shared properties the target also has. This disagrees most sharply with the Bayesian rule-learning model where the examples share both a broad and a narrow property (e.g. 16,32,2,8 are all even and all powers of 2): the Bayesian model concentrates on the narrow rule and rejects merely-even targets, whereas feature-overlap gives such targets graded, intermediate endorsement, and it gives far out-of-range rule-matching targets less endorsement because they fail the shared "within the span" property.

## interval_width_power_prior  — rank 17, 149.4 ± 27.7 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1646.4

A refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 45 rules (each with its own inferred prior weight, lambda_0 for the rules as a group) and all intervals of 1-100, with the size principle and 10% uniform example noise, answering "yes" with the posterior probability that the concept contains the target. The one change is that intervals are no longer equally likely a priori: an interval's prior weight scales as a power of its width (exponent inferred), so people can prefer narrow, tight ranges around the examples (or broad ones), which governs how far they generalize beyond the range of the examples.

## erlang_interval_independent_rules  — rank 18, 160.5 ± 29.9 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1657.5

Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules, each with its own inferred prior weight, plus intervals of 1-100, sampling examples from the concept (size principle, 10% noise) and saying "yes" with the posterior probability that the concept contains the target. The one change is the interval prior: instead of every interval being equally likely a priori, the prior over intervals follows an Erlang density on interval size (scale inferred, as in tenenbaum_rules_similarity), so people favour intervals of a characteristic, typically modest, width and generalize less far beyond the range of the examples.

## decimal_exemplar_similarity  — rank 19, 173.0 ± 25.4 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1670.0

People do not induce rules or concepts at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization, as in the Generalized Context Model). Numbers are represented in their decimal written form, so similarity decays exponentially with a weighted city-block distance that combines how far apart the numbers are in magnitude and whether they share their ones digit, with a learned attention weight between the two; the higher the total similarity relative to a response criterion, the more likely a "yes".

## family_weighted_rule_prior  — rank 20, 174.8 ± 33.6 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1671.8

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, "yes" = posterior probability the concept contains the target), but their prior over rules is organised by kind of rule rather than flat. The one change: the rules are grouped into seven families (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each family gets its own inferred share of the rule prior, and the rules within a family are equally likely — so a few interpretable family preferences (e.g. arithmetic multiples and powers over digit-based rules) rather than one weight per rule or none explain which rules people readily entertain.

## bigelow_piantadosi_independent  — rank 21, 179.7 ± 29.5 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1676.7

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## upward_counting_progression  — rank 22, 186.7 ± 23.0 nats behind the best (8.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1683.7

People think of the hidden program as a counter: it starts at some number and keeps adding a fixed step, producing numbers upward until it runs out of room at 100. They infer the step and the starting point from the examples (preferring starting points at or just below the smallest example and favouring counters that would make the examples likely to be hit, so more examples narrow the guess), and say "yes" with their belief that such a counter would reach the target — so a target beyond the largest example on the counter's path is readily accepted, while one below the smallest example is accepted only if the counter is thought to start that low.

## posterior_confidence_criterion  — rank 23, 220.1 ± 24.0 nats behind the best (9.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1717.1

People infer the program's concept exactly as the simplest Bayesian rule-and-interval learner does (equally likely rules, size principle), but they do not answer "yes" with the probability they believe the target belongs (probability matching); instead they compare their confidence to an internal criterion and answer "yes" only when it is high enough, with a soft, inferred-steepness threshold. This sharpens fairly-likely targets toward "yes" and doubtful ones toward "no", so sets differ more starkly in overall endorsement and targets violating the favoured rule are rejected more firmly than graded posterior beliefs would imply.

## most_specific_rule_winner_take_all  — rank 24, 241.9 ± 24.0 nats behind the best (10.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1738.9

People do not weigh many hypotheses at once; they settle on a single concept, the most specific (smallest) mathematical rule that every example satisfies (e.g. "powers of 2" rather than "even" for 16, 32, 2, 8), and answer "yes" essentially only to targets that obey that one rule, with occasional noisy answers. Only when no mathematical rule fits all the examples do they fall back on the range spanned by the examples. This predicts all-or-none generalization: merely-even targets for a powers-of-2 set, and out-of-range rule-matching targets, get no graded endorsement beyond noise.

## bigelow_piantadosi_mixture  — rank 25, 318.1 ± 35.9 nats behind the best (8.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1815.1

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 26, 339.9 ± 44.3 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1837.0

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
