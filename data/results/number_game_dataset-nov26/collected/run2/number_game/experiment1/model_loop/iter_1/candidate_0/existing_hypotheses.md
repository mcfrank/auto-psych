# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## rule_within_range_conjunction  — rank 0, the best model on this data, ELPD-LOO -1626.1

People infer a conjunctive concept: a mathematical rule (e.g. "even", "multiples of 7", or "any number") restricted to a range of magnitudes, and they infer both the rule and the range from the examples by Bayesian size-principle reasoning. Because the range can be tight around the examples, a target that fits the rule but lies far outside the span of the examples is judged less likely to be produced than one inside it.

## shared_feature_overlap  — rank 1, 6.5 ± 33.5 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -1632.6

People do not weigh hypotheses by how tightly they fit the examples (no size principle); instead they note every salient mathematical property the examples all share (parity, prime, square, cube, powers, multiples, final digit, contained digit, and lying within the examples' span) and say "yes" in proportion to the fraction of those shared properties the target also has. This disagrees most sharply with the Bayesian rule-learning model where the examples share both a broad and a narrow property (e.g. 16,32,2,8 are all even and all powers of 2): the Bayesian model concentrates on the narrow rule and rejects merely-even targets, whereas feature-overlap gives such targets graded, intermediate endorsement, and it gives far out-of-range rule-matching targets less endorsement because they fail the shared "within the span" property.

## interval_width_power_prior  — rank 2, 20.3 ± 18.6 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1646.4

A refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 45 rules (each with its own inferred prior weight, lambda_0 for the rules as a group) and all intervals of 1-100, with the size principle and 10% uniform example noise, answering "yes" with the posterior probability that the concept contains the target. The one change is that intervals are no longer equally likely a priori: an interval's prior weight scales as a power of its width (exponent inferred), so people can prefer narrow, tight ranges around the examples (or broad ones), which governs how far they generalize beyond the range of the examples.

## erlang_interval_independent_rules  — rank 3, 31.4 ± 19.6 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -1657.5

Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules, each with its own inferred prior weight, plus intervals of 1-100, sampling examples from the concept (size principle, 10% noise) and saying "yes" with the posterior probability that the concept contains the target. The one change is the interval prior: instead of every interval being equally likely a priori, the prior over intervals follows an Erlang density on interval size (scale inferred, as in tenenbaum_rules_similarity), so people favour intervals of a characteristic, typically modest, width and generalize less far beyond the range of the examples.

## decimal_exemplar_similarity  — rank 4, 44.0 ± 31.1 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1670.0

People do not induce rules or concepts at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization, as in the Generalized Context Model). Numbers are represented in their decimal written form, so similarity decays exponentially with a weighted city-block distance that combines how far apart the numbers are in magnitude and whether they share their ones digit, with a learned attention weight between the two; the higher the total similarity relative to a response criterion, the more likely a "yes".

## family_weighted_rule_prior  — rank 5, 45.8 ± 38.2 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1671.8

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, "yes" = posterior probability the concept contains the target), but their prior over rules is organised by kind of rule rather than flat. The one change: the rules are grouped into seven families (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each family gets its own inferred share of the rule prior, and the rules within a family are equally likely — so a few interpretable family preferences (e.g. arithmetic multiples and powers over digit-based rules) rather than one weight per rule or none explain which rules people readily entertain.

## bigelow_piantadosi_independent  — rank 6, 50.6 ± 17.7 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1676.7

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## bigelow_piantadosi_mixture  — rank 7, 189.0 ± 41.3 nats behind the best (4.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1815.1

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 8, 210.9 ± 49.9 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1837.0

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
