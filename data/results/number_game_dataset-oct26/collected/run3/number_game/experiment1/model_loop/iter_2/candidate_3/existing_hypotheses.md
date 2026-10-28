# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## sample_size_growing_evidence_concepts  — rank 0, the best model on this data, ELPD-LOO -1603.4

Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, but how much weight each example's evidence receives grows with the number of examples shown, because a single example gives little reason to believe it was sampled from a narrow concept, while several examples that all fit a tight concept convince people the examples are "strong" samples. The one change is that the evidence-weighting strength is a power function of the set size (an inferred growth exponent) rather than one constant, which predicts broader generalization from a single example than the incumbent, the discrepancy its critique flags.

## iter1_candidate4  — rank 1, 7.3 ± 6.4 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1610.7

Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and every interval of 1-100 (inferred rule weights, a Beta midpoint prior on intervals, size-principle sampling with 10% noise, examples' evidence weighted by an inferred strength), but they also hold a prior preference for intervals of intermediate size (Tenenbaum's Erlang size prior, scale inferred, grafted from `bp_rules_erlang_interval_size`) that multiplies the midpoint prior. The one change is this size prior, which discounts tiny near-singleton intervals a priori and so predicts broader generalization from a single example — where the incumbent says "yes" too rarely — while leaving multi-example sets, whose evidence already rules out tiny intervals, largely unchanged.

## first_example_weak_sampling_concepts  — rank 2, 13.1 ± 9.8 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -1616.5

People update their concept one example at a time, and the size principle only kicks in once there is a coincidence to explain: the first example merely rules out concepts that do not contain it (any consistent concept stays as plausible as its prior says), while every later example is treated as drawn from the concept, so small concepts that keep fitting become favoured ("suspicious coincidence"). This predicts broad generalization from a single example and a sharp tightening only when a second consistent example arrives.

## conservative_evidence_interval_concepts  — rank 3, 15.1 ± 6.9 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1618.5

Refinement of `bigelow_piantadosi_interval`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior weight inferred for each rule, an interval prior given by a Beta density over its midpoint, examples sampled from the concept by the size principle with 10% noise), but they are conservative updaters: they weight the evidence of the examples by a single inferred strength rather than at full Bayesian value, so the posterior over concepts is flatter (strength below 1) or sharper (above 1) than the ideal learner's. The one change is this evidence-weighting exponent on the examples' likelihood, which governs how quickly generalization tightens around the narrowest consistent concept as examples accumulate.

## size_principle_strength_intervals  — rank 4, 42.0 ± 11.0 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1645.4

Refinement of bigelow_piantadosi_interval: people use the same Bayesian concept learning over the 42 rules and the midpoint-weighted intervals of 1-100, but they apply the size principle only partially — how strongly a small concept is favoured for fitting the examples tightly is a graded, inferred strength, between weak sampling (examples merely consistent with a concept, no preference for small concepts) and the full strong sampling the incumbent assumes. The one change is this inferred size-principle strength, which predicts broader generalization than the incumbent (e.g. more "yes" to other even numbers after 16, 8, 64) if people underweight concept size.

## shared_property_overlap  — rank 5, 52.9 ± 18.8 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1656.4

People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.

## exemplar_feature_similarity_gcm  — rank 6, 61.6 ± 21.7 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1665.0

People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

## range_scaled_magnitude_gradient  — rank 7, 119.4 ± 25.1 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1722.8

People do not consider arithmetic rules or properties at all; they judge a target purely by its closeness on the number line to the nearest example, with a generalization gradient whose width stretches with the spread of the examples. A single example (no spread) still supports a fixed baseline width of generalization around it, so "yes" falls off smoothly with numerical distance, more slowly when the examples cover a wide range.

## bp_rules_erlang_interval_size  — rank 8, 171.1 ± 29.3 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1774.6

Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 rules (each with its own inferred prior weight) and every interval of 1-100, with examples sampled from the concept (the size principle, 10% noise), and answer "yes" with the posterior probability that the concept contains the target. The one change is that intervals are not equally likely a priori: people prefer intervals of intermediate size, with an Erlang prior on an interval's size whose scale is inferred (Tenenbaum's size prior grafted onto the independent-rule-weight model), so a few nearby examples generalize to a moderately wide range rather than to the very tightest or the very widest interval.

## bigelow_piantadosi_interval  — rank 9, 190.7 ± 28.7 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1794.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## bigelow_piantadosi_independent  — rank 10, 196.7 ± 32.2 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1800.2

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## tenenbaum_rules_similarity  — rank 11, 283.6 ± 46.4 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1887.0

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
