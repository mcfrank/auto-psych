# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## lapsing_growing_evidence_concepts  — rank 0, the best model on this data, ELPD-LOO -1598.7

Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but on some trials they answer without consulting their concept at all, guessing "yes" or "no" with their own inferred rates. The one change is this attention-lapse response floor and ceiling (an inferred chance of a blind "yes" and a separate chance of a blind "no"), which predicts a steady baseline of "yes" to targets the concept clearly excludes and occasional "no" to targets it clearly contains, where the incumbent puts both at near-certainty.

## sample_size_growing_evidence_concepts  — rank 1, 4.7 ± 5.0 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1603.4

Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, but how much weight each example's evidence receives grows with the number of examples shown, because a single example gives little reason to believe it was sampled from a narrow concept, while several examples that all fit a tight concept convince people the examples are "strong" samples. The one change is that the evidence-weighting strength is a power function of the set size (an inferred growth exponent) rather than one constant, which predicts broader generalization from a single example than the incumbent, the discrepancy its critique flags.

## soft_criterion_concept_decision  — rank 2, 11.5 ± 10.0 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1610.2

People infer the hidden concept by plain Bayesian concept learning over arithmetic rules and numeric intervals (the simplest current inference, with the size principle), but they do not report that belief by probability matching: they answer "yes" when their confidence that the target belongs clears a personal criterion, applied noisily. How high the criterion sits (a bias toward "yes" or "no") and how sharply confidence is turned into a choice are the only changes, so moderately believed targets are pushed toward one answer and near-certain beliefs are softened toward the other.

## iter1_candidate4  — rank 3, 12.0 ± 7.1 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -1610.7

Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and every interval of 1-100 (inferred rule weights, a Beta midpoint prior on intervals, size-principle sampling with 10% noise, examples' evidence weighted by an inferred strength), but they also hold a prior preference for intervals of intermediate size (Tenenbaum's Erlang size prior, scale inferred, grafted from `bp_rules_erlang_interval_size`) that multiplies the midpoint prior. The one change is this size prior, which discounts tiny near-singleton intervals a priori and so predicts broader generalization from a single example — where the incumbent says "yes" too rarely — while leaving multi-example sets, whose evidence already rules out tiny intervals, largely unchanged.

## first_example_weak_erlang_concepts  — rank 4, 14.9 ± 8.5 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1613.6

Refinement of `first_example_weak_sampling_concepts`: people update their concept one example at a time, and the first example only rules out concepts that do not contain it, while every later example is treated as drawn from the concept (size principle, weighted by an inferred strength) — but people also hold a prior preference for number ranges of intermediate size (Tenenbaum's Erlang size prior on intervals, scale inferred, grafted from `iter1_candidate4`). The one change is this size prior: because a single example carries no size-principle evidence under this account, what people generalize to from one example is set entirely by their prior, and the size prior predicts a moderately wide range around a lone example rather than the broad ranges near 1-100 that a size-blind prior favours.

## first_example_weak_sampling_concepts  — rank 5, 17.8 ± 8.9 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1616.5

People update their concept one example at a time, and the size principle only kicks in once there is a coincidence to explain: the first example merely rules out concepts that do not contain it (any consistent concept stays as plausible as its prior says), while every later example is treated as drawn from the concept, so small concepts that keep fitting become favoured ("suspicious coincidence"). This predicts broad generalization from a single example and a sharp tightening only when a second consistent example arrives.

## conservative_evidence_interval_concepts  — rank 6, 19.8 ± 7.8 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1618.5

Refinement of `bigelow_piantadosi_interval`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior weight inferred for each rule, an interval prior given by a Beta density over its midpoint, examples sampled from the concept by the size principle with 10% noise), but they are conservative updaters: they weight the evidence of the examples by a single inferred strength rather than at full Bayesian value, so the posterior over concepts is flatter (strength below 1) or sharper (above 1) than the ideal learner's. The one change is this evidence-weighting exponent on the examples' likelihood, which governs how quickly generalization tightens around the narrowest consistent concept as examples accumulate.

## recency_decayed_evidence_concepts  — rank 7, 24.8 ± 8.1 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1623.5

People learn the program's concept by conservative Bayesian concept learning over arithmetic rules and numeric intervals (size principle), but their memory of the examples fades with how long ago each was shown: the last example counts as full evidence, and each example further back in the sequence counts as progressively weaker evidence, by a constant inferred retention factor per position. Generalization is therefore driven mostly by the most recent examples, so reordering the same examples changes which targets get a "yes" (targets that fit only the first example are endorsed less often than targets that fit only the last).

## size_principle_strength_intervals  — rank 8, 46.7 ± 11.5 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1645.4

Refinement of bigelow_piantadosi_interval: people use the same Bayesian concept learning over the 42 rules and the midpoint-weighted intervals of 1-100, but they apply the size principle only partially — how strongly a small concept is favoured for fitting the examples tightly is a graded, inferred strength, between weak sampling (examples merely consistent with a concept, no preference for small concepts) and the full strong sampling the incumbent assumes. The one change is this inferred size-principle strength, which predicts broader generalization than the incumbent (e.g. more "yes" to other even numbers after 16, 8, 64) if people underweight concept size.

## digit_place_similarity_gradient  — rank 9, 48.7 ± 18.9 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1647.4

People judge a target by its similarity to the examples in their written decimal form rather than on the number line: they compare the target with each example place by place, and every mismatch in the units digit or in the tens digit multiplies similarity down by its own inferred factor. The more similar the target is on average to the examples under this digit-place metric, the more likely people are to say "yes" — so 21 after 91, or 58 after 28 and 98, is endorsed despite being numerically far away, while 88 after 89 is rejected despite being adjacent.

## shared_property_overlap  — rank 10, 57.7 ± 15.7 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1656.4

People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.

## exemplar_feature_similarity_gcm  — rank 11, 66.3 ± 19.1 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1665.0

People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

## range_scaled_magnitude_gradient  — rank 12, 124.1 ± 22.2 nats behind the best (5.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1722.8

People do not consider arithmetic rules or properties at all; they judge a target purely by its closeness on the number line to the nearest example, with a generalization gradient whose width stretches with the spread of the examples. A single example (no spread) still supports a fixed baseline width of generalization around it, so "yes" falls off smoothly with numerical distance, more slowly when the examples cover a wide range.

## bp_rules_erlang_interval_size  — rank 13, 175.9 ± 30.3 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1774.6

Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 rules (each with its own inferred prior weight) and every interval of 1-100, with examples sampled from the concept (the size principle, 10% noise), and answer "yes" with the posterior probability that the concept contains the target. The one change is that intervals are not equally likely a priori: people prefer intervals of intermediate size, with an Erlang prior on an interval's size whose scale is inferred (Tenenbaum's size prior grafted onto the independent-rule-weight model), so a few nearby examples generalize to a moderately wide range rather than to the very tightest or the very widest interval.

## bigelow_piantadosi_interval  — rank 14, 195.4 ± 29.9 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1794.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## bigelow_piantadosi_independent  — rank 15, 201.5 ± 33.3 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1800.2

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## tenenbaum_rules_similarity  — rank 16, 288.3 ± 46.3 nats behind the best (6.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1887.0

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
