# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## parity_restricted_concepts  — rank 0, the best model on this data, ELPD-LOO -1553.4

Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but parity is a basic feature they can conjoin with any concept, so they also entertain the odd-only and even-only version of each rule and interval (e.g. "odd multiples of 3", "even numbers between 4 and 96"), which together receive an inferred share of prior belief. The one change is adding these parity-restricted concepts (share 0 recovers the incumbent); because such a concept is half the size, the size principle favours it whenever all examples share a parity, so people reject opposite-parity targets — including far, weakly supported ones outside the examples' range — more firmly than the incumbent predicts.

## serial_position_weighted_evidence  — rank 1, 1.0 ± 9.6 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -1554.5

Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but they do not weigh every example equally as evidence — how much an example counts depends on its serial position in the list, with evidence weight shifting smoothly toward the first examples (primacy) or the last ones (recency) by an inferred serial-position gradient, while the total evidence strength stays the same. The one change is this order-dependent weighting (a zero gradient recovers the incumbent); it predicts that the same examples shown in a different order yield different generalization, and in particular that an example that breaks the pattern of the others (e.g. an odd number among evens) undermines the tight concept more when it sits at the favoured end of the list.

## iter2_candidate1  — rank 2, 40.9 ± 26.2 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -1594.3

People judge a target by Shepard-style exponential generalization in a psychological space whose dimensions are weighted by selective attention, and the examples set that attention: they attend to the properties the examples agree on (more strongly the more consistently the examples share a property and the more distinctive it is), and to magnitude in proportion to how tightly the examples cluster. The probability of "yes" falls off exponentially with the target's attention-weighted distance from the examples, so missing a consistently shared, distinctive property (e.g. the common last digit) or lying well outside a tight example range sharply lowers endorsement, while properties the examples do not agree on barely matter.

## tempered_size_principle_independent  — rank 3, 46.3 ± 10.5 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1599.7

Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over 45 rules (each with its own prior weight) and all intervals, but they under-trust the size principle — they treat the examples as weaker evidence than a fully rational learner would, as if the likelihood of the examples were tempered by an inferred exponent (sampling-assumption strength), so their posterior is less concentrated on the smallest consistent rule. The one change is this free evidence-strength exponent (1 recovers the original model); it predicts lower endorsement of minimal-rule members and graded endorsement among them, because members also covered by competing concepts (nearby intervals, other rules) keep more support.

## sampled_hypothesis_majority_vote  — rank 4, 49.2 ± 11.0 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1602.7

Refinement of tempered_size_principle_independent: people do the same tempered Bayesian concept learning over the 45 rules and all intervals, but they cannot hold the whole posterior in mind — they bring a small number of candidate concepts to mind by sampling them from their posterior and answer "yes" if the majority of those sampled concepts contain the target. The one change is this limited-sample majority vote (with an inferred number of samples; one sample recovers the original model), which makes answers more all-or-none than the posterior itself: likely members are endorsed more consistently and weakly supported targets, such as numbers beyond the range of the examples, are endorsed less.

## sampled_concepts_criterion_vote  — rank 5, 51.8 ± 11.4 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1605.2

Refinement of `sampled_hypothesis_majority_vote`: people do the same tempered Bayesian concept learning over the 45 individually weighted rules and all intervals, and answer by bringing a few candidate concepts to mind by sampling them from their posterior — but they say "yes" only if a criterion number of the sampled concepts contains the target, and that criterion is inferred rather than fixed at a simple majority (equivalently, they set how many sampled concepts must favour the target and how many may go against it). The one change is this free decision criterion (equal counts recover majority vote, a single sample recovers the incumbent); a demanding criterion predicts that targets supported only by a minority of concepts — such as numbers far outside the examples' range that fit no small rule — are rejected more firmly than majority vote allows, while targets in the dominant concept are still endorsed.

## compact_interval_tempered_bayes  — rank 6, 55.9 ± 13.1 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1609.4

Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 rules (each with its own prior weight) and all intervals of 1-100, but they do not consider all intervals equally plausible a priori — they prefer compact magnitude ranges, with an Erlang prior over interval size whose scale is inferred (the one change, taken from `tenenbaum_rules_similarity`). Because the tempered evidence no longer spreads interval belief onto wide ranges, people confine magnitude-based generalization close to the [min, max] range of the examples and endorse targets outside it less than the incumbent predicts.

## hierarchical_tempered_evidence  — rank 7, 60.1 ± 12.5 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1613.5

Refinement of `tempered_size_principle_independent`: people under-trust the examples only when deciding *which kind* of concept and which rule produced them, but once they entertain a magnitude range they apply the size principle at full strength, so the range they imagine stays tight around the examples. The one change is where the inferred evidence-strength exponent applies: it tempers the evidence for each rule and for the interval family as a whole, while belief among intervals follows the untempered size principle; this predicts firmer rejection of targets far outside the examples' range that no small rule covers, while keeping graded endorsement of rule members.

## size_penalized_interval_prior  — rank 8, 62.1 ± 13.2 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1615.6

Refinement of `tempered_size_principle_independent`: people do the same tempered Bayesian concept learning over 45 individually weighted rules and all intervals of 1-100, but a priori they expect a magnitude-based concept to be a narrow range rather than treating all intervals as equally likely — their prior on an interval falls off as a power of its width, with the steepness inferred. The one change is this size-penalizing interval prior (steepness 0 recovers a uniform interval prior); it predicts that people confine generalization more tightly to the range spanned by the examples, and endorse targets outside that range less, than the incumbent allows.

## conservative_evidence_concept_learning  — rank 9, 83.2 ± 23.7 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1636.6

Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they under-weight the evidence of the examples, as if each example counted for only a fraction of a full observation (a conservative, tempered likelihood). The one change is a free evidence-weight exponent on the likelihood of the examples, so confidence in the smallest consistent rule grows more slowly with examples and more posterior stays on broader rules and intervals — making learners less sure of the winning rule than the ideal learner.

## domain_specific_evidence_weight  — rank 10, 89.5 ± 24.1 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1642.9

Refinement of `conservative_evidence_concept_learning`: people do the same Bayesian concept learning over the 45 equally weighted rules and all intervals of 1-100 with the size principle, but how strongly they weigh the examples as evidence depends on the kind of concept — they are conservative about mathematical rules yet apply the size principle sharply to magnitude ranges, so once they think "a range of numbers" they confine it tightly to the span of the examples. The one change is splitting the single evidence-weight exponent into one for rule hypotheses and one for interval hypotheses; it predicts less endorsement of targets just outside the examples' [min, max] range (relative to inside it) than a single shared tempering allows, while keeping graded endorsement of rule members.

## log_odds_belief_readout  — rank 11, 95.8 ± 23.5 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1649.3

Refinement of bigelow_piantadosi_mixture: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they do not answer "yes" with probability equal to their belief that the concept contains the target; instead they read out that belief through a distorted, linear-in-log-odds probability weighting (as in judgments of probability and decisions under risk), which compresses confident beliefs toward the middle and shifts them by a response bias. The one change is this readout: confident beliefs that a minimal-rule member is in the concept are endorsed less than certainly, and confident beliefs that a number is out are rejected less than certainly.

## belief_criterion_threshold  — rank 12, 99.8 ± 23.9 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1653.2

People do the simplest Bayesian concept learning (45 equally likely rules and all intervals of 1-100, with the size principle), but they do not answer "yes" with probability equal to their belief; instead they compare their belief that the target is in the concept against an internal criterion of how sure they must be, and say "yes" only when the belief exceeds that criterion, with the comparison blurred by Gaussian decision noise (criterion and noise inferred). Because the criterion acts on the probability itself, targets with middling belief — such as far-away numbers supported only by broad rules or wide intervals — are rejected more firmly than belief alone predicts, while targets the learner is fairly sure of are endorsed nearly always.

## feature_overlap_similarity  — rank 13, 110.2 ± 28.3 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1663.7

People do not infer a single hidden rule; instead they judge the target by its featural similarity to the examples they were shown. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, and which decade it lies in), and the more properties the target shares with each example on average (feature overlap relative to the properties either has), the more likely people say yes — so endorsement is graded even among numbers that fit a rule, and partial matches such as a shared final digit or nearby magnitude still earn some yeses.

## featural_exemplar_similarity  — rank 14, 113.4 ± 28.1 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1666.8

People do not pick a single best-fitting rule; they judge a target by its graded exemplar similarity to the examples they saw, where two numbers are similar to the extent that they are close on the number line and share mathematical/digit features (parity, last digit, multiples, powers, primality, digits contained), and "yes" rises with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian model on members of the winning rule that are far from or featurally unlike the examples (which it endorses fully and this model only partly, producing graded typicality), and on non-members that are numerically close to the examples or share their last digit (which it rejects and this model partially endorses).

## bounded_range_rule_learning  — rank 15, 115.3 ± 36.6 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1668.7

People believe the program produces numbers that follow a mathematical rule but only within a bounded range spanned by the examples: they do Bayesian concept learning over "rule restricted to a range" concepts (any of the 45 rules, or no rule at all, intersected with an interval that covers every example), weighting each by the size principle with an inferred, tempered evidence strength. Because a rule restricted to the examples' range is a smaller, better-fitting concept than the unbounded rule, people generalize mainly to rule members inside the [min, max] span of the examples and endorse rule members far outside that span much less.

## featural_contrast_similarity  — rank 16, 126.9 ± 29.2 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1680.4

People do not pick a single rule; they judge a target by featural similarity to the examples, in the manner of Tversky's contrast model. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, decade), and the more of the examples' shared properties the target also has (weighted by how distinctive each property is), and the fewer extra properties the examples lack, the more likely people are to say yes — so members of a rule are graded by how many other properties they share with the examples, and non-members that share e.g. a last digit still get partial endorsement.

## outlier_tolerant_concept_learning  — rank 17, 170.9 ± 22.1 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1724.3

People do normative Bayesian concept learning over the 45 mathematical rules (each with its own prior weight) and all intervals of 1-100, using the size principle, but they mis-estimate how the examples were generated: they believe a much larger share of the examples are random strays from 1-100 than the ideal learner assumes (an inferred noise belief instead of a fixed 10%). Because a stray-tolerant learner keeps believing in a tight concept (a small rule or a narrow interval around the examples) even when one example does not fit it, belief stays on tight hypotheses rather than spreading to broad rules and wide intervals. So members of the tight hypotheses keep graded endorsement, while far-away non-members, which only broad hypotheses cover, are rejected more firmly.

## bigelow_piantadosi_mixture  — rank 18, 249.8 ± 39.3 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1803.3

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## bigelow_piantadosi_independent  — rank 19, 269.6 ± 40.8 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1823.1

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## tenenbaum_rules_similarity  — rank 20, 561.1 ± 82.4 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2114.5

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
