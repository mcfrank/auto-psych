# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## graded_profile_with_neighborhoods  — rank 0, the best model on this data, ELPD-LOO -802.4

Refinement of `graded_feature_profile_matching`: people still judge membership by matching the target against a frequency-weighted profile of the examples' properties (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but the profile also contains each example's own numerical neighbourhood ("numbers close to 27") as one more graded property, with a fitted width, weighted by the same distinctiveness rule and shared in proportion to how many examples fall near it. The one change is this soft neighbourhood property: targets just one to three away from an example gain strong support that decays steeply within a few units, closing the critique that human generalisation falls off over short numerical distances far more sharply than the current models predict.

## serial_position_weighted_evidence  — rank 1, 4.4 ± 12.0 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -806.8

People do tempered Bayesian concept learning over arithmetic rules and numerical intervals, but they do not weigh the examples equally: how much each example counts as evidence about the concept depends on its position in the order shown, rising or falling smoothly from the first example to the last (primacy or recency, the direction and strength fitted). So the same examples shown in a different order lead to different generalisations — an example that breaks the others' pattern is discounted less when it comes in the favoured position, and people generalise around it more.

## exception_tolerant_tempered_mixture  — rank 2, 6.2 ± 12.2 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -808.6

Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they believe the program often slips and produces a number outside its concept, so they tolerate exceptions — the fraction of examples assumed to be off-concept noise is a fitted parameter instead of a fixed 10%. The one change is this fitted exception rate: when most but not all examples share a regularity (e.g. three of four end in 2), the rule that covers the majority survives the exception, so people extend a partly shared last digit or arithmetic property to distant targets, and lean less on tight numerical intervals, than a learner who treats every example as strictly from the concept.

## decisive_exception_tolerant_mixture  — rank 3, 6.3 ± 11.9 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -808.7

Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they do not report their belief that the target is in the concept as a graded probability — they turn it into a near-categorical decision, so the odds of saying yes are the posterior odds raised to a fitted decisiveness exponent. The one change is this response-decisiveness transform: targets the learner thinks are probably in (or out of) the concept are endorsed (or rejected) more consistently than the posterior itself implies, making answers to each item more unanimous, sets more sharply split between narrow and broad generalisation, and near-but-off-rule neighbours of an outlier example rejected more firmly.

## graded_feature_profile_matching  — rank 4, 7.8 ± 3.4 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -810.1

Refinement of `common_feature_profile_matching`: people still judge membership by matching the target against the arithmetic and surface properties of the examples (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but a property enters the profile in proportion to how many of the examples have it rather than only when every example has it. The one change is this graded, frequency-weighted profile (its steepness fitted by one exponent), so a target that shares a final digit or another property with only some of the examples still gains support — closing the critique that people endorse partial last-digit matches far more than the incumbent predicts.

## decision_threshold_exception_tempered  — rank 5, 12.0 ± 12.1 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -814.4

Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they do not answer yes with probability equal to their belief that the target is in the concept; instead they compare that belief against an internal criterion and respond near-deterministically, with a fitted decision sharpness and criterion. The one change is this thresholded (sharpened) decision rule on the posterior: beliefs above the criterion are pushed toward a confident yes and beliefs below it toward a confident no, so responses to each item are more unanimous, sets differ more in breadth, and endorsement falls off more steeply with numerical distance than the graded posterior itself implies.

## tempered_size_principle_mixture  — rank 6, 39.8 ± 16.9 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -842.2

Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — the evidence from the examples is tempered by a single fitted exponent, so a few examples that happen to sit close together do not overwhelm a broad rule such as "even" or "ends in 2" the way strict strong sampling would. The one change is this evidence-weighting exponent (sharpness of the size principle), which shifts belief toward broad rules when examples share a regularity and lets generalisation tighten less with more examples.

## memory_limited_example_recall  — rank 7, 42.9 ± 16.9 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -845.2

People do Bayesian concept learning over arithmetic rules and numerical intervals, but they reason from an imperfect memory of the examples: each example is independently retained with some probability, and they generalise from only the examples they hold in mind. Because a judgement is often based on one or two of the examples rather than all of them, a target that shares a property (such as a final digit or a divisor) with just one example is often endorsed, and generalisation tightens less with more examples than an ideal learner who uses every example would predict.

## tempered_size_principle_digit_unions  — rank 8, 44.2 ± 16.8 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -846.6

Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but their rule vocabulary also includes disjunctive last-digit concepts such as "ends in 2 or 7" or "ends in 1, 6 or 0" (unions of two to four final digits), each made less plausible a priori by a fitted penalty per extra digit. The one change is this enlarged hypothesis space: when the examples' final digits differ, people can still infer a final-digit rule covering every example, so they say yes to distant targets that share a final digit with any one example far more than the incumbent predicts.

## soft_threshold_posterior_decision  — rank 9, 45.1 ± 15.7 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -847.5

People infer the hidden concept by the same simple Bayesian concept learning over arithmetic rules and numerical intervals as the basic mixture model, but they do not answer "yes" with a probability equal to their belief: they compare their belief that the target is in the concept against an internal criterion and answer yes when it clears that criterion, with some noise around it. This soft-threshold decision rule makes answers more all-or-none than the graded posterior (near-unanimous yes for well-supported targets, near-unanimous no for weakly supported ones), with a fitted criterion setting an overall bias toward yes or no.

## common_feature_profile_matching  — rank 10, 47.8 ± 14.3 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -850.2

People judge membership by common-feature matching rather than by probabilistic concept induction: they notice which arithmetic and surface properties (parity, being a multiple, power, square or prime, a final digit, a digit it contains, its decade) every example shares, and say yes in proportion to how much of that shared profile the target also has, with more distinctive (rarer) shared properties counting more. Numerical proximity plays no role, and there is no size-principle tightening with more examples beyond what changes the shared profile.

## tempered_size_principle_rules_intervals  — rank 11, 72.2 ± 23.2 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -874.6

Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but they weigh the evidence of the examples less sharply than an ideal learner, so the size principle is tempered (the likelihood of the examples is raised to a fitted power below or above one). The one change is this fitted evidence-strength exponent: a weak size principle lets broad rules such as "even" or "ends in 2" keep their prior weight against tight intervals, so people extend rules to distant targets and tighten generalisation with more examples less than the ideal learner does.

## size_weighted_prior_mixture  — rank 12, 74.3 ± 27.1 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -876.6

Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning (same 45 rules, all intervals of 1-100, size-principle likelihood), but their prior is not flat within each kind of concept — it favours broader concepts in proportion to a fitted power of each concept's size. The one change is this size-weighted prior: broad rules such as "even" or "ends in 3" start out more plausible than narrow ones like "multiples of 4" or tight intervals, a fixed head start that (unlike a tempered size principle) does not grow or shrink with the number of examples, so people extend a shared regularity to distant numbers more often and endorse near numbers that break it less than the incumbent predicts.

## example_subset_memory_bayes  — rank 13, 87.1 ± 21.0 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -889.4

People do Bayesian concept learning over the same arithmetic rules and numerical intervals as an ideal learner (strong sampling, size principle), but under a working-memory limit: each example shown enters their inference only with some fitted probability, so they often generalise from just a subset of the examples. Reasoning from one or two remembered examples lets a rule shared by only some examples (e.g. "ends in 3" from 23 in {16, 23}) win, and weakens the size-principle tightening that more examples would normally produce.

## exemplar_digit_feature_similarity  — rank 14, 131.0 ± 27.1 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -933.3

People judge membership by surface-feature similarity to the individual examples, not by inducing a rule shared by the whole set: they compare the target with each example on a few salient digit and arithmetic features (its last digit, its tens digit, its parity, and its remainder when divided by three), and the more features the target shares with a given example the more similar it feels, with similarity falling off exponentially with the number of mismatching features. The yes probability grows with the target's average similarity to the examples, so a target that shares only its final digit with one example (in a set without a common final digit) still earns credit, and plain numerical distance plays no role beyond sharing a tens digit.

## common_feature_matching  — rank 15, 168.0 ± 29.0 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -970.3

People abstract the set of features that every example has in common (parity, divisibility, being a square/cube/prime/power, final digit, decade) and say "yes" to a target in proportion to how many of those shared features it also has. This is a feature-matching (common-feature contrast) process with no size principle: a rule-like feature shared by all examples is extended to any number that has it, however far away, and adding examples only matters by removing features that are no longer shared.

## tempered_size_principle  — rank 16, 183.1 ± 41.7 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -985.5

People infer the hidden concept by Bayesian reasoning over the same mathematical rules and numerical intervals, but they apply the size principle only weakly: each example favours smaller concepts by a fractional power of their size rather than in full proportion. As a result broad rules such as "even" or "ends in 7" keep their strength against narrower rules and tight intervals, so the model disagrees most sharply with the current best model on far-away targets that fit a broad rule (e.g. distant even numbers for an all-even set), which it predicts people will endorse much more often, and on near-but-off-rule targets, which it predicts they will endorse less.

## numeral_edit_distance_similarity  — rank 17, 221.9 ± 35.8 nats behind the best (6.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1024.3

People judge membership by how similar the target looks, as a written numeral, to each example: similarity falls off exponentially with the number of digit edits (substitutions, insertions or deletions of a digit) needed to turn the target's numeral into an example's, and the yes probability grows with the target's average similarity across the examples. Arithmetic rules and number-line magnitude play no separate role, so a target sharing its tens digit or its final digit with many of the examples (e.g. 62 for 92, 82, 42) is endorsed even when it is far away, while one sharing no digit position with any example is rejected.

## bigelow_piantadosi_mixture  — rank 18, 224.2 ± 38.5 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1026.6

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 19, 271.9 ± 52.5 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1074.3

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
