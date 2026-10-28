# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## recency_weighted_exemplar_similarity  — rank 0, the best model on this data, ELPD-LOO -1594.9

People do not infer a hidden rule; they judge a target by its similarity to the remembered examples, as in an exemplar model. Similarity to each example decays exponentially with the psychological distance between the two numbers, a weighted sum of mismatches on how far apart they are, whether they share a final digit, whether they share parity, and whether one divides the other. Memory for the examples fades with how long ago each was shown, so the last-shown example counts most, and the probability of "yes" rises with the target's recency-weighted summed similarity to the examples.

## tempered_size_principle_concept_learning  — rank 1, 64.0 ± 20.4 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1658.9

A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100, but they apply the size principle only partially — they treat the examples as weaker evidence than a fully strong-sampling learner would, so the evidence from the examples is tempered by one inferred strength (1 recovers the incumbent; below 1 means broader generalization). The single change is this evidence-strength exponent, applied equally to every example regardless of order; it predicts more "yes" answers to targets outside the tightest consistent concept, most of all after a single example, where the incumbent generalises too narrowly.

## recency_forgetting_concept_learning  — rank 2, 77.3 ± 23.7 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1672.2

People are Bayesian concept learners (rules and intervals, size principle) who reason only from the examples they still hold in memory: the last-shown example is always remembered, and each earlier example is forgotten with a probability that grows the further back it was shown, so the concept is inferred from a recency-biased subset of the set. This disagrees most with the order-invariant best model on multi-example sets whose early examples point to a different concept than the last one: for targets close to the last example but outside the concept implied by the whole set it predicts many more "yes" answers, and for targets near only the first example, fewer.

## mixed_sampling_concept_learning  — rank 3, 85.5 ± 24.1 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1680.4

Refinement of bigelow_piantadosi_independent: the same Bayesian concept learner over 42 rules and all intervals of 1-100 (rule weights inferred separately, intervals equally likely, 10% noise examples), but people only partly believe the examples were sampled from the concept: each example is taken as drawn from the concept with an inferred probability (strong sampling, the size principle) and otherwise as an arbitrary number that merely happens to belong to it (weak sampling). The one change is this strong/weak sampling mixture, which weakens the size principle's penalty on large concepts, so people generalise more broadly than the incumbent predicts, most of all from a single example.

## recency_decayed_concept_learning  — rank 4, 125.3 ± 26.3 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.2

People infer the hidden program by Bayesian concept learning over mathematical rules and numeric intervals, but their memory of the examples decays with recency: the most recently shown example counts as full evidence while each earlier example is discounted by a constant factor per position back. Concepts consistent with the later examples therefore dominate, so targets near or sharing structure with the last-shown example get more "yes" answers than those tied to the first-shown one, and evidence from long sets is weaker (broader generalization).

## bigelow_piantadosi_independent  — rank 5, 197.2 ± 32.8 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1792.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## bigelow_piantadosi_interval  — rank 6, 207.2 ± 36.2 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1802.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## recency_weighted_concept_learning  — rank 7, 231.1 ± 38.9 nats behind the best (5.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1826.0

Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and 1-100 intervals (with a midpoint-dependent interval prior and the size principle), but they weigh the examples by recency — each example's evidence is discounted geometrically by how many examples were shown after it (one inferred decay rate, total evidence held fixed), so concepts consistent with the most recently shown examples gain posterior weight and targets near the last example are endorsed more. The single change is this order-dependent evidence weighting, motivated by the critique's recency discrepancy; a decay of 1 recovers the original model.

## tenenbaum_rules_similarity  — rank 8, 459.8 ± 73.9 nats behind the best (6.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2054.7

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## coherence_tuned_attention_exemplar  — no comparison row

People judge a target by its recency-weighted similarity to the remembered examples, but how much attention they give each kind of similarity (closeness in size, shared final digit, shared parity, shared factor) is learned from the examples themselves: while reading the set they notice the respects in which the examples resemble one another and stretch attention toward those dimensions. So when all examples share a factor, final digit or parity, targets matching every example on that property are endorsed far more than pairwise similarity with fixed attention predicts, while a single example gives no such cue and leaves attention at its baseline.

## log_odds_distorted_concept_learning  — no comparison row

People are Bayesian concept learners over mathematical rules (powers, multiples, final and contained digits, primes, parity, squares, cubes) and numeric intervals with the size principle, but they misperceive their own posterior belief that the target belongs to the concept the way people distort probabilities generally: the belief is compressed or expanded linearly in log-odds around an inferred crossover point before it is turned into a yes/no answer. Weak beliefs are therefore over-endorsed and near-certain ones under-endorsed (or the reverse), while the ordering of targets is still set entirely by the normative posterior.

## shared_divisor_rule_abstraction  — no comparison row

People abstract a single explicit rule from the examples: "multiples of g", where g is the greatest common divisor the examples share. They endorse targets that are multiples of g and reject those that are not, and they trust this rule more the less likely it is that the examples would share g by coincidence (more examples and a larger g); when the examples share no factor above 1, people fall back on nothing but how close the target is to the examples.

## common_factor_exemplar_similarity  — no comparison row

Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples (mismatches on magnitude, final digit, parity and pairwise divisibility), but when several examples share a common factor of 3 or more they also encode that shared factor as a property of the whole set, so a target that is not a multiple of it is psychologically farther from every example. The single change is this one set-level "shared common factor" mismatch dimension in the distance, which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise exemplar similarity alone.
