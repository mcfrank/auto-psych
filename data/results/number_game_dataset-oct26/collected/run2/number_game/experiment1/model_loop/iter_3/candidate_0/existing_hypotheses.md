# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## graded_factor_coherence_tuned_exemplar  — rank 0, the best model on this data, ELPD-LOO -1551.7

Refinement of graded_shared_factor_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples, with the number-theoretic part of similarity graded by how large a factor the target and each example share, but the attention they give each kind of similarity (size, final digit, parity, shared factor) is stretched toward the respects in which the examples resemble one another. The single change grafts this coherence-tuned attention (from coherence_tuned_attention_exemplar) onto the graded model, with the examples' factor coherence itself measured by the same graded shared-factor match, so sets whose examples share a large common factor make people generalize sharply to targets sharing that factor, closing the common-factor rule gap, while a single example leaves attention at baseline.

## coherence_tuned_attention_exemplar  — rank 1, 0.2 ± 1.3 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -1551.9

People judge a target by its recency-weighted similarity to the remembered examples, but how much attention they give each kind of similarity (closeness in size, shared final digit, shared parity, shared factor) is learned from the examples themselves: while reading the set they notice the respects in which the examples resemble one another and stretch attention toward those dimensions. So when all examples share a factor, final digit or parity, targets matching every example on that property are endorsed far more than pairwise similarity with fixed attention predicts, while a single example gives no such cue and leaves attention at its baseline.

## set_factor_tuned_attention_exemplar  — rank 2, 0.3 ± 1.4 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -1552.1

Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the examples, with attention stretched toward the dimensions on which the examples resemble one another, but when all the examples share a common factor of 3 or more they notice that specific factor and the factor dimension of similarity becomes "is the target a multiple of the examples' shared factor", rather than whether the target shares any factor with each example separately. The single change is this set-specific redefinition of factor similarity, which predicts a sharper "multiples of g" generalisation (e.g. after 12, 18, 30 the target 42 is endorsed but 15 and 21 are not), closing the common-factor rule gap in the critique.

## capacity_limited_attention_exemplar  — rank 3, 16.0 ± 8.9 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1567.8

Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples, with attention stretched toward the dimensions (size, final digit, parity, shared factor) on which the examples resemble one another, but attention is a fixed-capacity resource, as in selective-attention exemplar theory. The single change is that coherence reallocates a fixed total amount of attention rather than adding attention: stretching attention toward the property the examples share withdraws it from the other dimensions, chiefly closeness in size. So after a coherent set (e.g. 12, 18, 30) far-away targets that share the property are endorsed more and nearby targets that lack it less, sharpening the "multiples of g" gap and making set-level yes rates more extreme than the incumbent predicts, while a single example leaves attention at its baseline.

## luce_ratio_exemplar_decision  — rank 4, 30.7 ± 9.4 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1582.5

People judge a target by its recency-weighted similarity to the remembered examples (closeness in size, shared final digit, shared parity, divisibility), exactly as in the simple exemplar model, but they turn that familiarity into a yes/no answer by a Luce ratio decision rule: they compare the target's familiarity against a fixed internal criterion familiarity and say "yes" with probability equal to the target's share of (familiarity + criterion), each raised to a response-determinism exponent. Because the rule operates on familiarity ratios rather than differences, targets with very low familiarity are rejected almost always and sets whose examples resemble the targets weakly become near-all-"no", producing more extreme set-level yes rates than a linear-in-familiarity criterion.

## graded_shared_factor_exemplar_similarity  — rank 5, 41.0 ± 10.1 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1592.8

Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples, but the number-theoretic part of that similarity is graded by how large a factor the target and each example share, not by whether one divides the other. The single change replaces the all-or-none divisibility mismatch with a graded shared-factor mismatch (zero when one number divides the other, one when they share no factor, in between in proportion to the log size of their greatest common divisor relative to the smaller number), so targets that share a big factor with the examples (e.g. 42 after 12, 18, 30) are endorsed more than divisibility alone predicts, which is what the common-factor rule gap in the critique shows.

## recency_weighted_exemplar_similarity  — rank 6, 43.2 ± 10.1 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1594.9

People do not infer a hidden rule; they judge a target by its similarity to the remembered examples, as in an exemplar model. Similarity to each example decays exponentially with the psychological distance between the two numbers, a weighted sum of mismatches on how far apart they are, whether they share a final digit, whether they share parity, and whether one divides the other. Memory for the examples fades with how long ago each was shown, so the last-shown example counts most, and the probability of "yes" rises with the target's recency-weighted summed similarity to the examples.

## common_factor_exemplar_similarity  — rank 7, 44.9 ± 10.5 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1596.7

Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples (mismatches on magnitude, final digit, parity and pairwise divisibility), but when several examples share a common factor of 3 or more they also encode that shared factor as a property of the whole set, so a target that is not a multiple of it is psychologically farther from every example. The single change is this one set-level "shared common factor" mismatch dimension in the distance, which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise exemplar similarity alone.

## running_prototype_delta_rule  — rank 8, 48.8 ± 11.4 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1600.6

People do not store the examples separately; they keep a single running summary of what the program's numbers are like (their size, parity, final digit and which small factors they have) and revise it as each example arrives, moving the summary only part of the way toward each new example. A target is endorsed in proportion to how well it fits that final summary, so example order matters through one learning rate: a slow learner stays anchored on the first example (primacy), a fast one is dominated by the last (recency), and properties that every example shares survive the updating at full strength while properties of only one example are diluted.

## tempered_concept_learning_with_guessing  — rank 9, 75.8 ± 17.4 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1627.6

Refinement of tempered_size_principle_concept_learning: people do the same Bayesian concept learning over 42 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but on some fraction of trials they do not consult the inferred concept at all and simply guess, answering "yes" with their own inferred guessing bias. The single change is this inferred lapse-to-guessing process, which predicts that even targets the concept clearly excludes get a substantial floor of "yes" answers (and clearly included targets a ceiling of "no" answers) that is the same for every set and target, rather than the near-certain answers the pure concept learner gives.

## log_odds_distorted_concept_learning  — rank 10, 83.7 ± 17.1 nats behind the best (4.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1635.4

People are Bayesian concept learners over mathematical rules (powers, multiples, final and contained digits, primes, parity, squares, cubes) and numeric intervals with the size principle, but they misperceive their own posterior belief that the target belongs to the concept the way people distort probabilities generally: the belief is compressed or expanded linearly in log-odds around an inferred crossover point before it is turned into a yes/no answer. Weak beliefs are therefore over-endorsed and near-certain ones under-endorsed (or the reverse), while the ordering of targets is still set entirely by the normative posterior.

## tversky_contrast_feature_similarity  — rank 11, 83.8 ± 16.5 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1635.6

People judge a target by Tversky-style featural similarity to the examples: each number is mentally represented as a set of salient arithmetic features (odd/even, being a multiple of 3 through 10, square, cube, prime, power of two, its final digit and its decade), and a target's similarity to an example rises with the features they have in common and falls with the features each has that the other lacks, weighted asymmetrically. The probability of "yes" rises with the target's average featural similarity to the examples, so targets sharing a property with every example (e.g. a common factor) are endorsed sharply more, while closeness in size matters only through sharing a decade.

## tempered_size_principle_concept_learning  — rank 12, 107.1 ± 21.7 nats behind the best (4.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1658.9

A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100, but they apply the size principle only partially — they treat the examples as weaker evidence than a fully strong-sampling learner would, so the evidence from the examples is tempered by one inferred strength (1 recovers the incumbent; below 1 means broader generalization). The single change is this evidence-strength exponent, applied equally to every example regardless of order; it predicts more "yes" answers to targets outside the tightest consistent concept, most of all after a single example, where the incumbent generalises too narrowly.

## recency_forgetting_concept_learning  — rank 13, 120.5 ± 22.3 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1672.2

People are Bayesian concept learners (rules and intervals, size principle) who reason only from the examples they still hold in memory: the last-shown example is always remembered, and each earlier example is forgotten with a probability that grows the further back it was shown, so the concept is inferred from a recency-biased subset of the set. This disagrees most with the order-invariant best model on multi-example sets whose early examples point to a different concept than the last one: for targets close to the last example but outside the concept implied by the whole set it predicts many more "yes" answers, and for targets near only the first example, fewer.

## mixed_sampling_concept_learning  — rank 14, 128.7 ± 24.2 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1680.4

Refinement of bigelow_piantadosi_independent: the same Bayesian concept learner over 42 rules and all intervals of 1-100 (rule weights inferred separately, intervals equally likely, 10% noise examples), but people only partly believe the examples were sampled from the concept: each example is taken as drawn from the concept with an inferred probability (strong sampling, the size principle) and otherwise as an arbitrary number that merely happens to belong to it (weak sampling). The one change is this strong/weak sampling mixture, which weakens the size principle's penalty on large concepts, so people generalise more broadly than the incumbent predicts, most of all from a single example.

## recency_decayed_concept_learning  — rank 15, 168.4 ± 25.6 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.2

People infer the hidden program by Bayesian concept learning over mathematical rules and numeric intervals, but their memory of the examples decays with recency: the most recently shown example counts as full evidence while each earlier example is discounted by a constant factor per position back. Concepts consistent with the later examples therefore dominate, so targets near or sharing structure with the last-shown example get more "yes" answers than those tied to the first-shown one, and evidence from long sets is weaker (broader generalization).

## shared_divisor_rule_abstraction  — rank 16, 182.5 ± 21.5 nats behind the best (8.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1734.2

People abstract a single explicit rule from the examples: "multiples of g", where g is the greatest common divisor the examples share. They endorse targets that are multiples of g and reject those that are not, and they trust this rule more the less likely it is that the examples would share g by coincidence (more examples and a larger g); when the examples share no factor above 1, people fall back on nothing but how close the target is to the examples.

## bigelow_piantadosi_independent  — rank 17, 240.4 ± 30.0 nats behind the best (8.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1792.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## bigelow_piantadosi_interval  — rank 18, 250.4 ± 34.1 nats behind the best (7.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1802.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## recency_weighted_concept_learning  — rank 19, 274.2 ± 37.0 nats behind the best (7.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1826.0

Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and 1-100 intervals (with a midpoint-dependent interval prior and the size principle), but they weigh the examples by recency — each example's evidence is discounted geometrically by how many examples were shown after it (one inferred decay rate, total evidence held fixed), so concepts consistent with the most recently shown examples gain posterior weight and targets near the last example are endorsed more. The single change is this order-dependent evidence weighting, motivated by the critique's recency discrepancy; a decay of 1 recovers the original model.

## tenenbaum_rules_similarity  — rank 20, 503.0 ± 72.8 nats behind the best (6.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2054.7

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
