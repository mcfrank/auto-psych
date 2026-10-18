# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## shared_property_rule_matching  — rank 0, the best model on this data, ELPD-LOO -1674.5

People ignore magnitude and similarity entirely and judge by rule matching: they note which arithmetic properties all the examples have in common (even, odd, multiple of 3/4/5/10, a shared final digit, prime, square, power of 2) and say "yes" to the degree that the target has those same properties, regardless of how far it lies from the examples. So rule-consistent targets are accepted even when numerically distant, rule violators are rejected even when close, and a single example (which carries all its own properties) generalizes as readily as several.

## nearest_exemplar_luce_similarity  — rank 1, 24.6 ± 13.9 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1699.1

A refinement of `exemplar_feature_similarity_gcm`. It keeps the same exemplar representation: magnitude plus arithmetic features, with similarity falling off exponentially with distance weighted separately on each. The one change is that people judge the target by its similarity to the single most similar example shown (nearest-neighbour retrieval), compared against a fixed criterion (Luce's ratio rule), rather than by similarity summed over all examples. So adding examples does not inflate "yes" for every target. A lone example generalizes as broadly as a rich set, and a target that closely matches any one example on its arithmetic features is accepted however far it lies from the others, which addresses the critique's finding that the summed model under-generalizes from single examples and to numerically distant targets.

## erlang_size_noisy_concept_learning  — rank 2, 45.7 ± 20.0 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.2

A refinement of `bigelow_piantadosi_inferred_noise` (Bayesian concept learning over 42 weighted rules and every interval of 1-100, with the size principle and an inferred share of examples treated as random noise), with one change grafted from `tenenbaum_rules_similarity`: people do not consider all intervals equally plausible a priori but favour intervals of an intermediate, inferred typical size (an Erlang prior on interval size). Because tiny intervals are a priori implausible, a single example is generalized to a broad neighbourhood rather than collapsing onto itself, whereas several examples still pin the concept down by the size principle — so people say yes more after one example than after several, and accept distant numbers more often than graded similarity predicts.

## bayesian_concepts_log_odds_distortion  — rank 3, 45.7 ± 18.3 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.2

People carry out normative Bayesian concept learning over mathematical rules and numerical intervals with the size principle, but they distort the resulting belief that the target belongs to the concept before answering: like the linear-in-log-odds probability weighting seen in risky choice, their subjective confidence is a compressed (or expanded) and shifted version of the posterior, measured in log-odds. So they say "yes" to rule-consistent targets far from the examples and "no" to targets the posterior strongly excludes less extremely than the ideal learner would, while keeping its rank order of targets.

## exemplar_feature_similarity_gcm  — rank 4, 46.1 ± 14.3 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.6

People do not induce a rule; they judge the target by its exemplar similarity to the individual examples shown (as in the Generalized Context Model). Each number is represented on two psychological dimensions — its magnitude and its arithmetic features (parity, divisibility by 3/4/5/10, final digit, tens digit, being prime, square or a power of 2) — and similarity to each example decays exponentially with attention-weighted distance on these dimensions; the probability of "yes" grows with the summed similarity to the examples, so generalization is graded and does not sharpen with the size principle.

## shared_feature_attention_gcm  — rank 5, 49.1 ± 14.1 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1723.6

A refinement of `exemplar_feature_similarity_gcm`: people still judge the target by its summed exemplar similarity to the examples on magnitude and arithmetic features, but their attention to each arithmetic feature is drawn by the examples themselves — a feature on which all the examples agree (e.g. all even, all powers of 2, all ending in 5) receives extra attention, by an inferred amount, relative to features on which the examples disagree. The one change is this example-driven attention shift, which makes a target that breaks a property shared by every example much less similar to them, so rule-consistent targets are accepted (even far away) and rule-violating ones rejected more sharply than with fixed attention.

## range_extension_by_spacing  — rank 6, 74.2 ± 16.9 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1748.8

People ignore arithmetic rules and generalize by range extension: they treat every number between the smallest and largest example as equally likely to be produced, and their willingness to say yes falls off smoothly with how far a target lies beyond that range, measured in units of the typical spacing between the examples (so tightly packed, many-example sets generalize only a little past their edges, while a lone example generalizes over a learned width). This disagrees most sharply with the rule-based Bayesian model on rule-consistent targets far outside the example range (e.g. 64 after 4,16,12), which it predicts are rejected, and on in-range rule violators, which it predicts are accepted.

## bigelow_piantadosi_inferred_noise  — rank 7, 85.3 ± 24.6 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1759.9

A refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 42 rules (with a prior weight inferred for each) and every interval of 1-100, with the size principle, but people believe a much larger and unknown share of the examples are random draws from 1-100 rather than fixed at 10%; that noise share is inferred. Treating examples as more often noise lets one stray example be explained away, so the learner commits to tighter concepts around the bulk of rich example sets (less generalization outside their range) while staying graded rather than near-certain in its judgments.

## size_weighted_interval_learning  — rank 8, 89.9 ± 26.4 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1764.4

Refining `bigelow_piantadosi_interval` (Bayesian concept learning over 42 mathematical rules and every interval of 1-100, with per-rule prior weights and a Beta prior over interval midpoints), with one change: the strength of the size principle is inferred rather than fixed, so a learner may treat each example as stronger (or weaker) evidence than one random draw from the concept. If people apply a stronger size principle than the original model assumes, rich example sets should pull their belief onto the tightest concepts that cover the examples, and they should say "yes" less often to numbers outside the examples' range, which is the gap the critique found for sets of 3-4 examples.

## inferred_noise_concept_learning  — rank 9, 94.6 ± 28.5 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1769.2

A refinement of bigelow_piantadosi_independent: the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with the size principle, but the share of examples the learner believes are random noise (rather than drawn from the concept) is inferred from the data instead of fixed at 10%. If people treat examples as rarely noisy, they refuse to discard an awkward example in favour of a broad rule like "even", so rich, rule-less example sets are explained by the interval spanning them and numbers outside their range are rejected.

## win_stay_lose_shift_concept  — rank 10, 122.4 ± 29.5 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1796.9

People do not keep a full posterior over concepts; they hold a single concept at a time (win-stay, lose-shift). They pick a concept from the first example, keep it as long as each new example fits it, and only when an example contradicts it do they shift to a new concept chosen in light of all examples seen so far; they then say "yes" exactly when the target belongs to the concept they currently hold. Because consistent examples never make the learner narrow down to a tighter concept, judgments are all-or-none rule applications that do not sharpen with more examples the way the size principle predicts, and they depend on the order in which the examples are read.

## order_weighted_concept_learning  — rank 11, 242.1 ± 47.2 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1916.6

People do Bayesian concept learning over mathematical rules and numerical intervals, with the size principle, but they do not weigh the examples equally: how much each example shifts their belief depends on where it came in the order shown, growing or shrinking steadily from first to last (primacy or recency), while the total evidence stays the same. So the same numbers shown in a different order lead to different generalizations, favouring the concepts that fit the most heavily weighted examples.

## bigelow_piantadosi_independent  — rank 12, 263.6 ± 45.1 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1938.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## bigelow_piantadosi_interval  — rank 13, 295.6 ± 48.2 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1970.2

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## tenenbaum_rules_similarity  — rank 14, 370.1 ± 48.8 nats behind the best (7.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2044.7

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
