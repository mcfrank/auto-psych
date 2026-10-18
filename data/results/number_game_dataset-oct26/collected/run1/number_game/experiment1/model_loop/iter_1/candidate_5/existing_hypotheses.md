# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## exemplar_feature_similarity_gcm  — rank 0, the best model on this data, ELPD-LOO -1720.6

People do not induce a rule; they judge the target by its exemplar similarity to the individual examples shown (as in the Generalized Context Model). Each number is represented on two psychological dimensions — its magnitude and its arithmetic features (parity, divisibility by 3/4/5/10, final digit, tens digit, being prime, square or a power of 2) — and similarity to each example decays exponentially with attention-weighted distance on these dimensions; the probability of "yes" grows with the summed similarity to the examples, so generalization is graded and does not sharpen with the size principle.

## range_extension_by_spacing  — rank 1, 28.2 ± 10.7 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1748.8

People ignore arithmetic rules and generalize by range extension: they treat every number between the smallest and largest example as equally likely to be produced, and their willingness to say yes falls off smoothly with how far a target lies beyond that range, measured in units of the typical spacing between the examples (so tightly packed, many-example sets generalize only a little past their edges, while a lone example generalizes over a learned width). This disagrees most sharply with the rule-based Bayesian model on rule-consistent targets far outside the example range (e.g. 64 after 4,16,12), which it predicts are rejected, and on in-range rule violators, which it predicts are accepted.

## bigelow_piantadosi_inferred_noise  — rank 2, 39.3 ± 25.9 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -1759.9

A refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 42 rules (with a prior weight inferred for each) and every interval of 1-100, with the size principle, but people believe a much larger and unknown share of the examples are random draws from 1-100 rather than fixed at 10%; that noise share is inferred. Treating examples as more often noise lets one stray example be explained away, so the learner commits to tighter concepts around the bulk of rich example sets (less generalization outside their range) while staying graded rather than near-certain in its judgments.

## size_weighted_interval_learning  — rank 3, 43.8 ± 28.2 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -1764.4

Refining `bigelow_piantadosi_interval` (Bayesian concept learning over 42 mathematical rules and every interval of 1-100, with per-rule prior weights and a Beta prior over interval midpoints), with one change: the strength of the size principle is inferred rather than fixed, so a learner may treat each example as stronger (or weaker) evidence than one random draw from the concept. If people apply a stronger size principle than the original model assumes, rich example sets should pull their belief onto the tightest concepts that cover the examples, and they should say "yes" less often to numbers outside the examples' range, which is the gap the critique found for sets of 3-4 examples.

## inferred_noise_concept_learning  — rank 4, 48.6 ± 32.1 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -1769.2

A refinement of bigelow_piantadosi_independent: the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with the size principle, but the share of examples the learner believes are random noise (rather than drawn from the concept) is inferred from the data instead of fixed at 10%. If people treat examples as rarely noisy, they refuse to discard an awkward example in favour of a broad rule like "even", so rich, rule-less example sets are explained by the interval spanning them and numbers outside their range are rejected.

## order_weighted_concept_learning  — rank 5, 196.0 ± 50.4 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1916.6

People do Bayesian concept learning over mathematical rules and numerical intervals, with the size principle, but they do not weigh the examples equally: how much each example shifts their belief depends on where it came in the order shown, growing or shrinking steadily from first to last (primacy or recency), while the total evidence stays the same. So the same numbers shown in a different order lead to different generalizations, favouring the concepts that fit the most heavily weighted examples.

## bigelow_piantadosi_independent  — rank 6, 217.5 ± 48.8 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1938.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## bigelow_piantadosi_interval  — rank 7, 249.6 ± 49.9 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1970.2

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## tenenbaum_rules_similarity  — rank 8, 324.1 ± 52.7 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2044.7

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
