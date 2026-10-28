# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## bigelow_piantadosi_interval  — rank 0, the best model on this data, ELPD-LOO -1794.1

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## bigelow_piantadosi_independent  — rank 1, 6.1 ± 16.8 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -1800.2

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## tenenbaum_rules_similarity  — rank 2, 92.9 ± 43.0 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1887.0

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## exemplar_feature_similarity_gcm  — no comparison row

People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

## shared_property_overlap  — no comparison row

People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.
