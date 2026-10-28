# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_interval`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bigelow_piantadosi_independent — rank 1, 6.1 ± 16.8 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -1800.2

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — rank 2, 92.9 ± 43.0 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1887.0

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### exemplar_feature_similarity_gcm — no comparison row

**Hypothesis:** People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity_gcm.py`

### shared_property_overlap — no comparison row

**Hypothesis:** People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_overlap.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
