# Refinement menu

The models you may refine, other than the incumbent `sample_size_growing_evidence_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### iter1_candidate4 — rank 1, 7.3 ± 6.4 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1610.7

**Hypothesis:** Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and every interval of 1-100 (inferred rule weights, a Beta midpoint prior on intervals, size-principle sampling with 10% noise, examples' evidence weighted by an inferred strength), but they also hold a prior preference for intervals of intermediate size (Tenenbaum's Erlang size prior, scale inferred, grafted from `bp_rules_erlang_interval_size`) that multiplies the midpoint prior. The one change is this size prior, which discounts tiny near-singleton intervals a priori and so predicts broader generalization from a single example — where the incumbent says "yes" too rarely — while leaving multi-example sets, whose evidence already rules out tiny intervals, largely unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/iter1_candidate4.py`

### first_example_weak_sampling_concepts — rank 2, 13.1 ± 9.8 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -1616.5

**Hypothesis:** People update their concept one example at a time, and the size principle only kicks in once there is a coincidence to explain: the first example merely rules out concepts that do not contain it (any consistent concept stays as plausible as its prior says), while every later example is treated as drawn from the concept, so small concepts that keep fitting become favoured ("suspicious coincidence"). This predicts broad generalization from a single example and a sharp tightening only when a second consistent example arrives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/first_example_weak_sampling_concepts.py`

### conservative_evidence_interval_concepts — rank 3, 15.1 ± 6.9 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1618.5

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior weight inferred for each rule, an interval prior given by a Beta density over its midpoint, examples sampled from the concept by the size principle with 10% noise), but they are conservative updaters: they weight the evidence of the examples by a single inferred strength rather than at full Bayesian value, so the posterior over concepts is flatter (strength below 1) or sharper (above 1) than the ideal learner's. The one change is this evidence-weighting exponent on the examples' likelihood, which governs how quickly generalization tightens around the narrowest consistent concept as examples accumulate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/conservative_evidence_interval_concepts.py`

### size_principle_strength_intervals — rank 4, 42.0 ± 11.0 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1645.4

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people use the same Bayesian concept learning over the 42 rules and the midpoint-weighted intervals of 1-100, but they apply the size principle only partially — how strongly a small concept is favoured for fitting the examples tightly is a graded, inferred strength, between weak sampling (examples merely consistent with a concept, no preference for small concepts) and the full strong sampling the incumbent assumes. The one change is this inferred size-principle strength, which predicts broader generalization than the incumbent (e.g. more "yes" to other even numbers after 16, 8, 64) if people underweight concept size.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/size_principle_strength_intervals.py`

### shared_property_overlap — rank 5, 52.9 ± 18.8 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1656.4

**Hypothesis:** People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_overlap.py`

### exemplar_feature_similarity_gcm — rank 6, 61.6 ± 21.7 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1665.0

**Hypothesis:** People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity_gcm.py`

### range_scaled_magnitude_gradient — rank 7, 119.4 ± 25.1 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1722.8

**Hypothesis:** People do not consider arithmetic rules or properties at all; they judge a target purely by its closeness on the number line to the nearest example, with a generalization gradient whose width stretches with the spread of the examples. A single example (no spread) still supports a fixed baseline width of generalization around it, so "yes" falls off smoothly with numerical distance, more slowly when the examples cover a wide range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/range_scaled_magnitude_gradient.py`

### bp_rules_erlang_interval_size — rank 8, 171.1 ± 29.3 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1774.6

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 rules (each with its own inferred prior weight) and every interval of 1-100, with examples sampled from the concept (the size principle, 10% noise), and answer "yes" with the posterior probability that the concept contains the target. The one change is that intervals are not equally likely a priori: people prefer intervals of intermediate size, with an Erlang prior on an interval's size whose scale is inferred (Tenenbaum's size prior grafted onto the independent-rule-weight model), so a few nearby examples generalize to a moderately wide range rather than to the very tightest or the very widest interval.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/bp_rules_erlang_interval_size.py`

### bigelow_piantadosi_interval — rank 9, 190.7 ± 28.7 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1794.1

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_interval.py`

### bigelow_piantadosi_independent — rank 10, 196.7 ± 32.2 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1800.2

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — rank 11, 283.6 ± 46.4 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1887.0

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
