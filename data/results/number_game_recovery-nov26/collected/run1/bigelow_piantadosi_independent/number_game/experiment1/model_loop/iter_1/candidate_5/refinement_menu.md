# Refinement menu

The models you may refine, other than the incumbent `tempered_size_principle_mixture`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### common_feature_profile_matching — rank 1, 8.0 ± 21.0 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -850.2

**Hypothesis:** People judge membership by common-feature matching rather than by probabilistic concept induction: they notice which arithmetic and surface properties (parity, being a multiple, power, square or prime, a final digit, a digit it contains, its decade) every example shares, and say yes in proportion to how much of that shared profile the target also has, with more distinctive (rarer) shared properties counting more. Numerical proximity plays no role, and there is no size-principle tightening with more examples beyond what changes the shared profile.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/common_feature_profile_matching.py`

### tempered_size_principle_rules_intervals — rank 2, 32.4 ± 17.5 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -874.6

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but they weigh the evidence of the examples less sharply than an ideal learner, so the size principle is tempered (the likelihood of the examples is raised to a fitted power below or above one). The one change is this fitted evidence-strength exponent: a weak size principle lets broad rules such as "even" or "ends in 2" keep their prior weight against tight intervals, so people extend rules to distant targets and tighten generalisation with more examples less than the ideal learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_rules_intervals.py`

### size_weighted_prior_mixture — rank 3, 34.5 ± 29.0 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -876.6

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning (same 45 rules, all intervals of 1-100, size-principle likelihood), but their prior is not flat within each kind of concept — it favours broader concepts in proportion to a fitted power of each concept's size. The one change is this size-weighted prior: broad rules such as "even" or "ends in 3" start out more plausible than narrow ones like "multiples of 4" or tight intervals, a fixed head start that (unlike a tempered size principle) does not grow or shrink with the number of examples, so people extend a shared regularity to distant numbers more often and endorse near numbers that break it less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/size_weighted_prior_mixture.py`

### common_feature_matching — rank 4, 128.2 ± 32.2 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -970.3

**Hypothesis:** People abstract the set of features that every example has in common (parity, divisibility, being a square/cube/prime/power, final digit, decade) and say "yes" to a target in proportion to how many of those shared features it also has. This is a feature-matching (common-feature contrast) process with no size principle: a rule-like feature shared by all examples is extended to any number that has it, however far away, and adding examples only matters by removing features that are no longer shared.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/common_feature_matching.py`

### tempered_size_principle — rank 5, 143.3 ± 41.1 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -985.5

**Hypothesis:** People infer the hidden concept by Bayesian reasoning over the same mathematical rules and numerical intervals, but they apply the size principle only weakly: each example favours smaller concepts by a fractional power of their size rather than in full proportion. As a result broad rules such as "even" or "ends in 7" keep their strength against narrower rules and tight intervals, so the model disagrees most sharply with the current best model on far-away targets that fit a broad rule (e.g. distant even numbers for an all-even set), which it predicts people will endorse much more often, and on near-but-off-rule targets, which it predicts they will endorse less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle.py`

### bigelow_piantadosi_mixture — rank 6, 184.4 ± 28.7 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1026.6

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — rank 7, 232.1 ± 45.9 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1074.3

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
