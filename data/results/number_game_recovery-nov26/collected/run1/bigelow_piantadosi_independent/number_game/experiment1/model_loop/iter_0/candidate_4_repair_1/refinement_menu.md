# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_mixture`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### tenenbaum_rules_similarity — rank 1, 47.7 ± 35.4 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -1074.3

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### common_feature_matching — no comparison row

**Hypothesis:** People abstract the set of features that every example has in common (parity, divisibility, being a square/cube/prime/power, final digit, decade) and say "yes" to a target in proportion to how many of those shared features it also has. This is a feature-matching (common-feature contrast) process with no size principle: a rule-like feature shared by all examples is extended to any number that has it, however far away, and adding examples only matters by removing features that are no longer shared.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/common_feature_matching.py`

### tempered_size_principle — no comparison row

**Hypothesis:** People infer the hidden concept by Bayesian reasoning over the same mathematical rules and numerical intervals, but they apply the size principle only weakly: each example favours smaller concepts by a fractional power of their size rather than in full proportion. As a result broad rules such as "even" or "ends in 7" keep their strength against narrower rules and tight intervals, so the model disagrees most sharply with the current best model on far-away targets that fit a broad rule (e.g. distant even numbers for an all-even set), which it predicts people will endorse much more often, and on near-but-off-rule targets, which it predicts they will endorse less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle.py`

### common_feature_profile_matching — no comparison row

**Hypothesis:** People judge membership by common-feature matching rather than by probabilistic concept induction: they notice which arithmetic and surface properties (parity, being a multiple, power, square or prime, a final digit, a digit it contains, its decade) every example shares, and say yes in proportion to how much of that shared profile the target also has, with more distinctive (rarer) shared properties counting more. Numerical proximity plays no role, and there is no size-principle tightening with more examples beyond what changes the shared profile.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/common_feature_profile_matching.py`

### tempered_size_principle_mixture — no comparison row

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — the evidence from the examples is tempered by a single fitted exponent, so a few examples that happen to sit close together do not overwhelm a broad rule such as "even" or "ends in 2" the way strict strong sampling would. The one change is this evidence-weighting exponent (sharpness of the size principle), which shifts belief toward broad rules when examples share a regularity and lets generalisation tighten less with more examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_mixture.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
