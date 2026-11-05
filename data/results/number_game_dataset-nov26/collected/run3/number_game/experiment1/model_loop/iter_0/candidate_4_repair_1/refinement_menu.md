# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_mixture`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bigelow_piantadosi_independent — rank 1, 19.8 ± 55.2 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -1823.1

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — rank 2, 311.2 ± 79.8 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2114.5

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### featural_contrast_similarity — no comparison row

**Hypothesis:** People do not pick a single rule; they judge a target by featural similarity to the examples, in the manner of Tversky's contrast model. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, decade), and the more of the examples' shared properties the target also has (weighted by how distinctive each property is), and the fewer extra properties the examples lack, the more likely people are to say yes — so members of a rule are graded by how many other properties they share with the examples, and non-members that share e.g. a last digit still get partial endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/featural_contrast_similarity.py`

### featural_exemplar_similarity — no comparison row

**Hypothesis:** People do not pick a single best-fitting rule; they judge a target by its graded exemplar similarity to the examples they saw, where two numbers are similar to the extent that they are close on the number line and share mathematical/digit features (parity, last digit, multiples, powers, primality, digits contained), and "yes" rises with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian model on members of the winning rule that are far from or featurally unlike the examples (which it endorses fully and this model only partly, producing graded typicality), and on non-members that are numerically close to the examples or share their last digit (which it rejects and this model partially endorses).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/featural_exemplar_similarity.py`

### feature_overlap_similarity — no comparison row

**Hypothesis:** People do not infer a single hidden rule; instead they judge the target by its featural similarity to the examples they were shown. Each number is represented by its properties (parity, primality, being a square/cube/power, divisibility, final digit, digits it contains, and which decade it lies in), and the more properties the target shares with each example on average (feature overlap relative to the properties either has), the more likely people say yes — so endorsement is graded even among numbers that fit a rule, and partial matches such as a shared final digit or nearby magnitude still earn some yeses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/feature_overlap_similarity.py`

### conservative_evidence_concept_learning — no comparison row

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, but they under-weight the evidence of the examples, as if each example counted for only a fraction of a full observation (a conservative, tempered likelihood). The one change is a free evidence-weight exponent on the likelihood of the examples, so confidence in the smallest consistent rule grows more slowly with examples and more posterior stays on broader rules and intervals — making learners less sure of the winning rule than the ideal learner.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/conservative_evidence_concept_learning.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
