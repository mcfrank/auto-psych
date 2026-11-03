# Refinement menu

The models you may refine, other than the incumbent `family_weighted_concept_prior`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### learned_example_reliability_ceiling — rank 1, 249.8 ± 27.0 nats behind the best (9.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1189.7

**Hypothesis:** Refinement of `concept_learning_endorsement_ceiling`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals (size principle, rule-vs-interval prior), and endorse a believed member only up to a fixed ceiling, but they do not assume a fixed 10% of the examples are random noise — how much they trust the examples to come from the hidden concept is itself a free parameter. The one change is this learned example-reliability: if people trust the examples more than the original model assumes, a single example that breaks a rule (an odd number among evens, a different final digit) rules that concept out more decisively, sharpening rejection of parity- and digit-violating targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/learned_example_reliability_ceiling.py`

### concept_learning_endorsement_ceiling — rank 2, 260.6 ± 27.0 nats behind the best (9.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1200.5

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over rules and intervals (size principle, rule-vs-interval prior lambda_0), but they do not say "yes" whenever they believe the concept contains the target — they endorse a target only with a fixed probability (an endorsement ceiling below 1), so "yes" is that ceiling times the posterior probability of membership. The one change is this conservative endorsement ceiling, which lowers yes rates uniformly for every believed member (near-example, same-parity, same-digit targets alike) instead of letting the posterior map straight onto yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/concept_learning_endorsement_ceiling.py`

### bigelow_piantadosi_mixture — rank 3, 318.4 ± 38.5 nats behind the best (8.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1258.3

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### shared_property_matching — rank 4, 371.8 ± 30.5 nats behind the best (12.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1311.7

**Hypothesis:** People do not weigh competing concepts by how well they explain the examples; instead they note every salient number property that all the examples share (parity, final digit, tens digit, being a multiple or power of something, prime, square, containing a digit) and say "yes" to a target in proportion to how many of those shared properties it also has. A target is rejected to the degree it breaks the examples' common properties, regardless of how close it lies in magnitude to any example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_matching.py`

### shared_property_profile_match — rank 5, 412.6 ± 36.6 nats behind the best (11.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1352.5

**Hypothesis:** People do not average over candidate concepts or generalise by numerical closeness; instead they note which arithmetic and digit properties (parity, primality, squares, powers, multiples, final digit, contained digit) every example shares, and say "yes" in proportion to how much of that shared-property profile the target also has, with rarer (more distinctive) properties counting more. This disagrees most sharply with the best model on targets just next to an example that break the shared properties (e.g. 17 for the set 16, 8, 64), which it predicts are usually rejected because there is no interval or magnitude-based generalisation at all.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_profile_match.py`

### distinctive_feature_matching — rank 6, 429.5 ± 37.8 nats behind the best (11.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1369.3

**Hypothesis:** People do not entertain explicit hypotheses about the hidden program; instead they judge a target by feature matching: they notice which arithmetic and digit properties (parity, primality, being a square or a power, being a multiple of k, final digit, contained digits, decade) all the examples have in common, and say "yes" in proportion to how much of that common-feature profile the target also shows, with rarer (more distinctive) shared properties counting more than common ones. Numerical closeness matters only insofar as it produces shared features, so a target missing a distinctive common property is rejected even if it is adjacent to an example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/distinctive_feature_matching.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
