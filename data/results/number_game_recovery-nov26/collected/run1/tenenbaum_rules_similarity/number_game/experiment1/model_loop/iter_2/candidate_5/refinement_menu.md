# Refinement menu

The models you may refine, other than the incumbent `size_principle_strength_family_prior`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### family_weighted_concept_prior — rank 1, 0.0 ± 0.9 nats behind the best (0.0× dse: statistically tied with the best), ELPD-LOO -939.9

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same rules and intervals with the size principle, but they do not treat all rules as equally likely a priori. Instead, each family of concepts — parity (odd/even), special mathematical sets (primes, squares, cubes, powers), multiples of k, "ends in digit d", "contains digit d", and numeric intervals — has its own prior probability, shared equally by that family's members. So a broad but natural rule like "even" or "multiples of 3" can outweigh a tighter but implausible one like "contains the digit 2", which the size principle alone would favor.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/family_weighted_concept_prior.py`

### posterior_sharpened_concept_selection — rank 2, 1.0 ± 0.8 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -940.9

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people weigh the same rules and intervals by prior plausibility and the size principle, but they do not probability-match over the whole posterior — when settling on what the program does, they favour the most convincing concepts more strongly than Bayes would (or less, if their selection is noisy), as if choosing among concepts with a sharpening exponent on each concept's posterior weight. The one change is this inferred decision temperature: a sharpening exponent above one lets a clearly winning rule like "ends in 3" crowd out the many loosely fitting intervals, so in-range targets that break the shared final digit are rejected more often.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/posterior_sharpened_concept_selection.py`

### family_weighted_learned_reliability — rank 3, 2.8 ± 1.3 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -942.7

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass for each concept family, but they do not assume a fixed 10% of the examples are random noise — how much they trust the examples to come from the hidden concept is a free, inferred parameter (the component taken from `learned_example_reliability_ceiling`). The one change is this learned example reliability: if people trust the examples more, tight intervals that cover only one example and write the others off as noise lose their support, so for sets whose examples share a final digit people reject in-range targets that break that digit more firmly than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/family_weighted_learned_reliability.py`

### family_prior_reliability_ceiling — rank 4, 5.9 ± 2.5 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -945.8

**Hypothesis:** Refinement of `learned_example_reliability_ceiling`: people do Bayesian concept learning over the same 45 rules and all intervals with the size principle, trust the examples to a learned degree, and endorse a believed member only up to a ceiling — but they do not treat all rules as equally likely a priori. The one change is grafting the family-weighted concept prior: each family of concepts (parity, special mathematical sets, multiples of k, "ends in digit d", "contains digit d", intervals) has its own prior mass shared by its members, so a natural rule like "ends in 3" can outweigh the interval spanning the examples, which sharpens rejection of in-range targets that break a shared final digit.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/family_prior_reliability_ceiling.py`

### learned_example_reliability_ceiling — rank 5, 249.9 ± 27.1 nats behind the best (9.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1189.7

**Hypothesis:** Refinement of `concept_learning_endorsement_ceiling`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals (size principle, rule-vs-interval prior), and endorse a believed member only up to a fixed ceiling, but they do not assume a fixed 10% of the examples are random noise — how much they trust the examples to come from the hidden concept is itself a free parameter. The one change is this learned example-reliability: if people trust the examples more than the original model assumes, a single example that breaks a rule (an odd number among evens, a different final digit) rules that concept out more decisively, sharpening rejection of parity- and digit-violating targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/learned_example_reliability_ceiling.py`

### concept_learning_endorsement_ceiling — rank 6, 260.6 ± 27.1 nats behind the best (9.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1200.5

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over rules and intervals (size principle, rule-vs-interval prior lambda_0), but they do not say "yes" whenever they believe the concept contains the target — they endorse a target only with a fixed probability (an endorsement ceiling below 1), so "yes" is that ceiling times the posterior probability of membership. The one change is this conservative endorsement ceiling, which lowers yes rates uniformly for every believed member (near-example, same-parity, same-digit targets alike) instead of letting the posterior map straight onto yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/concept_learning_endorsement_ceiling.py`

### bigelow_piantadosi_mixture — rank 7, 318.5 ± 38.6 nats behind the best (8.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1258.3

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### evidence_overweighted_size_principle — rank 8, 356.4 ± 38.2 nats behind the best (9.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1296.3

**Hypothesis:** People do Bayesian concept learning over the same rules (equally likely) and numeric intervals with the size principle, but they over-weight the evidence: each example counts as if it were seen more than once, so the likelihood of the examples is raised to a power above one (an "evidence-overweighting" distortion of the normative posterior). This makes tight rules that fit every example (e.g. "ends in 3" for 13, 23, 43) crowd out broader intervals more decisively than the normative model allows, so in-range targets that break such a rule are rejected more sharply.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/evidence_overweighted_size_principle.py`

### shared_property_matching — rank 9, 371.8 ± 30.8 nats behind the best (12.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1311.7

**Hypothesis:** People do not weigh competing concepts by how well they explain the examples; instead they note every salient number property that all the examples share (parity, final digit, tens digit, being a multiple or power of something, prime, square, containing a digit) and say "yes" to a target in proportion to how many of those shared properties it also has. A target is rejected to the degree it breaks the examples' common properties, regardless of how close it lies in magnitude to any example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_matching.py`

### rule_only_concept_learning — rank 10, 386.4 ± 32.5 nats behind the best (11.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1326.3

**Hypothesis:** People generalize only through arithmetic and digit rules, never through numerical magnitude: they weigh the 45 candidate rules (parity, primes, squares, cubes, powers, multiples of k, "ends in d", "contains d") by how well each explains the examples under the size principle, and say "yes" with the posterior probability that the target obeys the rule, with no interval or closeness-based hypotheses at all. A target lying right between the examples but breaking every rule they share (e.g. 23 for 21, 41, 81) should therefore be rejected, not endorsed as nearby.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/rule_only_concept_learning.py`

### shared_property_profile_match — rank 11, 412.6 ± 36.8 nats behind the best (11.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1352.5

**Hypothesis:** People do not average over candidate concepts or generalise by numerical closeness; instead they note which arithmetic and digit properties (parity, primality, squares, powers, multiples, final digit, contained digit) every example shares, and say "yes" in proportion to how much of that shared-property profile the target also has, with rarer (more distinctive) properties counting more. This disagrees most sharply with the best model on targets just next to an example that break the shared properties (e.g. 17 for the set 16, 8, 64), which it predicts are usually rejected because there is no interval or magnitude-based generalisation at all.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_profile_match.py`

### distinctive_feature_matching — rank 12, 429.5 ± 37.9 nats behind the best (11.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1369.3

**Hypothesis:** People do not entertain explicit hypotheses about the hidden program; instead they judge a target by feature matching: they notice which arithmetic and digit properties (parity, primality, being a square or a power, being a multiple of k, final digit, contained digits, decade) all the examples have in common, and say "yes" in proportion to how much of that common-feature profile the target also shows, with rarer (more distinctive) shared properties counting more than common ones. Numerical closeness matters only insofar as it produces shared features, so a target missing a distinctive common property is rejected even if it is adjacent to an example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/distinctive_feature_matching.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
