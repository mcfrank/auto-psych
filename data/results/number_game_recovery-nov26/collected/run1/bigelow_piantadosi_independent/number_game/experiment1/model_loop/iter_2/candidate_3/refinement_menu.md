# Refinement menu

The models you may refine, other than the incumbent `exception_tolerant_tempered_mixture`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### graded_feature_profile_matching — rank 1, 1.6 ± 13.6 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -810.1

**Hypothesis:** Refinement of `common_feature_profile_matching`: people still judge membership by matching the target against the arithmetic and surface properties of the examples (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but a property enters the profile in proportion to how many of the examples have it rather than only when every example has it. The one change is this graded, frequency-weighted profile (its steepness fitted by one exponent), so a target that shares a final digit or another property with only some of the examples still gains support — closing the critique that people endorse partial last-digit matches far more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/graded_feature_profile_matching.py`

### tempered_size_principle_mixture — rank 2, 33.6 ± 8.3 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -842.2

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — the evidence from the examples is tempered by a single fitted exponent, so a few examples that happen to sit close together do not overwhelm a broad rule such as "even" or "ends in 2" the way strict strong sampling would. The one change is this evidence-weighting exponent (sharpness of the size principle), which shifts belief toward broad rules when examples share a regularity and lets generalisation tighten less with more examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_mixture.py`

### memory_limited_example_recall — rank 3, 36.7 ± 8.7 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -845.2

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numerical intervals, but they reason from an imperfect memory of the examples: each example is independently retained with some probability, and they generalise from only the examples they hold in mind. Because a judgement is often based on one or two of the examples rather than all of them, a target that shares a property (such as a final digit or a divisor) with just one example is often endorsed, and generalisation tightens less with more examples than an ideal learner who uses every example would predict.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/memory_limited_example_recall.py`

### tempered_size_principle_digit_unions — rank 4, 38.0 ± 9.0 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -846.6

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but their rule vocabulary also includes disjunctive last-digit concepts such as "ends in 2 or 7" or "ends in 1, 6 or 0" (unions of two to four final digits), each made less plausible a priori by a fitted penalty per extra digit. The one change is this enlarged hypothesis space: when the examples' final digits differ, people can still infer a final-digit rule covering every example, so they say yes to distant targets that share a final digit with any one example far more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_digit_unions.py`

### common_feature_profile_matching — rank 5, 41.6 ± 16.5 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -850.2

**Hypothesis:** People judge membership by common-feature matching rather than by probabilistic concept induction: they notice which arithmetic and surface properties (parity, being a multiple, power, square or prime, a final digit, a digit it contains, its decade) every example shares, and say yes in proportion to how much of that shared profile the target also has, with more distinctive (rarer) shared properties counting more. Numerical proximity plays no role, and there is no size-principle tightening with more examples beyond what changes the shared profile.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/common_feature_profile_matching.py`

### tempered_size_principle_rules_intervals — rank 6, 66.0 ± 18.7 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -874.6

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but they weigh the evidence of the examples less sharply than an ideal learner, so the size principle is tempered (the likelihood of the examples is raised to a fitted power below or above one). The one change is this fitted evidence-strength exponent: a weak size principle lets broad rules such as "even" or "ends in 2" keep their prior weight against tight intervals, so people extend rules to distant targets and tighten generalisation with more examples less than the ideal learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_rules_intervals.py`

### size_weighted_prior_mixture — rank 7, 68.1 ± 24.0 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -876.6

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning (same 45 rules, all intervals of 1-100, size-principle likelihood), but their prior is not flat within each kind of concept — it favours broader concepts in proportion to a fitted power of each concept's size. The one change is this size-weighted prior: broad rules such as "even" or "ends in 3" start out more plausible than narrow ones like "multiples of 4" or tight intervals, a fixed head start that (unlike a tempered size principle) does not grow or shrink with the number of examples, so people extend a shared regularity to distant numbers more often and endorse near numbers that break it less than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/size_weighted_prior_mixture.py`

### example_subset_memory_bayes — rank 8, 80.9 ± 15.0 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -889.4

**Hypothesis:** People do Bayesian concept learning over the same arithmetic rules and numerical intervals as an ideal learner (strong sampling, size principle), but under a working-memory limit: each example shown enters their inference only with some fitted probability, so they often generalise from just a subset of the examples. Reasoning from one or two remembered examples lets a rule shared by only some examples (e.g. "ends in 3" from 23 in {16, 23}) win, and weakens the size-principle tightening that more examples would normally produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/example_subset_memory_bayes.py`

### exemplar_digit_feature_similarity — rank 9, 124.8 ± 31.0 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -933.3

**Hypothesis:** People judge membership by surface-feature similarity to the individual examples, not by inducing a rule shared by the whole set: they compare the target with each example on a few salient digit and arithmetic features (its last digit, its tens digit, its parity, and its remainder when divided by three), and the more features the target shares with a given example the more similar it feels, with similarity falling off exponentially with the number of mismatching features. The yes probability grows with the target's average similarity to the examples, so a target that shares only its final digit with one example (in a set without a common final digit) still earns credit, and plain numerical distance plays no role beyond sharing a tens digit.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_digit_feature_similarity.py`

### common_feature_matching — rank 10, 161.8 ± 30.5 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -970.3

**Hypothesis:** People abstract the set of features that every example has in common (parity, divisibility, being a square/cube/prime/power, final digit, decade) and say "yes" to a target in proportion to how many of those shared features it also has. This is a feature-matching (common-feature contrast) process with no size principle: a rule-like feature shared by all examples is extended to any number that has it, however far away, and adding examples only matters by removing features that are no longer shared.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/common_feature_matching.py`

### tempered_size_principle — rank 11, 176.9 ± 37.5 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -985.5

**Hypothesis:** People infer the hidden concept by Bayesian reasoning over the same mathematical rules and numerical intervals, but they apply the size principle only weakly: each example favours smaller concepts by a fractional power of their size rather than in full proportion. As a result broad rules such as "even" or "ends in 7" keep their strength against narrower rules and tight intervals, so the model disagrees most sharply with the current best model on far-away targets that fit a broad rule (e.g. distant even numbers for an all-even set), which it predicts people will endorse much more often, and on near-but-off-rule targets, which it predicts they will endorse less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle.py`

### bigelow_piantadosi_mixture — rank 12, 218.0 ± 31.2 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1026.6

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — rank 13, 265.7 ± 47.7 nats behind the best (5.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1074.3

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
