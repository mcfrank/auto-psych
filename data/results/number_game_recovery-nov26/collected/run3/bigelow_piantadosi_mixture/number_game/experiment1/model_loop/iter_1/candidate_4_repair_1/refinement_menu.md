# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### interval_width_prior_concepts — rank 1, 0.3 ± 0.7 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -840.6

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learner over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100, with the size principle, but people's prior over magnitude intervals depends on the interval's width — each interval's prior is proportional to its width raised to an inferred exponent (0 recovers the incumbent's uniform interval prior; negative values mean narrow ranges like "numbers in the 60s" are a priori more plausible than wide ones). The single change is this width-dependent interval prior, which governs how readily people generalise to nearby versus distant magnitudes when no rule explains the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/interval_width_prior_concepts.py`

### bayesian_rules_with_response_lapse — rank 2, 0.9 ± 0.7 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -841.2

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, a prior weight inferred per rule), but on a fixed fraction of trials they do not consult that posterior at all and answer yes or no at random. The one change is this response-lapse rate, which lets the model explain "no" answers to targets that are themselves examples and stray "yes" answers to targets that no plausible concept contains.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_rules_with_response_lapse.py`

### tenenbaum_digit_rule_inventory — rank 3, 1.9 ± 5.6 nats behind the best (0.3× dse: statistically tied with the best), ELPD-LOO -842.2

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Tenenbaum's Bayesian concept learning (rules share one prior fraction equally, intervals are weighted by an Erlang density on their size, examples are sampled from the concept), but the rules they entertain are the richer inventory of the Bigelow-Piantadosi grammar — notably "contains the digit d" and "ends in 0", plus the two near-miss powers-of-2 sets — instead of Tenenbaum's 33. The one change is the hypothesis space; the claim is that what the people consider, not a separately learned weight for every rule, is what separates the two models, so equal rule weights with two free parameters should suffice.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_digit_rule_inventory.py`

### tenenbaum_rules_similarity — rank 4, 311.6 ± 47.5 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1151.9

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### digit_exemplar_similarity — rank 5, 348.9 ± 33.1 nats behind the best (10.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1189.2

**Hypothesis:** People do not induce a rule at all; they judge a target by its summed similarity to the remembered examples, where numbers are represented by their two decimal digits: a target feels like it "belongs" to the extent that its tens digit is close to an example's tens digit and its ones digit matches an example's ones digit. Attention is split between the tens (magnitude) and ones (final-digit) dimensions, similarity decays exponentially with this digit-wise distance, and the probability of saying yes rises with the average similarity of the target to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/digit_exemplar_similarity.py`

### digit_overlap_exemplar_similarity — rank 6, 385.0 ± 32.8 nats behind the best (11.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1225.2

**Hypothesis:** People do not induce mathematical rules or magnitude intervals at all; they judge a target by its surface resemblance to the examples, treating each number as the set of decimal digits it is written with, and say "yes" in proportion to the average digit-overlap similarity between the target and the examples (an exemplar-similarity process over written digits). This disagrees most sharply with the rule-based Bayesian model on targets that share a digit with some examples but fit no rule or interval consistent with all of them (e.g. 12 after "31,19,49,100"), and on rule-consistent targets written with entirely different digits (e.g. 2 after "16,8,64"), where it predicts low endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/digit_overlap_exemplar_similarity.py`

### spread_scaled_range_generalization — rank 7, 416.7 ± 33.2 nats behind the best (12.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1256.9

**Hypothesis:** People do not search over mathematical rules; they treat the examples as a sample from a band of magnitudes and generalize by how far the target falls outside the span the examples cover, with a tolerance that grows with how spread out the examples are. A target inside the span of the examples is endorsed at a constant high rate, and endorsement decays smoothly with the target's distance beyond the nearest edge, more slowly for widely spaced examples than for tightly clustered ones (or a single example).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/spread_scaled_range_generalization.py`

### leaky_example_memory_concepts — no comparison row

**Hypothesis:** People generalize by Bayesian concept learning over mathematical rules and magnitude intervals (the size principle, a prior weight per rule), but their working memory is leaky: each example shown is independently retained with some probability and the forgotten ones play no part in the inference. So a judgment is the rule-and-interval posterior computed from whichever examples survive, which predicts weaker rule-based tightening and broader generalization for larger example sets than full Bayesian learning, and occasional "no" answers to targets that were themselves examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/leaky_example_memory_concepts.py`

### map_single_concept_commitment — no comparison row

**Hypothesis:** People do not average over all the concepts consistent with the examples; they commit to the single concept that best explains them — the most probable rule or magnitude interval under a size-principle Bayesian evaluation with a flat prior (rules as a group as likely as intervals as a group) — and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. This predicts all-or-none generalization: no graded falloff near the examples' range and no partial endorsement of targets that only some plausible concepts contain (e.g. 32 after "16,8,64" is rejected if the smaller "powers of 2 without 32" wins).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/map_single_concept_commitment.py`

### inferred_example_noise_concepts — no comparison row

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but how much they trust the examples to come from the concept is a free parameter rather than fixed at 90%. The one change is this inferred example-noise rate: the more people believe some examples are stray draws from 1-100, the more readily they keep a tight rule or range that misses one example, rather than retreating to a broad concept that covers all of them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/inferred_example_noise_concepts.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
