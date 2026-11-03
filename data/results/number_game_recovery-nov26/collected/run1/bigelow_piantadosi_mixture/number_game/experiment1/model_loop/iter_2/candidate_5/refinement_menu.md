# Refinement menu

The models you may refine, other than the incumbent `erlang_interval_prior_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### power_law_interval_endpoints — rank 1, 1.4 ± 3.6 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -871.1

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight per rule), but when they imagine the program picking a numerical range they do not treat every endpoint as equally likely: the two endpoints are drawn from a power-law distribution over 1-100 whose exponent is fitted, so a range's prior plausibility depends on where on the number line it lies (among small versus large numbers), not just on whether it exists. This magnitude-dependent prior on interval endpoints is the one change; an exponent of zero recovers the incumbent's uniform endpoints. It changes how strongly range concepts compete with rules for small-number versus large-number examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/power_law_interval_endpoints.py`

### erlang_width_interval_concepts — rank 2, 3.0 ± 1.7 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -872.6

**Hypothesis:** Refinement of bayesian_concepts_logodds_readout: people do the same Bayesian concept learning over the 45 mathematical rules and all intervals of 1-100 (size principle, 10% noisy examples) and answer through the same scaled, shifted log-odds readout, but they do not consider all interval sizes equally plausible a priori. The one change: an interval's prior depends on its width through an Erlang density with a fitted typical width, so people expect the program's range of numbers to have a characteristic breadth, which sets how far they generalise beyond the span of the examples independently of the size principle.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/erlang_width_interval_concepts.py`

### bigelow_piantadosi_independent — rank 3, 3.2 ± 3.2 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -872.8

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### bayesian_concepts_logodds_readout — rank 4, 3.5 ± 2.7 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -873.2

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, 10% noisy examples), but they do not report the posterior probability of membership directly; instead they answer "yes" through a graded, possibly biased decision readout in which the log-odds of membership is scaled by a free confidence factor and shifted by a free yes/no criterion. This lets near-certain posteriors (e.g. for the examples themselves, or for numbers far outside the concept) map onto less extreme, empirically observed yes rates, and lets judges be more or less liberal than the posterior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_concepts_logodds_readout.py`

### inferred_sampling_reliability_concepts — rank 5, 3.9 ± 4.3 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -873.5

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100, with the size principle, but how strongly they believe the examples were drawn from the concept (rather than at random from 1-100) is a free parameter inferred from the data instead of being fixed at 90%. This one change sets how sharply the size principle favours small, specific concepts over broad ones, and so how tightly people generalise from a few examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/inferred_sampling_reliability_concepts.py`

### leaky_memory_concept_learning — rank 6, 4.8 ± 3.2 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -874.4

**Hypothesis:** People do Bayesian concept learning over mathematical rules and intervals (with the size principle), but they read the examples one at a time and their memory of earlier examples fades: each example's evidence is discounted by a free forgetting factor for every example shown after it, so the most recent example counts fully and the first one of a long set counts least. As a result, concepts that fit only the later examples (and rules or intervals violated by an early example) remain plausible, and generalisation from long sets is broader and recency-biased.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/leaky_memory_concept_learning.py`

### serial_position_weighted_concepts — rank 7, 6.1 ± 3.9 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -875.7

**Hypothesis:** People do Bayesian concept learning over the same 45 mathematical rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not weigh the examples equally: because the examples are seen in sequence, each example's evidence is weighted by its serial position, with a single fitted order-bias parameter that shifts weight toward the most recent examples (recency) or toward the first ones (primacy) while keeping the total amount of evidence fixed. So the same examples shown in different orders lead to different generalisations, favouring concepts that fit the weighted-up examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/serial_position_weighted_concepts.py`

### partial_strong_sampling_rules_intervals — rank 8, 91.2 ± 22.1 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -960.9

**Hypothesis:** Refinement of the dropped `tenenbaum_rules_similarity` model (Bayesian concept learning over 33 mathematical rules and all intervals of 1-100, rules sharing a prior fraction equally and intervals weighted by an Erlang density on their size). The one change: learners do not assume every example was deliberately sampled from the concept (pure strong sampling, the size principle); they treat each example as drawn from the concept with probability theta (likelihood 1/|concept|) and otherwise as an arbitrary number that merely happens to fall in it (weak sampling, likelihood independent of concept size), with theta fitted — so the size principle is only partly applied, and generalisation is broader, e.g. beyond the examples' range and to larger rules, than strong sampling predicts. (The original's crash on cross-validation folds, a data-shape dependence on the fold, is also fixed.)

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/partial_strong_sampling_rules_intervals.py`

### map_single_concept_selection — rank 9, 232.9 ± 23.9 nats behind the best (9.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1102.6

**Hypothesis:** People do not average over all the concepts consistent with the examples: they commit to the single most probable concept (a rule such as "powers of 2" or "multiples of 10", or a numeric interval, chosen by a fixed prior and the size principle) and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. Generalisation is therefore all-or-none relative to the winning concept, with no graded uncertainty from runner-up concepts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/map_single_concept_selection.py`

### spread_scaled_exemplar_similarity — rank 10, 261.7 ± 29.6 nats behind the best (8.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1131.3

**Hypothesis:** People do not infer any mathematical rule; they generalize by numerical similarity alone, judging a target more likely to be produced the closer it lies to the individual examples on the number line. The breadth of that similarity gradient is set by how spread out the examples are: tightly clustered examples yield a narrow gradient, widely scattered examples a broad one, so the same distance counts for less when the examples cover a wide range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/spread_scaled_exemplar_similarity.py`

### exemplar_digit_magnitude_gcm — rank 11, 266.3 ± 25.5 nats behind the best (10.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1135.9

**Hypothesis:** People do not infer a hidden rule; they judge a target by its summed similarity to the remembered examples (an exemplar, generalized-context-model process). Similarity to each example decays exponentially with a city-block psychological distance built from how far apart the two numbers are in magnitude and whether they differ in their tens digit and in their ones digit, and the probability of "yes" grows with the target's total similarity to the examples relative to a response criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_digit_magnitude_gcm.py`

### exemplar_similarity_number_line — rank 12, 296.0 ± 27.9 nats behind the best (10.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1165.6

**Hypothesis:** People do not induce any rule or interval: they judge a target by its summed perceptual similarity on the number line to the examples shown, with similarity decaying exponentially with numeric distance (an exemplar / GCM account), and say "yes" in proportion to that summed similarity relative to a fixed response criterion. This disagrees most sharply with the rule-and-interval learner in the gaps between widely spaced examples (e.g. 50 for the set 92,14,20,5, where an interval spanning the examples predicts "yes" but similarity predicts "no") and for targets just outside the examples' range (where the interval predicts "no" but similarity predicts "yes").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_similarity_number_line.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
