# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bayesian_concepts_logodds_readout — rank 1, 0.4 ± 2.3 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -873.2

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, 10% noisy examples), but they do not report the posterior probability of membership directly; instead they answer "yes" through a graded, possibly biased decision readout in which the log-odds of membership is scaled by a free confidence factor and shifted by a free yes/no criterion. This lets near-certain posteriors (e.g. for the examples themselves, or for numbers far outside the concept) map onto less extreme, empirically observed yes rates, and lets judges be more or less liberal than the posterior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_concepts_logodds_readout.py`

### inferred_sampling_reliability_concepts — rank 2, 0.7 ± 2.9 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -873.5

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100, with the size principle, but how strongly they believe the examples were drawn from the concept (rather than at random from 1-100) is a free parameter inferred from the data instead of being fixed at 90%. This one change sets how sharply the size principle favours small, specific concepts over broad ones, and so how tightly people generalise from a few examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/inferred_sampling_reliability_concepts.py`

### partial_strong_sampling_rules_intervals — rank 3, 88.1 ± 21.9 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -960.9

**Hypothesis:** Refinement of the dropped `tenenbaum_rules_similarity` model (Bayesian concept learning over 33 mathematical rules and all intervals of 1-100, rules sharing a prior fraction equally and intervals weighted by an Erlang density on their size). The one change: learners do not assume every example was deliberately sampled from the concept (pure strong sampling, the size principle); they treat each example as drawn from the concept with probability theta (likelihood 1/|concept|) and otherwise as an arbitrary number that merely happens to fall in it (weak sampling, likelihood independent of concept size), with theta fitted — so the size principle is only partly applied, and generalisation is broader, e.g. beyond the examples' range and to larger rules, than strong sampling predicts. (The original's crash on cross-validation folds, a data-shape dependence on the fold, is also fixed.)

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/partial_strong_sampling_rules_intervals.py`

### spread_scaled_exemplar_similarity — rank 4, 258.5 ± 29.4 nats behind the best (8.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1131.3

**Hypothesis:** People do not infer any mathematical rule; they generalize by numerical similarity alone, judging a target more likely to be produced the closer it lies to the individual examples on the number line. The breadth of that similarity gradient is set by how spread out the examples are: tightly clustered examples yield a narrow gradient, widely scattered examples a broad one, so the same distance counts for less when the examples cover a wide range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/spread_scaled_exemplar_similarity.py`

### exemplar_digit_magnitude_gcm — rank 5, 263.1 ± 25.7 nats behind the best (10.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1135.9

**Hypothesis:** People do not infer a hidden rule; they judge a target by its summed similarity to the remembered examples (an exemplar, generalized-context-model process). Similarity to each example decays exponentially with a city-block psychological distance built from how far apart the two numbers are in magnitude and whether they differ in their tens digit and in their ones digit, and the probability of "yes" grows with the target's total similarity to the examples relative to a response criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_digit_magnitude_gcm.py`

### exemplar_similarity_number_line — rank 6, 292.8 ± 27.2 nats behind the best (10.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1165.6

**Hypothesis:** People do not induce any rule or interval: they judge a target by its summed perceptual similarity on the number line to the examples shown, with similarity decaying exponentially with numeric distance (an exemplar / GCM account), and say "yes" in proportion to that summed similarity relative to a fixed response criterion. This disagrees most sharply with the rule-and-interval learner in the gaps between widely spaced examples (e.g. 50 for the set 92,14,20,5, where an interval spanning the examples predicts "yes" but similarity predicts "no") and for targets just outside the examples' range (where the interval predicts "no" but similarity predicts "yes").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_similarity_number_line.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
