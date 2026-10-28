# Refinement menu

The models you may refine, other than the incumbent `tempered_evidence_concept_guessing`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### decision_gain_concept_learning — rank 1, 22.3 ± 7.1 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -827.4

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not report their belief that the target is in the concept directly — they turn it into a yes/no decision with an inferred decision gain on the odds. The one change is this gain: above one, people maximize (beliefs above even odds become near-certain yeses, below even odds near-certain noes); below one, they hedge, so weak beliefs (targets just outside plausible concepts) get more yeses and strong beliefs (the examples themselves) fewer, symmetrically in log-odds rather than as a flat guessing floor.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/decision_gain_concept_learning.py`

### inferred_example_noise_belief — rank 2, 23.5 ± 8.1 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -828.5

**Hypothesis:** People do Bayesian concept learning over weighted mathematical rules and number intervals with the size principle, but they distort one assumption of the normative learner: how often they believe an example is a random number not produced by the concept. That believed noise rate is a free, inferred quantity rather than a fixed small value; the more examples people discount as noise, the more readily they accept rules that fit only some of the examples (e.g. "ends in 2" after 92,82,27,42) and the broader the intervals they entertain.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/inferred_example_noise_belief.py`

### example_noise_concept_learning — rank 3, 24.4 ± 7.0 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -829.4

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle) with occasional guessing, but how readily they treat an example as a stray "noise" number that does not belong to the concept is a property of the learner, inferred from the data, rather than fixed at 10%. The one change is this inferred example-noise rate: a high rate lets a rule survive an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42) and a low rate forces a concept that covers every example, so it changes which concept people infer from sets containing an outlier.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/example_noise_concept_learning.py`

### concept_learning_with_guessing — rank 4, 25.7 ± 7.7 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -830.8

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over 42 weighted rules and all intervals (size principle, 10% noisy examples), but on a fraction of trials they do not consult the inferred concept at all and instead guess, answering "yes" with their own guessing bias. The one change is this inferred guessing (lapse) rate with a yes-bias, which puts a floor under "yes" for targets far outside every plausible concept (the far-outside-range yeses the incumbent under-produces) and a ceiling below 1 for the examples themselves.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/concept_learning_with_guessing.py`

### serial_position_weighted_concept_learning — rank 5, 26.2 ± 8.9 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -831.2

**Hypothesis:** People read the examples one at a time and integrate them into the same Bayesian concept learning over rules and intervals (with occasional guessing) as the best model, but their memory and attention are unequal across the sequence: each example's evidence is weighted by its serial position, with a single inferred gradient that favours the most recent examples (leaky memory / forgetting) or the earliest ones (primacy, fading attention). Because earlier and later examples then count differently, generalization from multi-example sets depends on the order in which the same numbers were shown, unlike the order-blind incumbent; a flat gradient recovers it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/serial_position_weighted_concept_learning.py`

### bigelow_piantadosi_independent — rank 6, 30.8 ± 10.2 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -835.8

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### interval_width_power_prior — rank 7, 33.1 ± 11.2 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -838.2

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 42 weighted rules and all intervals of 1-100 with the size principle, but their prior over intervals is not flat — it rises or falls as a power of the interval's width, with the exponent inferred. A prior that favours wide intervals offsets the size principle and makes people generalize further beyond the range of the examples (the incumbent under-produces "yes" to targets far outside that range); an exponent of zero recovers the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/interval_width_power_prior.py`

### weak_sampling_concept_elimination — rank 8, 68.0 ± 21.0 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -873.0

**Hypothesis:** People generalize by weak-sampling hypothesis elimination: they entertain the same kinds of candidate concepts (mathematical rules and numeric intervals) but treat the examples only as positive labels on numbers that happened to come up, not as draws from the concept, so every concept consistent with the examples stays equally plausible regardless of its size (no size principle). "Yes" is the prior-weighted share of the still-consistent concepts that contain the target, with an occasional example tolerated as an error; small, tight concepts get no advantage, so generalization is far broader than under strong sampling.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/weak_sampling_concept_elimination.py`

### tempered_size_principle_rules_intervals — rank 9, 189.2 ± 26.1 nats behind the best (7.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -994.2

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over mathematical rules and intervals as in that model, but they discount the evidence of the examples — each example counts for only a fraction (or a multiple) of a fully independent random draw from the concept, by a fitted evidence weight — so the size principle is tempered and generalization stays broader than strong sampling predicts, reaching targets well outside the examples' range. The single change is this fitted tempering of the likelihood (the strength of the size principle), which the original fixes at full strength.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_rules_intervals.py`

### shared_property_match_fraction — rank 10, 325.8 ± 38.7 nats behind the best (8.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1130.8

**Hypothesis:** People do not weigh competing concepts or the size of each concept; instead they note every property that all the examples have in common (parity, final digit, tens decade, being a multiple, power, square, cube or prime, lying within the examples' range) and endorse a target in proportion to how many of those shared properties it also has. The more of the examples' common properties the target matches, the more likely a "yes", regardless of how many numbers in 1-100 share those properties.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_match_fraction.py`

### featural_exemplar_similarity — rank 11, 327.6 ± 30.0 nats behind the best (10.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1132.6

**Hypothesis:** People do not entertain candidate rules at all; instead they judge a target by its featural similarity to each example they saw, treating numbers as bundles of salient properties (parity, divisibility by 3, 4, 5 and 10, last digit, decade, being prime, square or a power of two). Each example's similarity falls off exponentially with the number of properties on which it and the target disagree, and the chance of saying "yes" rises with the average similarity across the examples (an exemplar, GCM-style judgment with no size principle).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/featural_exemplar_similarity.py`

### tenenbaum_rules_similarity — rank 12, 335.6 ± 56.1 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1140.6

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### log_magnitude_exemplar_similarity — rank 13, 584.9 ± 42.0 nats behind the best (13.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1389.9

**Hypothesis:** People ignore arithmetic rules entirely and generalize by exemplar similarity on a compressed (logarithmic) mental number line: the chance of saying "yes" falls off smoothly with the target's distance to the nearest example, and the falloff is wider when the examples themselves are spread far apart. This disagrees most sharply with the rule-based best model on rule-consistent targets far from every example (e.g. 64 after 16,32,2,8 — rules say yes, similarity says unlikely) and on rule-violating targets nestled among the examples (e.g. 20 after 16,32,2,8 — rules say no, similarity says likely).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/log_magnitude_exemplar_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
