# Refinement menu

The models you may refine, other than the incumbent `final_digit_similarity_gradient`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### shared_property_rule_matching — rank 1, 10.5 ± 11.9 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -1674.5

**Hypothesis:** People ignore magnitude and similarity entirely and judge by rule matching: they note which arithmetic properties all the examples have in common (even, odd, multiple of 3/4/5/10, a shared final digit, prime, square, power of 2) and say "yes" to the degree that the target has those same properties, regardless of how far it lies from the examples. So rule-consistent targets are accepted even when numerically distant, rule violators are rejected even when close, and a single example (which carries all its own properties) generalizes as readily as several.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_rule_matching.py`

### rule_matching_with_range_extension — rank 2, 14.6 ± 13.1 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -1678.7

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, but they also treat the span of the examples (from smallest to largest) as part of what the program produces, so a target inside that span gets an extra, inferred boost to "yes" that fades smoothly with how far the target lies beyond the span's edges (for a single example, with its distance from that example). The one change is this range-membership component grafted from `range_extension_by_spacing`, which addresses the critique that the rule model misses interval-based generalization: in-range rule violators and in-range targets of rule-less sets are accepted more than distant ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/rule_matching_with_range_extension.py`

### rule_matching_with_range_fill — rank 3, 15.4 ± 13.3 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1679.5

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, regardless of its distance from them, but they also treat the span between the smallest and largest example as territory the program covers, so a target lying inside that range gets an extra, inferred boost toward "yes" (the one change, a range-membership component taken from `range_extension_by_spacing`). So, as the critique found, in-range targets are accepted more often than rule matching alone predicts, while rule-consistent targets far outside the range are still accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/rule_matching_with_range_fill.py`

### sequential_delta_rule_property_learning — rank 4, 20.2 ± 9.4 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1684.2

**Hypothesis:** People build their guess about the program's rule incrementally, in the order the examples are shown: the first example suggests a set of arithmetic properties (parity, divisibility by 3/4/5/10, final digit, prime, square, power of 2), and each later example revises the strength of belief in each property by an error-correcting (delta-rule) update at an inferred learning rate, so that a slow learner stays anchored on the first example (primacy) and a fast learner is dominated by the last one (recency). They then say "yes" to the degree that the target has the properties their final, order-dependent guess favours, regardless of its magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/sequential_delta_rule_property_learning.py`

### nearest_exemplar_luce_similarity — rank 5, 35.0 ± 14.2 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1699.1

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`. It keeps the same exemplar representation: magnitude plus arithmetic features, with similarity falling off exponentially with distance weighted separately on each. The one change is that people judge the target by its similarity to the single most similar example shown (nearest-neighbour retrieval), compared against a fixed criterion (Luce's ratio rule), rather than by similarity summed over all examples. So adding examples does not inflate "yes" for every target. A lone example generalizes as broadly as a rich set, and a target that closely matches any one example on its arithmetic features is accepted however far it lies from the others, which addresses the critique's finding that the summed model under-generalizes from single examples and to numerically distant targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/nearest_exemplar_luce_similarity.py`

### span_filled_nearest_exemplar — rank 6, 36.4 ± 14.3 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1700.4

**Hypothesis:** A refinement of `nearest_exemplar_luce_similarity`: people still judge the target by its similarity to the most similar example on arithmetic features, combined with magnitude similarity and compared against a Luce criterion, but the one change is that on the magnitude dimension they fill in the gaps between the examples, treating the whole span from the smallest to the largest example as already "seen", so magnitude dissimilarity is the target's distance to that span (zero anywhere inside it) rather than to the nearest single example. Targets inside the examples' range are therefore accepted more readily than equally feature-similar targets outside it, addressing the critique that the incumbent misses range-based generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/span_filled_nearest_exemplar.py`

### erlang_size_noisy_concept_learning — rank 7, 56.1 ± 19.7 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.2

**Hypothesis:** A refinement of `bigelow_piantadosi_inferred_noise` (Bayesian concept learning over 42 weighted rules and every interval of 1-100, with the size principle and an inferred share of examples treated as random noise), with one change grafted from `tenenbaum_rules_similarity`: people do not consider all intervals equally plausible a priori but favour intervals of an intermediate, inferred typical size (an Erlang prior on interval size). Because tiny intervals are a priori implausible, a single example is generalized to a broad neighbourhood rather than collapsing onto itself, whereas several examples still pin the concept down by the size principle — so people say yes more after one example than after several, and accept distant numbers more often than graded similarity predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/erlang_size_noisy_concept_learning.py`

### bayesian_concepts_log_odds_distortion — rank 8, 56.1 ± 20.3 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.2

**Hypothesis:** People carry out normative Bayesian concept learning over mathematical rules and numerical intervals with the size principle, but they distort the resulting belief that the target belongs to the concept before answering: like the linear-in-log-odds probability weighting seen in risky choice, their subjective confidence is a compressed (or expanded) and shifted version of the posterior, measured in log-odds. So they say "yes" to rule-consistent targets far from the examples and "no" to targets the posterior strongly excludes less extremely than the ideal learner would, while keeping its rank order of targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/bayesian_concepts_log_odds_distortion.py`

### set_specific_yes_criterion — rank 9, 56.3 ± 17.4 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.4

**Hypothesis:** People judge by the same simple rule matching as the best current model (saying yes to the degree that the target has the arithmetic properties all the examples share), but the decision rule differs: on seeing each example set, a person settles on a response criterion for that set, a set-specific bias toward answering yes or no, which varies from set to set by an inferred amount and is then applied to every target asked about with that set. So sets differ in overall generalization breadth well beyond what their shared properties explain, while the rank order of targets within a set still follows rule matching.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/set_specific_yes_criterion.py`

### exemplar_feature_similarity_gcm — rank 10, 56.5 ± 14.2 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.6

**Hypothesis:** People do not induce a rule; they judge the target by its exemplar similarity to the individual examples shown (as in the Generalized Context Model). Each number is represented on two psychological dimensions — its magnitude and its arithmetic features (parity, divisibility by 3/4/5/10, final digit, tens digit, being prime, square or a power of 2) — and similarity to each example decays exponentially with attention-weighted distance on these dimensions; the probability of "yes" grows with the summed similarity to the examples, so generalization is graded and does not sharpen with the size principle.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity_gcm.py`

### shared_feature_attention_gcm — rank 11, 59.6 ± 14.2 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1723.6

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`: people still judge the target by its summed exemplar similarity to the examples on magnitude and arithmetic features, but their attention to each arithmetic feature is drawn by the examples themselves — a feature on which all the examples agree (e.g. all even, all powers of 2, all ending in 5) receives extra attention, by an inferred amount, relative to features on which the examples disagree. The one change is this example-driven attention shift, which makes a target that breaks a property shared by every example much less similar to them, so rule-consistent targets are accepted (even far away) and rule-violating ones rejected more sharply than with fixed attention.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/shared_feature_attention_gcm.py`

### range_extension_by_spacing — rank 12, 84.7 ± 16.3 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1748.8

**Hypothesis:** People ignore arithmetic rules and generalize by range extension: they treat every number between the smallest and largest example as equally likely to be produced, and their willingness to say yes falls off smoothly with how far a target lies beyond that range, measured in units of the typical spacing between the examples (so tightly packed, many-example sets generalize only a little past their edges, while a lone example generalizes over a learned width). This disagrees most sharply with the rule-based Bayesian model on rule-consistent targets far outside the example range (e.g. 64 after 4,16,12), which it predicts are rejected, and on in-range rule violators, which it predicts are accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/range_extension_by_spacing.py`

### bigelow_piantadosi_inferred_noise — rank 13, 95.8 ± 22.6 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1759.9

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 42 rules (with a prior weight inferred for each) and every interval of 1-100, with the size principle, but people believe a much larger and unknown share of the examples are random draws from 1-100 rather than fixed at 10%; that noise share is inferred. Treating examples as more often noise lets one stray example be explained away, so the learner commits to tighter concepts around the bulk of rich example sets (less generalization outside their range) while staying graded rather than near-certain in its judgments.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_inferred_noise.py`

### size_weighted_interval_learning — rank 14, 100.4 ± 26.8 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1764.4

**Hypothesis:** Refining `bigelow_piantadosi_interval` (Bayesian concept learning over 42 mathematical rules and every interval of 1-100, with per-rule prior weights and a Beta prior over interval midpoints), with one change: the strength of the size principle is inferred rather than fixed, so a learner may treat each example as stronger (or weaker) evidence than one random draw from the concept. If people apply a stronger size principle than the original model assumes, rich example sets should pull their belief onto the tightest concepts that cover the examples, and they should say "yes" less often to numbers outside the examples' range, which is the gap the critique found for sets of 3-4 examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/size_weighted_interval_learning.py`

### inferred_noise_concept_learning — rank 15, 105.1 ± 29.7 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1769.2

**Hypothesis:** A refinement of bigelow_piantadosi_independent: the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with the size principle, but the share of examples the learner believes are random noise (rather than drawn from the concept) is inferred from the data instead of fixed at 10%. If people treat examples as rarely noisy, they refuse to discard an awkward example in favour of a broad rule like "even", so rich, rule-less example sets are explained by the interval spanning them and numbers outside their range are rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/inferred_noise_concept_learning.py`

### win_stay_lose_shift_concept — rank 16, 132.8 ± 30.2 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1796.9

**Hypothesis:** People do not keep a full posterior over concepts; they hold a single concept at a time (win-stay, lose-shift). They pick a concept from the first example, keep it as long as each new example fits it, and only when an example contradicts it do they shift to a new concept chosen in light of all examples seen so far; they then say "yes" exactly when the target belongs to the concept they currently hold. Because consistent examples never make the learner narrow down to a tighter concept, judgments are all-or-none rule applications that do not sharpen with more examples the way the size principle predicts, and they depend on the order in which the examples are read.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/win_stay_lose_shift_concept.py`

### order_weighted_concept_learning — rank 17, 252.6 ± 49.2 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1916.6

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical intervals, with the size principle, but they do not weigh the examples equally: how much each example shifts their belief depends on where it came in the order shown, growing or shrinking steadily from first to last (primacy or recency), while the total evidence stays the same. So the same numbers shown in a different order lead to different generalizations, favouring the concepts that fit the most heavily weighted examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/order_weighted_concept_learning.py`

### bigelow_piantadosi_independent — rank 18, 274.1 ± 47.0 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1938.1

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### bigelow_piantadosi_interval — rank 19, 306.1 ± 48.6 nats behind the best (6.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1970.2

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_interval.py`

### tenenbaum_rules_similarity — rank 20, 380.6 ± 52.5 nats behind the best (7.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2044.7

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
