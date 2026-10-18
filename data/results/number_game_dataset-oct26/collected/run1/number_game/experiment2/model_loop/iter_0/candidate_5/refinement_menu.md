# Refinement menu

The models you may refine, other than the incumbent `digit_similarity_with_common_factor`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### common_factor_digit_range_similarity — rank 1, 14.3 ± 13.8 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3222.0

**Hypothesis:** A refinement of `digit_similarity_with_range_extension`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), with an extra boost for targets inside the examples' span that fades beyond its edges, but the one change is that they also notice when several examples are all multiples of a common factor of 3 or more (their greatest common divisor, e.g. 8 for 16, 8, 64) and infer a "multiples of that factor" rule, giving an inferred extra push toward "yes" to targets that are multiples of it. This addresses the critique that, among parity-matched targets of sets sharing a common divisor, people accept divisible targets more than the digit model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/common_factor_digit_range_similarity.py`

### digit_similarity_with_range_extension — rank 2, 36.1 ± 16.8 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3243.8

**Hypothesis:** A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but they also treat the span from the smallest to the largest example as territory the program covers, so a target inside that span gets an extra, inferred boost toward "yes" that fades smoothly with how far it lies beyond the span's edges (for a single example, with its distance from that example). The one change is this range-membership component grafted from `range_extension_by_spacing`, addressing the critique that the magnitude-blind digit model under-predicts yes for in-range versus out-of-range targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/digit_similarity_with_range_extension.py`

### place_value_grid_similarity — rank 3, 36.7 ± 15.0 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3244.4

**Hypothesis:** People represent each two-digit number by its place values, as a position on a tens-digit by units-digit grid, and judge the target by its similarity to each example on that grid: similarity is highest for the same final digit, partial for a different final digit of the same parity, and lowest for opposite parity, and it is multiplied down by how many decades apart the target and that example lie. They say "yes" in proportion to the target's average place-value similarity to the examples, so a final-digit match earns credit mainly when it falls in the same or a nearby decade as the matching example. This disagrees most sharply with the magnitude-blind final-digit model on digit-matched targets several decades away from every matching example (e.g. 93 after 13), which it predicts are rejected far more often than nearby digit-matched targets (e.g. 23).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/place_value_grid_similarity.py`

### proximity_cued_digit_retrieval — rank 4, 38.4 ± 14.8 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3246.1

**Hypothesis:** People compare the target's final digit with the final digits of the examples (same digit most similar, same parity partly, opposite parity least), but the target acts as a retrieval cue: examples close to it in magnitude come to mind readily, while numerically distant examples are only sometimes recalled for the comparison. So "yes" grows with the final-digit similarity to the examples that are actually retrieved, which makes a digit match with a nearby example count for more than the same match with a far-away one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/proximity_cued_digit_retrieval.py`

### digit_similarity_with_span_falloff — rank 5, 45.2 ± 15.9 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3252.9

**Hypothesis:** A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but the one change is that they also treat the span from the smallest to the largest example as covered by the program, so their willingness to say "yes" falls off steadily with how far the target lies beyond that span (with no penalty anywhere inside it). This addresses the critique that people accept targets inside the examples' range more than the magnitude-blind digit model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/digit_similarity_with_span_falloff.py`

### final_digit_similarity_gradient — rank 6, 61.4 ± 13.4 nats behind the best (4.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3269.1

**Hypothesis:** People ignore magnitude and judge the target by how much its final digit resembles the final digits of the examples: a target ending in the same digit as an example is maximally similar to it, one ending in a different digit of the same parity (odd/even) is partly similar, and one of the opposite parity is least similar. They say "yes" in proportion to the target's average final-digit similarity to all the examples shown, so the answer is graded by the share of examples it resembles rather than by any rule that every example must obey, and numerically close targets get no credit for closeness.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/final_digit_similarity_gradient.py`

### serial_position_digit_similarity — rank 7, 63.0 ± 13.3 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3270.7

**Hypothesis:** People compare the target's final digit with the final digits of the examples (same digit most similar, same parity partly, opposite parity least), but they do not hold every example equally strongly in memory: how much each example counts in the comparison depends on its serial position in the order shown, changing steadily from first to last (recency if later examples are held more strongly, primacy if earlier ones are). So "yes" grows with a position-weighted average of final-digit similarity, and the same numbers shown in a different order change which digit matches drive acceptance.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/serial_position_digit_similarity.py`

## Pruned models (out of the set; narrowest margin first)

### prototype_abstraction_similarity — pruned (experiment1 end of experiment)

**Margin:** 11.9 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 13

**Hypothesis:** People do not store the individual examples or infer a rule; they abstract a single prototype of the set — its average magnitude and the proportion of examples having each arithmetic property (even, multiple of 3/4/5/10, same final digit, prime, square, power of 2) — and say "yes" to the degree that the target is similar to that prototype, with similarity decaying exponentially with the attention-weighted distance between the target's own properties and the prototype's averaged ones. So a property shared by every example acts as a strong expectation, a property split among the examples barely matters, and targets near the set's centre of magnitude (rather than near any one example) are favoured.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/prototype_abstraction_similarity.py`

### shared_property_rule_matching — pruned (experiment1 end of experiment)

**Margin:** 14.3 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 13

**Hypothesis:** People ignore magnitude and similarity entirely and judge by rule matching: they note which arithmetic properties all the examples have in common (even, odd, multiple of 3/4/5/10, a shared final digit, prime, square, power of 2) and say "yes" to the degree that the target has those same properties, regardless of how far it lies from the examples. So rule-consistent targets are accepted even when numerically distant, rule violators are rejected even when close, and a single example (which carries all its own properties) generalizes as readily as several.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_rule_matching.py`

### salience_weighted_rule_matching — pruned (experiment1 end of experiment)

**Margin:** 18.1 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still note which arithmetic properties all the examples share (even, odd, multiple of 3/4/5/10, prime, square, power of 2, a common final digit) and say "yes" to the degree that the target has those same properties, regardless of its magnitude, but the one change is that these properties are not equally salient — each kind of property carries its own inferred weight, so whether the target matches a highly salient shared property (e.g. parity or being a power of 2) matters far more than whether it matches a weak one. This makes judgments sharper across items than equal-weight rule matching (the critique's excess item and set variance) and lets a rule such as "multiples of 10" or "powers of 2" decide the answer even when the target shares no final digit with the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/salience_weighted_rule_matching.py`

### rule_matching_with_range_extension — pruned (experiment1 end of experiment)

**Margin:** 18.4 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, but they also treat the span of the examples (from smallest to largest) as part of what the program produces, so a target inside that span gets an extra, inferred boost to "yes" that fades smoothly with how far the target lies beyond the span's edges (for a single example, with its distance from that example). The one change is this range-membership component grafted from `range_extension_by_spacing`, which addresses the critique that the rule model misses interval-based generalization: in-range rule violators and in-range targets of rule-less sets are accepted more than distant ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_matching_with_range_extension.py`

### rule_matching_with_range_fill — pruned (experiment1 end of experiment)

**Margin:** 19.2 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, regardless of its distance from them, but they also treat the span between the smallest and largest example as territory the program covers, so a target lying inside that range gets an extra, inferred boost toward "yes" (the one change, a range-membership component taken from `range_extension_by_spacing`). So, as the critique found, in-range targets are accepted more often than rule matching alone predicts, while rule-consistent targets far outside the range are still accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_matching_with_range_fill.py`

### sequential_delta_rule_property_learning — pruned (experiment1 end of experiment)

**Margin:** 24.0 nats behind digit_similarity_with_common_factor (2.5× dse)

**Hypothesis:** People build their guess about the program's rule incrementally, in the order the examples are shown: the first example suggests a set of arithmetic properties (parity, divisibility by 3/4/5/10, final digit, prime, square, power of 2), and each later example revises the strength of belief in each property by an error-correcting (delta-rule) update at an inferred learning rate, so that a slow learner stays anchored on the first example (primacy) and a fast learner is dominated by the last one (recency). They then say "yes" to the degree that the target has the properties their final, order-dependent guess favours, regardless of its magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/sequential_delta_rule_property_learning.py`

### nearest_exemplar_luce_similarity — pruned (experiment1 end of experiment)

**Margin:** 38.8 nats behind digit_similarity_with_common_factor (2.5× dse)

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`. It keeps the same exemplar representation: magnitude plus arithmetic features, with similarity falling off exponentially with distance weighted separately on each. The one change is that people judge the target by its similarity to the single most similar example shown (nearest-neighbour retrieval), compared against a fixed criterion (Luce's ratio rule), rather than by similarity summed over all examples. So adding examples does not inflate "yes" for every target. A lone example generalizes as broadly as a rich set, and a target that closely matches any one example on its arithmetic features is accepted however far it lies from the others, which addresses the critique's finding that the summed model under-generalizes from single examples and to numerically distant targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/nearest_exemplar_luce_similarity.py`

### span_filled_nearest_exemplar — pruned (experiment1 end of experiment)

**Margin:** 40.2 nats behind digit_similarity_with_common_factor (2.6× dse)

**Hypothesis:** A refinement of `nearest_exemplar_luce_similarity`: people still judge the target by its similarity to the most similar example on arithmetic features, combined with magnitude similarity and compared against a Luce criterion, but the one change is that on the magnitude dimension they fill in the gaps between the examples, treating the whole span from the smallest to the largest example as already "seen", so magnitude dissimilarity is the target's distance to that span (zero anywhere inside it) rather than to the nearest single example. Targets inside the examples' range are therefore accepted more readily than equally feature-similar targets outside it, addressing the critique that the incumbent misses range-based generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/span_filled_nearest_exemplar.py`

### bayesian_concepts_log_odds_distortion — pruned (experiment1 end of experiment)

**Margin:** 59.9 nats behind digit_similarity_with_common_factor (3.2× dse)

**Hypothesis:** People carry out normative Bayesian concept learning over mathematical rules and numerical intervals with the size principle, but they distort the resulting belief that the target belongs to the concept before answering: like the linear-in-log-odds probability weighting seen in risky choice, their subjective confidence is a compressed (or expanded) and shifted version of the posterior, measured in log-odds. So they say "yes" to rule-consistent targets far from the examples and "no" to targets the posterior strongly excludes less extremely than the ideal learner would, while keeping its rank order of targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bayesian_concepts_log_odds_distortion.py`

### erlang_size_noisy_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 59.9 nats behind digit_similarity_with_common_factor (3.1× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_inferred_noise` (Bayesian concept learning over 42 weighted rules and every interval of 1-100, with the size principle and an inferred share of examples treated as random noise), with one change grafted from `tenenbaum_rules_similarity`: people do not consider all intervals equally plausible a priori but favour intervals of an intermediate, inferred typical size (an Erlang prior on interval size). Because tiny intervals are a priori implausible, a single example is generalized to a broad neighbourhood rather than collapsing onto itself, whereas several examples still pin the concept down by the size principle — so people say yes more after one example than after several, and accept distant numbers more often than graded similarity predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_size_noisy_concept_learning.py`

### set_specific_yes_criterion — pruned (experiment1 end of experiment)

**Margin:** 60.1 nats behind digit_similarity_with_common_factor (3.6× dse)

**Hypothesis:** People judge by the same simple rule matching as the best current model (saying yes to the degree that the target has the arithmetic properties all the examples share), but the decision rule differs: on seeing each example set, a person settles on a response criterion for that set, a set-specific bias toward answering yes or no, which varies from set to set by an inferred amount and is then applied to every target asked about with that set. So sets differ in overall generalization breadth well beyond what their shared properties explain, while the rank order of targets within a set still follows rule matching.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/set_specific_yes_criterion.py`

### exemplar_feature_similarity_gcm — pruned (experiment1 end of experiment)

**Margin:** 60.3 nats behind digit_similarity_with_common_factor (3.8× dse)

**Hypothesis:** People do not induce a rule; they judge the target by its exemplar similarity to the individual examples shown (as in the Generalized Context Model). Each number is represented on two psychological dimensions — its magnitude and its arithmetic features (parity, divisibility by 3/4/5/10, final digit, tens digit, being prime, square or a power of 2) — and similarity to each example decays exponentially with attention-weighted distance on these dimensions; the probability of "yes" grows with the summed similarity to the examples, so generalization is graded and does not sharpen with the size principle.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_feature_similarity_gcm.py`

### arithmetic_progression_step_inference — pruned (experiment1 end of experiment)

**Margin:** 61.5 nats behind digit_similarity_with_common_factor (4.5× dse)

**Hypothesis:** People infer that the program counts in fixed steps: the concepts they consider are arithmetic progressions through 1-100 (all numbers with the same remainder after dividing by some step size, e.g. counting by 2s, 5s, 10s or 7s from the examples), preferring small step sizes a priori but, by the size principle, favouring the largest step that all the examples fit. They say "yes" with the posterior probability that the target lies on the inferred progression, so targets congruent with the examples are accepted however far away they are, and acceptance is all-or-none per step rather than graded by final-digit resemblance.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arithmetic_progression_step_inference.py`

### shared_feature_attention_gcm — pruned (experiment1 end of experiment)

**Margin:** 63.4 nats behind digit_similarity_with_common_factor (4.0× dse)

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`: people still judge the target by its summed exemplar similarity to the examples on magnitude and arithmetic features, but their attention to each arithmetic feature is drawn by the examples themselves — a feature on which all the examples agree (e.g. all even, all powers of 2, all ending in 5) receives extra attention, by an inferred amount, relative to features on which the examples disagree. The one change is this example-driven attention shift, which makes a target that breaks a property shared by every example much less similar to them, so rule-consistent targets are accepted (even far away) and rule-violating ones rejected more sharply than with fixed attention.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_feature_attention_gcm.py`

### relational_step_extrapolation — pruned (experiment1 end of experiment)

**Margin:** 71.4 nats behind digit_similarity_with_common_factor (4.0× dse)

**Hypothesis:** People do not look for shared properties or similarity; they infer the generating operation from the relations among the examples themselves (relational, analogical extrapolation). They notice the differences and whole-number ratios linking pairs of examples (e.g. 16 is 4 times 4, 83 is 15 below 98) and say "yes" to a target that the same step produces from one of the examples (64 after 4,16,12; 68 after 98,83), and "no" otherwise, so a lone example, which carries no relation, gives no basis for extrapolation.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/relational_step_extrapolation.py`

### range_extension_by_spacing — pruned (experiment1 end of experiment)

**Margin:** 88.5 nats behind digit_similarity_with_common_factor (5.0× dse)

**Hypothesis:** People ignore arithmetic rules and generalize by range extension: they treat every number between the smallest and largest example as equally likely to be produced, and their willingness to say yes falls off smoothly with how far a target lies beyond that range, measured in units of the typical spacing between the examples (so tightly packed, many-example sets generalize only a little past their edges, while a lone example generalizes over a learned width). This disagrees most sharply with the rule-based Bayesian model on rule-consistent targets far outside the example range (e.g. 64 after 4,16,12), which it predicts are rejected, and on in-range rule violators, which it predicts are accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/range_extension_by_spacing.py`

### bigelow_piantadosi_inferred_noise — pruned (experiment1 end of experiment)

**Margin:** 99.6 nats behind digit_similarity_with_common_factor (4.6× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 42 rules (with a prior weight inferred for each) and every interval of 1-100, with the size principle, but people believe a much larger and unknown share of the examples are random draws from 1-100 rather than fixed at 10%; that noise share is inferred. Treating examples as more often noise lets one stray example be explained away, so the learner commits to tighter concepts around the bulk of rich example sets (less generalization outside their range) while staying graded rather than near-certain in its judgments.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_inferred_noise.py`

### size_weighted_interval_learning — pruned (experiment1 end of experiment)

**Margin:** 104.1 nats behind digit_similarity_with_common_factor (4.2× dse)

**Hypothesis:** Refining `bigelow_piantadosi_interval` (Bayesian concept learning over 42 mathematical rules and every interval of 1-100, with per-rule prior weights and a Beta prior over interval midpoints), with one change: the strength of the size principle is inferred rather than fixed, so a learner may treat each example as stronger (or weaker) evidence than one random draw from the concept. If people apply a stronger size principle than the original model assumes, rich example sets should pull their belief onto the tightest concepts that cover the examples, and they should say "yes" less often to numbers outside the examples' range, which is the gap the critique found for sets of 3-4 examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/size_weighted_interval_learning.py`

### inferred_noise_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 108.9 nats behind digit_similarity_with_common_factor (3.9× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with the size principle, but the share of examples the learner believes are random noise (rather than drawn from the concept) is inferred from the data instead of fixed at 10%. If people treat examples as rarely noisy, they refuse to discard an awkward example in favour of a broad rule like "even", so rich, rule-less example sets are explained by the interval spanning them and numbers outside their range are rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_noise_concept_learning.py`

### win_stay_lose_shift_concept — pruned (experiment1 end of experiment)

**Margin:** 136.6 nats behind digit_similarity_with_common_factor (4.9× dse)

**Hypothesis:** People do not keep a full posterior over concepts; they hold a single concept at a time (win-stay, lose-shift). They pick a concept from the first example, keep it as long as each new example fits it, and only when an example contradicts it do they shift to a new concept chosen in light of all examples seen so far; they then say "yes" exactly when the target belongs to the concept they currently hold. Because consistent examples never make the learner narrow down to a tighter concept, judgments are all-or-none rule applications that do not sharpen with more examples the way the size principle predicts, and they depend on the order in which the examples are read.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_concept.py`

### order_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 256.3 nats behind digit_similarity_with_common_factor (5.4× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical intervals, with the size principle, but they do not weigh the examples equally: how much each example shifts their belief depends on where it came in the order shown, growing or shrinking steadily from first to last (primacy or recency), while the total evidence stays the same. So the same numbers shown in a different order lead to different generalizations, favouring the concepts that fit the most heavily weighted examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/order_weighted_concept_learning.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 277.8 nats behind digit_similarity_with_common_factor (6.2× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Margin:** 309.9 nats behind digit_similarity_with_common_factor (6.6× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_interval.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 384.4 nats behind digit_similarity_with_common_factor (7.6× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
