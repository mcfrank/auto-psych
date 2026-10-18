# Tried before

34 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### exemplar_feature_similarity_gcm — pruned (experiment1 end of experiment)

**Outcome:** 60.3 nats behind digit_similarity_with_common_factor (3.8× dse)

**Hypothesis:** People do not induce a rule; they judge the target by its exemplar similarity to the individual examples shown (as in the Generalized Context Model). Each number is represented on two psychological dimensions — its magnitude and its arithmetic features (parity, divisibility by 3/4/5/10, final digit, tens digit, being prime, square or a power of 2) — and similarity to each example decays exponentially with attention-weighted distance on these dimensions; the probability of "yes" grows with the summed similarity to the examples, so generalization is graded and does not sharpen with the size principle.

### range_extension_by_spacing — pruned (experiment1 end of experiment)

**Outcome:** 88.5 nats behind digit_similarity_with_common_factor (5.0× dse)

**Hypothesis:** People ignore arithmetic rules and generalize by range extension: they treat every number between the smallest and largest example as equally likely to be produced, and their willingness to say yes falls off smoothly with how far a target lies beyond that range, measured in units of the typical spacing between the examples (so tightly packed, many-example sets generalize only a little past their edges, while a lone example generalizes over a learned width). This disagrees most sharply with the rule-based Bayesian model on rule-consistent targets far outside the example range (e.g. 64 after 4,16,12), which it predicts are rejected, and on in-range rule violators, which it predicts are accepted.

### order_weighted_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 256.3 nats behind digit_similarity_with_common_factor (5.4× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical intervals, with the size principle, but they do not weigh the examples equally: how much each example shifts their belief depends on where it came in the order shown, growing or shrinking steadily from first to last (primacy or recency), while the total evidence stays the same. So the same numbers shown in a different order lead to different generalizations, favouring the concepts that fit the most heavily weighted examples.

### bigelow_piantadosi_inferred_noise — pruned (experiment1 end of experiment)

**Outcome:** 99.6 nats behind digit_similarity_with_common_factor (4.6× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 42 rules (with a prior weight inferred for each) and every interval of 1-100, with the size principle, but people believe a much larger and unknown share of the examples are random draws from 1-100 rather than fixed at 10%; that noise share is inferred. Treating examples as more often noise lets one stray example be explained away, so the learner commits to tighter concepts around the bulk of rich example sets (less generalization outside their range) while staying graded rather than near-certain in its judgments.

### inferred_noise_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 108.9 nats behind digit_similarity_with_common_factor (3.9× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with the size principle, but the share of examples the learner believes are random noise (rather than drawn from the concept) is inferred from the data instead of fixed at 10%. If people treat examples as rarely noisy, they refuse to discard an awkward example in favour of a broad rule like "even", so rich, rule-less example sets are explained by the interval spanning them and numbers outside their range are rejected.

### size_weighted_interval_learning — pruned (experiment1 end of experiment)

**Outcome:** 104.1 nats behind digit_similarity_with_common_factor (4.2× dse)

**Hypothesis:** Refining `bigelow_piantadosi_interval` (Bayesian concept learning over 42 mathematical rules and every interval of 1-100, with per-rule prior weights and a Beta prior over interval midpoints), with one change: the strength of the size principle is inferred rather than fixed, so a learner may treat each example as stronger (or weaker) evidence than one random draw from the concept. If people apply a stronger size principle than the original model assumes, rich example sets should pull their belief onto the tightest concepts that cover the examples, and they should say "yes" less often to numbers outside the examples' range, which is the gap the critique found for sets of 3-4 examples.

### win_stay_lose_shift_concept — pruned (experiment1 end of experiment)

**Outcome:** 136.6 nats behind digit_similarity_with_common_factor (4.9× dse)

**Hypothesis:** People do not keep a full posterior over concepts; they hold a single concept at a time (win-stay, lose-shift). They pick a concept from the first example, keep it as long as each new example fits it, and only when an example contradicts it do they shift to a new concept chosen in light of all examples seen so far; they then say "yes" exactly when the target belongs to the concept they currently hold. Because consistent examples never make the learner narrow down to a tighter concept, judgments are all-or-none rule applications that do not sharpen with more examples the way the size principle predicts, and they depend on the order in which the examples are read.

### bayesian_concepts_log_odds_distortion — pruned (experiment1 end of experiment)

**Outcome:** 59.9 nats behind digit_similarity_with_common_factor (3.2× dse)

**Hypothesis:** People carry out normative Bayesian concept learning over mathematical rules and numerical intervals with the size principle, but they distort the resulting belief that the target belongs to the concept before answering: like the linear-in-log-odds probability weighting seen in risky choice, their subjective confidence is a compressed (or expanded) and shifted version of the posterior, measured in log-odds. So they say "yes" to rule-consistent targets far from the examples and "no" to targets the posterior strongly excludes less extremely than the ideal learner would, while keeping its rank order of targets.

### shared_property_rule_matching — pruned (experiment1 end of experiment)

**Outcome:** 14.3 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 13

**Hypothesis:** People ignore magnitude and similarity entirely and judge by rule matching: they note which arithmetic properties all the examples have in common (even, odd, multiple of 3/4/5/10, a shared final digit, prime, square, power of 2) and say "yes" to the degree that the target has those same properties, regardless of how far it lies from the examples. So rule-consistent targets are accepted even when numerically distant, rule violators are rejected even when close, and a single example (which carries all its own properties) generalizes as readily as several.

### nearest_exemplar_feature_similarity — rejected (experiment1 round 1 candidate 3 refine incumbent exemplar_feature_similarity_gcm)

**Outcome:** MCMC did not converge (max R-hat 1.375 > 1.05; min bulk ESS 9 < 100), too far from converging for smaller NUTS steps to help (the fitter refits only a near miss — at most 2% divergent transitions, R-hat <= 1.2, bulk ESS >= 20 — at target_accept 0.99), so raising target_accept will not help. Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`. It keeps the same exemplar representation: magnitude plus arithmetic features, with similarity falling off exponentially with attention-weighted distance. The one change is that people judge the target by its similarity to the single most similar example shown (nearest-neighbour retrieval), not by its similarity summed over all the examples. So adding examples does not inflate "yes" for every target. A lone example generalizes as broadly as a rich set, and a target that closely matches any one example on its arithmetic features is accepted however far it is from the others. This addresses the critique's finding that the summed model under-generalizes from single examples and from numerically distant targets.

### shared_feature_attention_gcm — pruned (experiment1 end of experiment)

**Outcome:** 63.4 nats behind digit_similarity_with_common_factor (4.0× dse)

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`: people still judge the target by its summed exemplar similarity to the examples on magnitude and arithmetic features, but their attention to each arithmetic feature is drawn by the examples themselves — a feature on which all the examples agree (e.g. all even, all powers of 2, all ending in 5) receives extra attention, by an inferred amount, relative to features on which the examples disagree. The one change is this example-driven attention shift, which makes a target that breaks a property shared by every example much less similar to them, so rule-consistent targets are accepted (even far away) and rule-violating ones rejected more sharply than with fixed attention.

### erlang_size_noisy_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 59.9 nats behind digit_similarity_with_common_factor (3.1× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_inferred_noise` (Bayesian concept learning over 42 weighted rules and every interval of 1-100, with the size principle and an inferred share of examples treated as random noise), with one change grafted from `tenenbaum_rules_similarity`: people do not consider all intervals equally plausible a priori but favour intervals of an intermediate, inferred typical size (an Erlang prior on interval size). Because tiny intervals are a priori implausible, a single example is generalized to a broad neighbourhood rather than collapsing onto itself, whereas several examples still pin the concept down by the size principle — so people say yes more after one example than after several, and accept distant numbers more often than graded similarity predicts.

### nearest_exemplar_luce_similarity — pruned (experiment1 end of experiment)

**Outcome:** 38.8 nats behind digit_similarity_with_common_factor (2.5× dse)

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`. It keeps the same exemplar representation: magnitude plus arithmetic features, with similarity falling off exponentially with distance weighted separately on each. The one change is that people judge the target by its similarity to the single most similar example shown (nearest-neighbour retrieval), compared against a fixed criterion (Luce's ratio rule), rather than by similarity summed over all examples. So adding examples does not inflate "yes" for every target. A lone example generalizes as broadly as a rich set, and a target that closely matches any one example on its arithmetic features is accepted however far it lies from the others, which addresses the critique's finding that the summed model under-generalizes from single examples and to numerically distant targets.

### sequential_delta_rule_property_learning — pruned (experiment1 end of experiment)

**Outcome:** 24.0 nats behind digit_similarity_with_common_factor (2.5× dse)

**Hypothesis:** People build their guess about the program's rule incrementally, in the order the examples are shown: the first example suggests a set of arithmetic properties (parity, divisibility by 3/4/5/10, final digit, prime, square, power of 2), and each later example revises the strength of belief in each property by an error-correcting (delta-rule) update at an inferred learning rate, so that a slow learner stays anchored on the first example (primacy) and a fast learner is dominated by the last one (recency). They then say "yes" to the degree that the target has the properties their final, order-dependent guess favours, regardless of its magnitude.

### set_specific_yes_criterion — pruned (experiment1 end of experiment)

**Outcome:** 60.1 nats behind digit_similarity_with_common_factor (3.6× dse)

**Hypothesis:** People judge by the same simple rule matching as the best current model (saying yes to the degree that the target has the arithmetic properties all the examples share), but the decision rule differs: on seeing each example set, a person settles on a response criterion for that set, a set-specific bias toward answering yes or no, which varies from set to set by an inferred amount and is then applied to every target asked about with that set. So sets differ in overall generalization breadth well beyond what their shared properties explain, while the rank order of targets within a set still follows rule matching.

### rule_matching_with_range_extension — pruned (experiment1 end of experiment)

**Outcome:** 18.4 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, but they also treat the span of the examples (from smallest to largest) as part of what the program produces, so a target inside that span gets an extra, inferred boost to "yes" that fades smoothly with how far the target lies beyond the span's edges (for a single example, with its distance from that example). The one change is this range-membership component grafted from `range_extension_by_spacing`, which addresses the critique that the rule model misses interval-based generalization: in-range rule violators and in-range targets of rule-less sets are accepted more than distant ones.

### rule_matching_with_range_fill — pruned (experiment1 end of experiment)

**Outcome:** 19.2 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, regardless of its distance from them, but they also treat the span between the smallest and largest example as territory the program covers, so a target lying inside that range gets an extra, inferred boost toward "yes" (the one change, a range-membership component taken from `range_extension_by_spacing`). So, as the critique found, in-range targets are accepted more often than rule matching alone predicts, while rule-consistent targets far outside the range are still accepted.

### span_filled_nearest_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 40.2 nats behind digit_similarity_with_common_factor (2.6× dse)

**Hypothesis:** A refinement of `nearest_exemplar_luce_similarity`: people still judge the target by its similarity to the most similar example on arithmetic features, combined with magnitude similarity and compared against a Luce criterion, but the one change is that on the magnitude dimension they fill in the gaps between the examples, treating the whole span from the smallest to the largest example as already "seen", so magnitude dissimilarity is the target's distance to that span (zero anywhere inside it) rather than to the nearest single example. Targets inside the examples' range are therefore accepted more readily than equally feature-similar targets outside it, addressing the critique that the incumbent misses range-based generalization.

### arithmetic_progression_step_inference — pruned (experiment1 end of experiment)

**Outcome:** 61.5 nats behind digit_similarity_with_common_factor (4.5× dse)

**Hypothesis:** People infer that the program counts in fixed steps: the concepts they consider are arithmetic progressions through 1-100 (all numbers with the same remainder after dividing by some step size, e.g. counting by 2s, 5s, 10s or 7s from the examples), preferring small step sizes a priori but, by the size principle, favouring the largest step that all the examples fit. They say "yes" with the posterior probability that the target lies on the inferred progression, so targets congruent with the examples are accepted however far away they are, and acceptance is all-or-none per step rather than graded by final-digit resemblance.

### prototype_abstraction_similarity — pruned (experiment1 end of experiment)

**Outcome:** 11.9 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 13

**Hypothesis:** People do not store the individual examples or infer a rule; they abstract a single prototype of the set — its average magnitude and the proportion of examples having each arithmetic property (even, multiple of 3/4/5/10, same final digit, prime, square, power of 2) — and say "yes" to the degree that the target is similar to that prototype, with similarity decaying exponentially with the attention-weighted distance between the target's own properties and the prototype's averaged ones. So a property shared by every example acts as a strong expectation, a property split among the examples barely matters, and targets near the set's centre of magnitude (rather than near any one example) are favoured.

### relational_step_extrapolation — pruned (experiment1 end of experiment)

**Outcome:** 71.4 nats behind digit_similarity_with_common_factor (4.0× dse)

**Hypothesis:** People do not look for shared properties or similarity; they infer the generating operation from the relations among the examples themselves (relational, analogical extrapolation). They notice the differences and whole-number ratios linking pairs of examples (e.g. 16 is 4 times 4, 83 is 15 below 98) and say "yes" to a target that the same step produces from one of the examples (64 after 4,16,12; 68 after 98,83), and "no" otherwise, so a lone example, which carries no relation, gives no basis for extrapolation.

### salience_weighted_rule_matching — pruned (experiment1 end of experiment)

**Outcome:** 18.1 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still note which arithmetic properties all the examples share (even, odd, multiple of 3/4/5/10, prime, square, power of 2, a common final digit) and say "yes" to the degree that the target has those same properties, regardless of its magnitude, but the one change is that these properties are not equally salient — each kind of property carries its own inferred weight, so whether the target matches a highly salient shared property (e.g. parity or being a power of 2) matters far more than whether it matches a weak one. This makes judgments sharper across items than equal-weight rule matching (the critique's excess item and set variance) and lets a rule such as "multiples of 10" or "powers of 2" decide the answer even when the target shares no final digit with the examples.

### place_value_digit_similarity — rejected (experiment1 round 4 candidate 1 lens 2)

**Outcome:** predicts like existing model 'place_value_grid_similarity' (p_yes RMSE 0.00015 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of place_value_grid_similarity, not a new hypothesis.

**Hypothesis:** People represent each number by its decimal place values — a tens digit and a units digit — and judge the target by its digit-wise similarity to each example: similarity is the product of how alike the units digits are (identical most, same parity partly, opposite parity least) and how alike the tens digits are (falling off with the number of decades between them). They say "yes" in proportion to the target's average place-value similarity to the examples, so a target sharing an example's final digit is accepted strongly when it is in the same or a neighbouring decade but much less when it lies many decades away.

### proximity_weighted_digit_similarity — rejected (experiment1 round 4 candidate 4 refine incumbent final_digit_similarity_gradient)

**Outcome:** predicts like existing model 'proximity_cued_digit_retrieval' (p_yes RMSE 0.00114 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of proximity_cued_digit_retrieval, not a new hypothesis.

**Hypothesis:** A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but the one change is that each example's digit resemblance is weighted by how numerically close that example is to the target, so a shared ending counts fully when the matching example is nearby and only partly (down to an inferred floor) when it is far away. This makes magnitude modulate digit similarity example by example rather than adding a separate range bonus, addressing the critique that digit-matched targets near an example are accepted more than digit-matched targets far from every example.

### proximity_gated_digit_similarity — rejected (experiment1 round 4 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'proximity_cued_digit_retrieval' (p_yes RMSE 0.00139 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of proximity_cued_digit_retrieval, not a new hypothesis.

**Hypothesis:** A refinement of `digit_similarity_with_range_extension`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but the one change is how magnitude enters — instead of a separate, digit-independent boost for lying inside the examples' span, each example's final-digit resemblance counts for more the closer that example is numerically to the target (its weight fading smoothly with distance down to an inferred floor). So a digit match with a nearby example drives "yes" far more than the same match with a distant one, which addresses the critique that numerical proximity boosts digit-matched targets.

### common_divisor_digit_similarity — rejected (experiment1 round 4 candidate 4 refine incumbent final_digit_similarity_gradient repair 1)

**Outcome:** predicts like existing model 'digit_similarity_with_common_factor' (p_yes RMSE 0.00049 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of digit_similarity_with_common_factor, not a new hypothesis.

**Hypothesis:** A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but the one change is that they also notice when all the examples are multiples of a common factor of 3 or more (their greatest common divisor) and give an extra, inferred boost toward "yes" to targets that are multiples of that same factor. This addresses the critique that, among parity-matched targets, people accept multiples of the examples' common divisor more than the final-digit model can represent (e.g. 36 after 12,24 or 48 after 16), so counting-by-a-factor generalization operates alongside, not instead of, digit resemblance.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 384.4 nats behind digit_similarity_with_common_factor (7.6× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 277.8 nats behind digit_similarity_with_common_factor (6.2× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Outcome:** 309.9 nats behind digit_similarity_with_common_factor (6.6× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### drifting_yes_criterion_digit_similarity — rejected (experiment2 round 1 candidate 1 lens 8)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** People judge the target by the simple final-digit similarity used in the simplest current model (same final digit most similar, same parity partly, opposite parity least), but their decision criterion is not fixed over the session: as the session goes on and they see more and more of the program's output without feedback, their bias toward answering "yes" shifts steadily, so the same (set, target) pair is accepted more (or less) often the later in the session it is asked.

### decade_band_span_digit_common_factor — rejected (experiment2 round 1 candidate 3 refine incumbent nearest_decade_digit_common_factor)

**Outcome:** predicts like existing model 'decade_band_digit_common_factor' (p_yes RMSE 0.00166 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of decade_band_digit_common_factor, not a new hypothesis.

**Hypothesis:** A refinement of `nearest_decade_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with an extra boost for multiples of the examples' common factor and a leading-digit (decade) familiarity boost that decays with each decade of separation. The one change is that people code the examples' magnitudes coarsely by decade and treat the whole band of decades the examples span (from the lowest example's decade to the highest's) as covered, so the decade boost is full for any target whose decade lies inside that band and decays only with how many decades the target lies beyond it, rather than with its distance to the single nearest example's decade. This addresses the critiques that the incumbent under-predicts "yes" for targets inside the examples' range and over-predicts it for targets just across a decade boundary outside the examples.

### span_filled_decade_digit_factor — rejected (experiment2 round 1 candidate 4 refine incumbent nearest_decade_digit_common_factor)

**Outcome:** predicts like existing model 'decade_band_digit_common_factor' (p_yes RMSE 0.00166 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of decade_band_digit_common_factor, not a new hypothesis.

**Hypothesis:** A refinement of `nearest_decade_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with a boost for multiples of the examples' common factor and a leading-digit (decade) familiarity boost that decays with each decade of separation. The one change is that people fill in the decades lying between the smallest and largest example, treating that whole band of decades as familiar territory, so the decade boost is full for any target whose decade lies within the examples' decade span and decays only with how many decades the target lies beyond that span, rather than with its distance to the nearest single example's decade. This addresses the critique that the incumbent under-predicts "yes" for targets inside the examples' range relative to targets outside it.

### decade_span_coverage_digit_factor — rejected (experiment2 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'decade_band_digit_common_factor' (p_yes RMSE 0.00166 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of decade_band_digit_common_factor, not a new hypothesis.

**Hypothesis:** A refinement of `digit_similarity_with_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with an extra boost for multiples of the examples' common factor (3 or more). The one change is that people code magnitude coarsely by decade and treat the whole block of decades from the examples' lowest to their highest decade as territory the program covers: a target whose tens digit falls anywhere in that decade span (even in a decade no example occupies) gets the full magnitude boost, and the boost fades with each decade the target lies beyond the span's edge. This addresses the critique that people accept in-range targets more than out-of-range ones (interpolation the nearest-decade incumbent misses) while respecting decade boundaries just outside the span (29 after 31 is not treated as nearby).

### iter1_candidate1 — rejected (experiment2 round 1 candidate 1 lens 8 repair 1)

**Outcome:** predicts like existing model 'final_digit_similarity_gradient' (p_yes RMSE 0.00164 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of final_digit_similarity_gradient, not a new hypothesis.

**Hypothesis:** People judge the target by the simple final-digit similarity used in the simplest current model (same final digit most similar, same parity partly, opposite parity least), but they do not answer in graded proportion to it: they apply a firm criterion, answering "yes" almost deterministically when the target's average similarity to the examples clears it and "no" when it falls short, except on an occasional fraction of trials where attention lapses and they answer by a coin flip. This all-or-none decision rule with lapses predicts more polarized item-level yes rates than a graded similarity response.
