# Refinement menu

The models you may refine, other than the incumbent `mutual_similarity_contrast_standard`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### rule_gated_decade_familiarity — rank 1, 4.3 ± 6.0 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -4740.7

**Hypothesis:** A refinement of `shared_digit_rule_preempts_parity` (final-digit similarity averaged over the examples, an abstracted "ends in that digit" rule pre-empting parity credit, a common-factor boost and decade-band familiarity all kept). The one change is that magnitude familiarity is a fallback: when people detect an explicit rule linking the examples — they all end in the same digit, or are all multiples of a common factor of 3 or more — the rule pre-empts "numbers near these", and the decade-band familiarity boost is cut by an inferred share; when the set has no detectable rule (a lone example, or examples with no shared digit or factor), people lean on magnitude neighbourhood at full strength. This addresses the critiques that people accept same-decade, different-digit targets after a single example more than predicted, and that generalization breadth differs between sets more than the model produces.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/rule_gated_decade_familiarity.py`

### position_weighted_rule_gated_digits — rank 2, 6.5 ± 6.5 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -4742.9

**Hypothesis:** People judge the target as the current best model says (final-digit similarity to the examples with parity credit pre-empted by a shared final digit, a boost for multiples of the examples' common factor, and decade-band familiarity that an explicit rule pre-empts), but they do not hold every example equally in memory when comparing final digits: each example's weight in the digit comparison changes steadily with its serial position, from the first shown to the last (recency if later examples count more, primacy if the first sets the impression), with the direction and strength inferred. So a target sharing the final digit of the most heavily weighted example is accepted more than one matching an equally similar example in a weaker position, and the same numbers in a different order shift which digit matches drive "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/position_weighted_rule_gated_digits.py`

### ruleless_set_anything_goes_floor — rank 3, 7.4 ± 6.3 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -4743.8

**Hypothesis:** A refinement of `rule_gated_decade_familiarity` (final-digit similarity with rule-pre-empted parity credit, common-factor boost, and decade-band familiarity cut back when an explicit rule is found, all kept). The one change: when two or more examples show no detectable rule (mixed final digits and no common factor of 3 or more), people entertain, with an inferred probability, the hypothesis that the program simply produces any number at all; under that "anything goes" belief they say "yes" regardless of the target, so rule-less multi-example sets get a floor of acceptance that lifts most the targets that resemble the examples least (far, unmatched targets), while lone examples and rule-bearing sets keep the incumbent's judgments. This addresses the critique that people accept far, digit-unmatched targets of rule-less sets more than predicted and that generalization breadth differs between sets more than the model produces.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/ruleless_set_anything_goes_floor.py`

### rule_gated_mixed_parity_criterion — rank 4, 9.2 ± 6.4 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -4745.6

**Hypothesis:** A refinement of `mixed_parity_loosens_criterion` (final-digit similarity with parity credit suppressed by a shared final digit, a common-factor boost, decade-band familiarity, and a looser "yes" criterion when the examples mix odd and even numbers, all kept). The one change, a component taken from `rule_gated_decade_familiarity`, is that magnitude familiarity is only a fallback: when two or more examples obey an explicit rule (all end in the same digit, or all are multiples of a common factor of 3 or more), the rule pre-empts "numbers near these" and the decade-band boost is cut by an inferred share, whereas rule-less sets and lone examples get it in full. Together with the mixed-parity criterion shift, this predicts that rule-less, mixed-parity sets are accepted broadly — even far from the examples' decades — while rule-bound sets are rejected outside their rule, widening the differences between sets that the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/rule_gated_mixed_parity_criterion.py`

### coincidence_graded_rule_preemption — rank 5, 18.4 ± 8.0 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4754.8

**Hypothesis:** A refinement of `rule_gated_decade_familiarity` (final-digit similarity with an "ends in that digit" rule pre-empting parity credit, a common-factor boost, and decade-band familiarity that a detected rule pre-empts, all kept). The one change: rule detection is not all-or-none but graded by how unlikely the examples' shared property would be by coincidence — a shared final digit (1 in 10 per extra example), a common factor g (1 in g per extra example) or merely a shared parity (1 in 2 per extra example), with evidence piling up across examples — and the more suspicious the coincidence, the more firmly people believe in a rule and the more completely it pre-empts "numbers near these" magnitude familiarity. So four examples sharing a digit or a large factor shut off neighbourhood generalization almost entirely, two examples sharing only parity or a factor of 3 barely do, and sets with no shared property at all keep magnitude familiarity in full, spreading generalization breadth across sets more than a binary rule detector does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_graded_rule_preemption.py`

### mixed_parity_loosens_criterion — rank 6, 70.3 ± 13.8 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4806.7

**Hypothesis:** A refinement of `final_digit_rule_suppresses_parity` (final-digit similarity with parity credit suppressed by a shared final digit, common-factor boost and decade-band familiarity all kept). The one change: whether the examples agree in parity sets how broad people think the program is — when the examples mix odd and even numbers, people conclude the program is not bound by any parity or digit rule and lower their criterion, saying "yes" more readily to every target, whereas examples that all share a parity signal a constrained program and keep the criterion strict. This addresses the critique that generalization breadth differs between sets more than the incumbent produces: mixed-parity sets (e.g. 92, 14, 20, 5) are accepted broadly while same-parity sets (e.g. 68, 58) are rejected almost everywhere outside their rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/mixed_parity_loosens_criterion.py`

## Pruned models (out of the set; narrowest margin first)

### random_program_inference_ruleless — pruned (experiment2 end of experiment)

**Margin:** 9.9 nats behind mutual_similarity_contrast_standard (2.1× dse)

**Hypothesis:** A refinement of `rule_gated_decade_familiarity` (final-digit similarity with an "ends in that digit" rule pre-empting parity credit, a common-factor boost, and decade-band familiarity that detected rules pre-empt, all kept). The one change: when two or more examples share no explicit rule (mixed final digits and no common factor of 3 or more), people sometimes conclude that the program just produces arbitrary numbers, and on those occasions they stop comparing the target with the examples and say "yes" at a flat rate that does not depend on the target. So rule-less sets flatten toward uniform acceptance: far, unmatched targets get accepted much more than similarity predicts, and sets differ more in how broadly people generalize.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/random_program_inference_ruleless.py`

### prototype_abstraction_similarity — pruned (experiment1 end of experiment)

**Margin:** 11.9 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 13

**Hypothesis:** People do not store the individual examples or infer a rule; they abstract a single prototype of the set — its average magnitude and the proportion of examples having each arithmetic property (even, multiple of 3/4/5/10, same final digit, prime, square, power of 2) — and say "yes" to the degree that the target is similar to that prototype, with similarity decaying exponentially with the attention-weighted distance between the target's own properties and the prototype's averaged ones. So a property shared by every example acts as a strong expectation, a property split among the examples barely matters, and targets near the set's centre of magnitude (rather than near any one example) are favoured.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/prototype_abstraction_similarity.py`

### shared_property_rule_matching — pruned (experiment1 end of experiment)

**Margin:** 14.3 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 13

**Hypothesis:** People ignore magnitude and similarity entirely and judge by rule matching: they note which arithmetic properties all the examples have in common (even, odd, multiple of 3/4/5/10, a shared final digit, prime, square, power of 2) and say "yes" to the degree that the target has those same properties, regardless of how far it lies from the examples. So rule-consistent targets are accepted even when numerically distant, rule violators are rejected even when close, and a single example (which carries all its own properties) generalizes as readily as several.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_rule_matching.py`

### final_digit_rule_suppresses_parity — pruned (experiment2 end of experiment)

**Margin:** 16.2 nats behind mutual_similarity_contrast_standard (2.1× dse)

**Hypothesis:** A refinement of `decade_band_digit_common_factor` (final-digit similarity, common-factor boost, decade-band familiarity all kept). The one change: the partial credit people give a target for merely sharing the examples' parity is not fixed but shrinks with how consistently the examples share one final digit — when the examples all end in the same digit, people infer an "ends in that digit" rule and give little or no credit to same-parity targets ending in a different digit, while for examples with varied final digits parity still earns its usual partial credit. This addresses the critique that, in sets whose examples share a final digit, people accept parity-only matches far less than the model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/final_digit_rule_suppresses_parity.py`

### salience_weighted_rule_matching — pruned (experiment1 end of experiment)

**Margin:** 18.1 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still note which arithmetic properties all the examples share (even, odd, multiple of 3/4/5/10, prime, square, power of 2, a common final digit) and say "yes" to the degree that the target has those same properties, regardless of its magnitude, but the one change is that these properties are not equally salient — each kind of property carries its own inferred weight, so whether the target matches a highly salient shared property (e.g. parity or being a power of 2) matters far more than whether it matches a weak one. This makes judgments sharper across items than equal-weight rule matching (the critique's excess item and set variance) and lets a rule such as "multiples of 10" or "powers of 2" decide the answer even when the target shares no final digit with the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/salience_weighted_rule_matching.py`

### rule_matching_with_range_extension — pruned (experiment1 end of experiment)

**Margin:** 18.4 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, but they also treat the span of the examples (from smallest to largest) as part of what the program produces, so a target inside that span gets an extra, inferred boost to "yes" that fades smoothly with how far the target lies beyond the span's edges (for a single example, with its distance from that example). The one change is this range-membership component grafted from `range_extension_by_spacing`, which addresses the critique that the rule model misses interval-based generalization: in-range rule violators and in-range targets of rule-less sets are accepted more than distant ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_matching_with_range_extension.py`

### shared_digit_rule_preempts_parity — pruned (experiment2 end of experiment)

**Margin:** 18.4 nats behind mutual_similarity_contrast_standard (2.3× dse)

**Hypothesis:** A refinement of `decade_band_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples, with a boost for multiples of the examples' common factor and full decade-band familiarity fading outside the band. The one change is rule abstraction on the final digit: when two or more examples all end in the same digit, people abstract the explicit rule "ends in that digit", and this rule pre-empts the graded parity similarity, so a target of the same parity but a different final digit loses (by an inferred share) the partial credit it would otherwise get. The sharpest disagreement with the incumbent lives in uniform-final-digit sets (e.g. 13, 43, 73): the incumbent accepts same-parity, different-digit targets (e.g. 15) far more than opposite-parity ones, whereas this model predicts they are rejected about as often.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/shared_digit_rule_preempts_parity.py`

### example_count_diluted_decade_familiarity — pruned (experiment2 end of experiment)

**Margin:** 18.7 nats behind mutual_similarity_contrast_standard (2.4× dse)

**Hypothesis:** People judge the target as the current best model says (final-digit similarity to the examples with parity credit suppressed when the examples share one final digit, a boost for multiples of the examples' common factor, and familiarity with the band of decades the examples span), but how much weight they put on that magnitude (decade) familiarity depends on how many examples they saw: with a lone example there is no relational structure to extract, so people lean heavily on "numbers near that one" and readily accept other numbers in its decade, while each additional example supplies digit and arithmetic structure that draws attention away from magnitude, so the decade-familiarity boost is diluted as the number of examples grows. This addresses the critique that, after a single example, people accept same-decade targets with a different final digit far more than the model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/example_count_diluted_decade_familiarity.py`

### rule_matching_with_range_fill — pruned (experiment1 end of experiment)

**Margin:** 19.2 nats behind digit_similarity_with_common_factor; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 13

**Hypothesis:** A refinement of `shared_property_rule_matching`: people still say "yes" to the degree that the target has the arithmetic properties all the examples share, regardless of its distance from them, but they also treat the span between the smallest and largest example as territory the program covers, so a target lying inside that range gets an extra, inferred boost toward "yes" (the one change, a range-membership component taken from `range_extension_by_spacing`). So, as the critique found, in-range targets are accepted more often than rule matching alone predicts, while rule-consistent targets far outside the range are still accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_matching_with_range_fill.py`

### size_principle_decade_band — pruned (experiment2 end of experiment)

**Margin:** 19.3 nats behind mutual_similarity_contrast_standard (2.4× dse)

**Hypothesis:** People generalize as a Bayesian learner would from the band of decades the examples span (with final-digit similarity, parity credit suppressed by a shared final digit, and a common-factor boost as in the current best model), applying the size principle to that magnitude concept: the narrower the band of decades the examples occupy, the more strongly a target inside it is judged likely, so a single example or a set confined to one decade produces strong within-decade generalization while a band covering many decades confers little. The one distortion is that the size principle is applied with an inferred, partial strength rather than in full, so familiarity falls with the band's width by a weaker-than-normative power.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/size_principle_decade_band.py`

### shared_digit_sharpened_decade_band — pruned (experiment2 end of experiment)

**Margin:** 19.5 nats behind mutual_similarity_contrast_standard (2.3× dse)

**Hypothesis:** A refinement of `diminishing_similarity_decade_band`: people still judge the target by its final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with diminishing returns from each further resembling example, a boost for multiples of the examples' common factor, and full leading-digit familiarity inside the band of decades the examples span, fading outside it. The one change is that when every example ends in the same final digit, people take that shared digit as the program's rule and, by a size-principle-like sharpening, give a different digit of the same parity less and less partial credit with each additional example that confirms the shared digit. This addresses the critique that, in sets whose examples all share one final digit, people accept parity-only matches far less than the model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/shared_digit_sharpened_decade_band.py`

### variability_tuned_exemplar_gradient — pruned (experiment2 end of experiment)

**Margin:** 21.0 nats behind mutual_similarity_contrast_standard (2.1× dse)

**Hypothesis:** People judge the target by its similarity to the stored examples, and the breadth of their generalization around those exemplars is set by how variable the exemplars are (the category-variability effect): when the examples all end in the same digit, the similarity gradient on final digit is tight, so a target that merely shares their parity gets almost no credit, whereas when the examples' final digits vary, the gradient is broad and same-parity targets are accepted almost as readily as digit matches. Apart from this variability-tuned digit gradient, the comparison is the current best model's (final-digit similarity averaged over examples, full leading-digit familiarity within the examples' band of decades fading outside it, and a boost for multiples of their common factor).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/variability_tuned_exemplar_gradient.py`

### attention_shift_magnitude_to_digit — pruned (experiment2 end of experiment)

**Margin:** 21.8 nats behind mutual_similarity_contrast_standard (2.7× dse)

**Hypothesis:** People divide attention between a number's magnitude (its decade) and its final digit, and that split is set by how many examples they have read: a single example offers no regularity among digits to notice, so people judge mostly by magnitude and readily accept other numbers in the example's decade, but each further example lets digit regularities be detected and draws attention away from magnitude toward the final digit. Apart from this set-size-driven attention shift, judgments follow the current best model (final-digit similarity with parity credit suppressed by a shared final digit, decade-band familiarity, and a common-factor boost).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/attention_shift_magnitude_to_digit.py`

### iter3_candidate4 — pruned (experiment2 end of experiment)

**Margin:** 22.9 nats behind mutual_similarity_contrast_standard (2.5× dse)

**Hypothesis:** A refinement of the incumbent `final_digit_rule_suppresses_parity` (final-digit similarity with rule-suppressed parity credit, common-factor boost and decade-band familiarity all kept). The one change: how much people lean on decade-band (magnitude-neighbourhood) familiarity depends on how many examples they have seen — with a single example there is no shared pattern to abstract, so people generalize mostly to its numerical neighbourhood (its decade), and each additional example shifts reliance away from the neighbourhood toward the digit and factor patterns the examples share, with an inferred rate of decline. This addresses the critique that after a single example people accept same-decade targets with a different final digit far more than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/iter3_candidate4.py`

### sequential_delta_rule_property_learning — pruned (experiment1 end of experiment)

**Margin:** 24.0 nats behind digit_similarity_with_common_factor (2.5× dse)

**Hypothesis:** People build their guess about the program's rule incrementally, in the order the examples are shown: the first example suggests a set of arithmetic properties (parity, divisibility by 3/4/5/10, final digit, prime, square, power of 2), and each later example revises the strength of belief in each property by an error-correcting (delta-rule) update at an inferred learning rate, so that a slow learner stays anchored on the first example (primacy) and a fast learner is dominated by the last one (recency). They then say "yes" to the degree that the target has the properties their final, order-dependent guess favours, regardless of its magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/sequential_delta_rule_property_learning.py`

### decade_band_digit_common_factor — pruned (experiment2 end of experiment)

**Margin:** 24.6 nats behind mutual_similarity_contrast_standard (2.6× dse)

**Hypothesis:** A refinement of `nearest_decade_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with an extra boost for multiples of the examples' common factor. The one change is the metric for magnitude similarity: people treat the whole band of decades running from the lowest example's decade to the highest example's decade as familiar territory. So a target anywhere in that band, including decades between the examples that no example occupies, gets the full leading-digit boost, and the boost decays with each decade the target lies outside the band. This addresses the critique that people accept targets inside the examples' range more than targets outside it (range interpolation), while accepting adjacent-decade near-misses outside the band less than the nearest-decade model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/decade_band_digit_common_factor.py`

### accumulated_digit_mismatch_evidence — pruned (experiment2 end of experiment)

**Margin:** 25.4 nats behind mutual_similarity_contrast_standard (2.9× dse)

**Hypothesis:** People treat each example as a separate piece of evidence about the program and accumulate that evidence across examples, rather than merely averaging their resemblance to them: every example whose final digit the target fails to share costs some credibility (a little for a same-parity mismatch, more for an opposite-parity one), and these costs pile up with the number of examples shown (with an inferred degree of accumulation between pure averaging and full summing), so judgments sharpen as more examples agree. Thus a target that breaks a final digit shared by three examples (e.g. 69 after 59, 89, 79) is rejected more firmly than one breaking a single example's digit, while decade-band familiarity and membership in the examples' common-factor multiples still shift the answer as in the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/accumulated_digit_mismatch_evidence.py`

### diminishing_similarity_decade_band — pruned (experiment2 end of experiment)

**Margin:** 27.1 nats behind mutual_similarity_contrast_standard (2.8× dse)

**Hypothesis:** A refinement of `decade_band_digit_common_factor`: people still judge the target by its final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with a boost for multiples of the examples' common factor and full leading-digit familiarity anywhere in the band of decades the examples span, fading with each decade outside it. The one change is that support from similar examples has diminishing returns: the first resembling example raises the willingness to say "yes" most, and each further one adds less (a concave, power-law transform of the average similarity with an inferred exponent), rather than every resembling example adding an equal share. So a target resembling just one of several examples, such as a single final-digit match, is accepted more, relative to a target resembling all of them, than the linear-share model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/diminishing_similarity_decade_band.py`

### log_compressed_decade_band — pruned (experiment2 end of experiment)

**Margin:** 31.5 nats behind mutual_similarity_contrast_standard (3.0× dse)

**Hypothesis:** A refinement of `decade_band_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with a boost for multiples of the examples' common factor and full familiarity anywhere in the band of decades the examples span. The one change is that familiarity fades outside that band with distance on a compressed, logarithmic mental number line rather than by counting decades. So a target the same number of decades above the band (e.g. 90 after 60, 70) stays more familiar than one the same number of decades below it (e.g. 30 after 60, 70), and leaving a band of small numbers costs far more than leaving a band of large ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/log_compressed_decade_band.py`

### nearest_decade_digit_factor_span_fill — pruned (experiment2 end of experiment)

**Margin:** 33.5 nats behind mutual_similarity_contrast_standard (3.3× dse)

**Hypothesis:** A refinement of `nearest_decade_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with an extra boost for multiples of the examples' common factor and a familiarity boost from the example whose decade is closest to the target's. The one change, a range-interpolation component taken from `common_factor_digit_range_similarity`, is that people also treat the numerical span from the smallest to the largest example as territory the program covers, so a target lying inside that span gets an extra push toward "yes" that fades smoothly with how far (in raw numerical distance, regardless of decade) the target lies beyond the span's edges. This addresses the critique that the incumbent under-produces interval generalization: targets inside the examples' range are accepted more than targets outside it even when they share no final digit with any example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/nearest_decade_digit_factor_span_fill.py`

### divisor_lattice_size_principle — pruned (experiment2 end of experiment)

**Margin:** 37.2 nats behind mutual_similarity_contrast_standard (3.1× dse)

**Hypothesis:** A refinement of `nearest_decade_digit_common_factor` that changes its space of arithmetic concepts: instead of entertaining only one rule, "multiples of the examples' greatest common factor (3 or more)", people entertain every "multiples of d" concept that all the examples obey (d = 2 and up, every common divisor, e.g. multiples of 2, 4 and 8 after 16, 8, 64), and weigh them by the size principle, so smaller, more specific concepts are favoured more strongly the more examples there are, with an inferred strength. A target's arithmetic boost toward "yes" is the resulting graded belief that it lies in the inferred multiples concept, alongside the unchanged final-digit similarity and nearest-decade familiarity.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/divisor_lattice_size_principle.py`

### in_range_nearest_decade_digit_factor — pruned (experiment2 end of experiment)

**Margin:** 38.4 nats behind mutual_similarity_contrast_standard (3.6× dse)

**Hypothesis:** A refinement of `nearest_decade_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with a boost for multiples of the examples' common factor and a familiarity boost from the example in the nearest decade. The one change is that, when several examples are shown, people also take the stretch of the number line between the smallest and largest example as ground the program covers, so a target lying strictly inside the examples' range gets an extra, inferred boost toward "yes", whatever its digits or decade. This addresses the critique that the incumbent under-predicts "yes" for targets inside the examples' range compared with targets outside it (range interpolation).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/in_range_nearest_decade_digit_factor.py`

### nearest_exemplar_luce_similarity — pruned (experiment1 end of experiment)

**Margin:** 38.8 nats behind digit_similarity_with_common_factor (2.5× dse)

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`. It keeps the same exemplar representation: magnitude plus arithmetic features, with similarity falling off exponentially with distance weighted separately on each. The one change is that people judge the target by its similarity to the single most similar example shown (nearest-neighbour retrieval), compared against a fixed criterion (Luce's ratio rule), rather than by similarity summed over all examples. So adding examples does not inflate "yes" for every target. A lone example generalizes as broadly as a rich set, and a target that closely matches any one example on its arithmetic features is accepted however far it lies from the others, which addresses the critique's finding that the summed model under-generalizes from single examples and to numerically distant targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/nearest_exemplar_luce_similarity.py`

### span_filled_nearest_exemplar — pruned (experiment1 end of experiment)

**Margin:** 40.2 nats behind digit_similarity_with_common_factor (2.6× dse)

**Hypothesis:** A refinement of `nearest_exemplar_luce_similarity`: people still judge the target by its similarity to the most similar example on arithmetic features, combined with magnitude similarity and compared against a Luce criterion, but the one change is that on the magnitude dimension they fill in the gaps between the examples, treating the whole span from the smallest to the largest example as already "seen", so magnitude dissimilarity is the target's distance to that span (zero anywhere inside it) rather than to the nearest single example. Targets inside the examples' range are therefore accepted more readily than equally feature-similar targets outside it, addressing the critique that the incumbent misses range-based generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/span_filled_nearest_exemplar.py`

### nearest_decade_digit_common_factor — pruned (experiment2 end of experiment)

**Margin:** 40.5 nats behind mutual_similarity_contrast_standard (3.7× dse)

**Hypothesis:** A refinement of `digit_similarity_with_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with an extra boost for multiples of the examples' common factor, but the one change is that they also compare the target's leading (tens) digit with the examples' and get a familiarity boost from the example whose decade is closest to the target's, a boost that is largest when the target shares that example's decade and decays with each decade further away. This addresses the critique that the model under-predicts "yes" for targets sharing a decade with an example (and, through that, for targets lying among the examples).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/nearest_decade_digit_common_factor.py`

### decade_digit_similarity_common_factor — pruned (experiment2 end of experiment)

**Margin:** 53.0 nats behind mutual_similarity_contrast_standard (4.7× dse)

**Hypothesis:** A refinement of `digit_similarity_with_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least) and still boost targets that are multiples of the examples' common factor (3 or more), but the one change is that they also compare the target's leading (tens) digit with each example's, so a target in the same decade as an example is judged similar to it, one in an adjacent decade partly similar, and "yes" also grows with the share of examples it resembles in decade. This addresses the critique that people accept targets sharing a decade with an example (even without a final-digit match) more than the magnitude-blind incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/decade_digit_similarity_common_factor.py`

### digit_similarity_with_common_factor — pruned (experiment2 end of experiment)

**Margin:** 54.7 nats behind mutual_similarity_contrast_standard (4.2× dse)

**Hypothesis:** A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), but the one change is that they also notice the largest whole number (3 or more) that divides every example and treat "multiples of that number" as a candidate rule, so a target divisible by the examples' common factor gets an extra, inferred boost toward "yes" whatever its final digit. This addresses the critique that, among parity-matched targets, people accept those divisible by the examples' common divisor (e.g. multiples of 8 after 16, 8, 64) more than the final-digit model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_similarity_with_common_factor.py`

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

### decade_digit_range_factor_similarity — pruned (experiment2 end of experiment)

**Margin:** 62.3 nats behind mutual_similarity_contrast_standard (4.5× dse)

**Hypothesis:** A refinement of `common_factor_digit_range_similarity`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), with a boost for targets inside the examples' span that fades beyond its edges and an extra push for multiples of the examples' common factor (3 or more). The one change is that people also compare leading digits: the share of examples that lie in the same decade as the target (same tens digit) adds an inferred boost toward "yes", so a target like 34 after 31 is accepted more than its final digit alone predicts, even when it ends in a different digit. This addresses the critique that, among targets sharing no final digit with any example, those in an example's decade are accepted more than the best model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/decade_digit_range_factor_similarity.py`

### shared_feature_attention_gcm — pruned (experiment1 end of experiment)

**Margin:** 63.4 nats behind digit_similarity_with_common_factor (4.0× dse)

**Hypothesis:** A refinement of `exemplar_feature_similarity_gcm`: people still judge the target by its summed exemplar similarity to the examples on magnitude and arithmetic features, but their attention to each arithmetic feature is drawn by the examples themselves — a feature on which all the examples agree (e.g. all even, all powers of 2, all ending in 5) receives extra attention, by an inferred amount, relative to features on which the examples disagree. The one change is this example-driven attention shift, which makes a target that breaks a property shared by every example much less similar to them, so rule-consistent targets are accepted (even far away) and rule-violating ones rejected more sharply than with fixed attention.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_feature_attention_gcm.py`

### common_factor_digit_range_similarity — pruned (experiment2 end of experiment)

**Margin:** 69.0 nats behind mutual_similarity_contrast_standard (4.8× dse)

**Hypothesis:** A refinement of `digit_similarity_with_range_extension`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), with an extra boost for targets inside the examples' span that fades beyond its edges, but the one change is that they also notice when several examples are all multiples of a common factor of 3 or more (their greatest common divisor, e.g. 8 for 16, 8, 64) and infer a "multiples of that factor" rule, giving an inferred extra push toward "yes" to targets that are multiples of it. This addresses the critique that, among parity-matched targets of sets sharing a common divisor, people accept divisible targets more than the digit model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/common_factor_digit_range_similarity.py`

### relational_step_extrapolation — pruned (experiment1 end of experiment)

**Margin:** 71.4 nats behind digit_similarity_with_common_factor (4.0× dse)

**Hypothesis:** People do not look for shared properties or similarity; they infer the generating operation from the relations among the examples themselves (relational, analogical extrapolation). They notice the differences and whole-number ratios linking pairs of examples (e.g. 16 is 4 times 4, 83 is 15 below 98) and say "yes" to a target that the same step produces from one of the examples (64 after 4,16,12; 68 after 98,83), and "no" otherwise, so a lone example, which carries no relation, gives no basis for extrapolation.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/relational_step_extrapolation.py`

### iter0_candidate0 — pruned (experiment2 end of experiment)

**Margin:** 86.9 nats behind mutual_similarity_contrast_standard (5.7× dse)

**Hypothesis:** People generalize like an ideal Bayesian concept learner — weighing candidate concepts (numbers with the same final digit, the same parity, the same decade, multiples of a number, or a numerical interval) by their prior plausibility and by the size principle — but with one resource limit: they cannot integrate the examples jointly, so they generalize from each example on its own and answer "yes" according to the average of these single-example generalizations. Because evidence never accumulates across examples, judgments do not sharpen as examples are added, a target is accepted in proportion to how many examples individually support it, and a shown example is accepted almost always.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/iter0_candidate0.py`

### range_extension_by_spacing — pruned (experiment1 end of experiment)

**Margin:** 88.5 nats behind digit_similarity_with_common_factor (5.0× dse)

**Hypothesis:** People ignore arithmetic rules and generalize by range extension: they treat every number between the smallest and largest example as equally likely to be produced, and their willingness to say yes falls off smoothly with how far a target lies beyond that range, measured in units of the typical spacing between the examples (so tightly packed, many-example sets generalize only a little past their edges, while a lone example generalizes over a learned width). This disagrees most sharply with the rule-based Bayesian model on rule-consistent targets far outside the example range (e.g. 64 after 4,16,12), which it predicts are rejected, and on in-range rule violators, which it predicts are accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/range_extension_by_spacing.py`

### digit_similarity_with_range_extension — pruned (experiment2 end of experiment)

**Margin:** 90.8 nats behind mutual_similarity_contrast_standard (5.6× dse)

**Hypothesis:** A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but they also treat the span from the smallest to the largest example as territory the program covers, so a target inside that span gets an extra, inferred boost toward "yes" that fades smoothly with how far it lies beyond the span's edges (for a single example, with its distance from that example). The one change is this range-membership component grafted from `range_extension_by_spacing`, addressing the critique that the magnitude-blind digit model under-predicts yes for in-range versus out-of-range targets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_similarity_with_range_extension.py`

### place_value_grid_similarity — pruned (experiment2 end of experiment)

**Margin:** 91.4 nats behind mutual_similarity_contrast_standard (5.6× dse)

**Hypothesis:** People represent each two-digit number by its place values, as a position on a tens-digit by units-digit grid, and judge the target by its similarity to each example on that grid: similarity is highest for the same final digit, partial for a different final digit of the same parity, and lowest for opposite parity, and it is multiplied down by how many decades apart the target and that example lie. They say "yes" in proportion to the target's average place-value similarity to the examples, so a final-digit match earns credit mainly when it falls in the same or a nearby decade as the matching example. This disagrees most sharply with the magnitude-blind final-digit model on digit-matched targets several decades away from every matching example (e.g. 93 after 13), which it predicts are rejected far more often than nearby digit-matched targets (e.g. 23).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/place_value_grid_similarity.py`

### proximity_cued_digit_retrieval — pruned (experiment2 end of experiment)

**Margin:** 93.1 nats behind mutual_similarity_contrast_standard (5.8× dse)

**Hypothesis:** People compare the target's final digit with the final digits of the examples (same digit most similar, same parity partly, opposite parity least), but the target acts as a retrieval cue: examples close to it in magnitude come to mind readily, while numerically distant examples are only sometimes recalled for the comparison. So "yes" grows with the final-digit similarity to the examples that are actually retrieved, which makes a digit match with a nearby example count for more than the same match with a far-away one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/proximity_cued_digit_retrieval.py`

### bigelow_piantadosi_inferred_noise — pruned (experiment1 end of experiment)

**Margin:** 99.6 nats behind digit_similarity_with_common_factor (4.6× dse)

**Hypothesis:** A refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 42 rules (with a prior weight inferred for each) and every interval of 1-100, with the size principle, but people believe a much larger and unknown share of the examples are random draws from 1-100 rather than fixed at 10%; that noise share is inferred. Treating examples as more often noise lets one stray example be explained away, so the learner commits to tighter concepts around the bulk of rich example sets (less generalization outside their range) while staying graded rather than near-certain in its judgments.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_inferred_noise.py`

### digit_similarity_with_span_falloff — pruned (experiment2 end of experiment)

**Margin:** 100.0 nats behind mutual_similarity_contrast_standard (5.8× dse)

**Hypothesis:** A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but the one change is that they also treat the span from the smallest to the largest example as covered by the program, so their willingness to say "yes" falls off steadily with how far the target lies beyond that span (with no penalty anywhere inside it). This addresses the critique that people accept targets inside the examples' range more than the magnitude-blind digit model predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_similarity_with_span_falloff.py`

### size_weighted_interval_learning — pruned (experiment1 end of experiment)

**Margin:** 104.1 nats behind digit_similarity_with_common_factor (4.2× dse)

**Hypothesis:** Refining `bigelow_piantadosi_interval` (Bayesian concept learning over 42 mathematical rules and every interval of 1-100, with per-rule prior weights and a Beta prior over interval midpoints), with one change: the strength of the size principle is inferred rather than fixed, so a learner may treat each example as stronger (or weaker) evidence than one random draw from the concept. If people apply a stronger size principle than the original model assumes, rich example sets should pull their belief onto the tightest concepts that cover the examples, and they should say "yes" less often to numbers outside the examples' range, which is the gap the critique found for sets of 3-4 examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/size_weighted_interval_learning.py`

### digitwise_common_feature_matching — pruned (experiment2 end of experiment)

**Margin:** 107.4 nats behind mutual_similarity_contrast_standard (6.3× dse)

**Hypothesis:** People read each number as a string of two digit features, a tens digit (its decade) and a units digit together with that digit's parity, and judge the target by Tversky-style common-feature matching: every feature the target shares with an example (same decade, same final digit, same parity) adds its own independent increment of support for "yes", averaged over the examples. Because the features add rather than multiply, a target in the same decade as an example gains credit even when its final digit differs (e.g. 17 after 12), and a final-digit match earns credit however many decades away it lies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digitwise_common_feature_matching.py`

### inferred_noise_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 108.9 nats behind digit_similarity_with_common_factor (3.9× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 with the size principle, but the share of examples the learner believes are random noise (rather than drawn from the concept) is inferred from the data instead of fixed at 10%. If people treat examples as rarely noisy, they refuse to discard an awkward example in favour of a broad rule like "even", so rich, rule-less example sets are explained by the interval spanning them and numbers outside their range are rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_noise_concept_learning.py`

### final_digit_similarity_gradient — pruned (experiment2 end of experiment)

**Margin:** 116.2 nats behind mutual_similarity_contrast_standard (6.6× dse)

**Hypothesis:** People ignore magnitude and judge the target by how much its final digit resembles the final digits of the examples: a target ending in the same digit as an example is maximally similar to it, one ending in a different digit of the same parity (odd/even) is partly similar, and one of the opposite parity is least similar. They say "yes" in proportion to the target's average final-digit similarity to all the examples shown, so the answer is graded by the share of examples it resembles rather than by any rule that every example must obey, and numerically close targets get no credit for closeness.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/final_digit_similarity_gradient.py`

### serial_position_digit_similarity — pruned (experiment2 end of experiment)

**Margin:** 117.7 nats behind mutual_similarity_contrast_standard (6.7× dse)

**Hypothesis:** People compare the target's final digit with the final digits of the examples (same digit most similar, same parity partly, opposite parity least), but they do not hold every example equally strongly in memory: how much each example counts in the comparison depends on its serial position in the order shown, changing steadily from first to last (recency if later examples are held more strongly, primacy if earlier ones are). So "yes" grows with a position-weighted average of final-digit similarity, and the same numbers shown in a different order change which digit matches drive acceptance.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/serial_position_digit_similarity.py`

### recency_faded_magnitude_anchor — pruned (experiment2 end of experiment)

**Margin:** 124.8 nats behind mutual_similarity_contrast_standard (7.1× dse)

**Hypothesis:** People keep the final digit of every example as durable gist, and say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least); but their memory of each example's exact magnitude fades with every later example shown, so the most recent example acts as a magnitude anchor. Targets numerically close to the remembered magnitudes, above all to the last example shown, get an extra push toward "yes", so the same numbers in a different order shift which neighbourhood of the number line is accepted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/pruned/recency_faded_magnitude_anchor.py`

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
