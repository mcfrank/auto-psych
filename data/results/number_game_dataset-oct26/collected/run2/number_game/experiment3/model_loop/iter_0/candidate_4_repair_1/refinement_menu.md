# Refinement menu

The models you may refine, other than the incumbent `zigzag_breadth_sequence_exemplar`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### spread_calibrated_size_gradient — rank 1, 14.3 ± 6.5 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4735.9

**Hypothesis:** People judge how close a target is in size to a set of examples on a yardstick calibrated by the examples themselves: the distance by which a target falls outside the stretch the examples span is felt relative to how widely the examples are spread, so after tightly clustered examples (e.g. 41, 43, 44) a number ten steps away feels far, whereas after widely scattered examples (e.g. 12, 47, 83) the same ten steps feel negligible. The size-similarity gradient around a set therefore flattens as the examples' spread grows, making scattered sets generalized far more broadly beyond their span than tight ones (the too-compressed between-set yes-rate spread in the critique); everything else — recency-weighted similarity on final digit, parity and shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, the drifting criterion and the order-directed momentum of the incumbent — is unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/spread_calibrated_size_gradient.py`

### sequence_momentum_span_exemplar — rank 2, 15.1 ± 6.4 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4736.8

**Hypothesis:** People read the examples as a sequence the program is producing in the order shown and expect it to keep going in the direction it has been moving: beyond the span the examples fill in, a number lying ahead of the last-shown example in the direction of travel from the first example to the last feels much closer in size than an equally distant number lying back behind where the sequence started. So after 20, 40 people generalize to 50-60 more than to 0-10, but after 40, 20 they generalize to 10 more than to 50 — a momentum in the order of presentation, not a fixed upward preference, which produces the critique's recency (closer-to-last) advantage.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/sequence_momentum_span_exemplar.py`

### span_filling_size_exemplar — rank 3, 21.4 ± 7.9 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4743.1

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples on final digit, parity and graded shared factor (with coherence-stretched attention), still consult the "multiples of g" and "ends in d" rules as separate evidence, and still relax their criterion over the session. The single change is how closeness in size is judged: people treat the examples as marking out a filled-in stretch of the number line, so a target's size mismatch is how far it falls outside the span from the smallest to the largest example (zero anywhere inside it), not its distance from each example. The disagreement with the best model lives inside widely spaced sets — after 10, 90 the target 50 is as close in size as 10 itself and is endorsed far more often, so widely spread sets are generalized much more broadly than tight ones (the too-compressed between-set yes-rate spread in the critique).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/span_filling_size_exemplar.py`

### spread_scaled_recency_span_exemplar — rank 4, 24.5 ± 8.5 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4746.2

**Hypothesis:** Refinement of recency_anchored_span_exemplar_2 (recency-weighted exemplar similarity on size, final digit, parity and graded shared factor with coherence-stretched attention; each earlier example joined to the last-shown one by a filled-in path; the "multiples of g" and "ends in d" rules as separate evidence; a criterion that relaxes over the session). The single change is what sets how broadly people generalize in size: instead of the number of examples, it is how widely the examples are spread — people take the spread of the examples as evidence of how far the program ranges, so the size gradient around a tightly clustered set (e.g. 41, 43, 44) is steep and around a widely spread set (e.g. 12, 57, 90) is shallow, even beyond the span the examples fill. Tight sets are thus generalized narrowly and spread sets broadly, widening the between-set differences in yes-rate that the critique finds too compressed.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/spread_scaled_recency_span_exemplar.py`

### recency_anchored_span_exemplar_2 — rank 5, 29.7 ± 10.5 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4751.4

**Hypothesis:** Refinement of set_size_scaled_magnitude_gradient (recency-weighted exponential similarity to the examples on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules as separate evidence, a criterion that relaxes over the session, and a size gradient that broadens with the number of examples). The single change, adapting the incumbent's span-filling idea, is that people treat each earlier example as connected to the most recently shown one by a filled-in path the program travelled, so a target's size mismatch with an earlier example is how far it lies outside the stretch between that example and the last one (zero inside it), while its mismatch with the last example is its plain distance from it. Generalization thus fills the gaps between the examples yet stays centred on the latest example, so among targets inside or near a set, those nearer the last-shown example are endorsed more than those nearer the first (the recency gap the critique reports, which an order-blind span cannot produce).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/recency_anchored_span_exemplar_2.py`

### recollection_shortcut_span_exemplar — no comparison row

**Hypothesis:** People answer by two routes at the decision: if they explicitly recollect that the target is one of the numbers the program actually showed, they say "yes" outright on that recollection alone (an all-or-none, high-threshold recognition that succeeds on some fraction of such trials); only when no such recollection occurs do they fall back on weighing graded familiarity (the simplest current span-filling dual-route exemplar evidence) against their criterion. Repeated examples are therefore endorsed far more than numbers just 1-3 away, by a boost that familiarity alone cannot produce (the critique's exact-example gap).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/recollection_shortcut_span_exemplar.py`

### enumerated_list_concept_exemplar — no comparison row

**Hypothesis:** Refinement of zigzag_breadth_sequence_exemplar (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, the relaxing criterion, the filled-in span, the momentum of the sequence's direction of travel and the zigzag-set breadth). The single change adds one concept to the space people consider: that the program simply produces a fixed list of particular numbers, of which the shown examples are members. This concept is consulted as separate evidence at the decision, and it applies only to targets that are themselves among the shown examples. People therefore say "yes" to a repeated example far more often than to a number only 1-3 away, even though the two are almost equally similar in size (the critique's exact-example boost).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/enumerated_list_concept_exemplar.py`

### scalar_variability_exemplar_traces — no comparison row

**Hypothesis:** People judge a target by its recency-weighted similarity to the stored examples (closeness in size, shared final digit, shared parity and shared factors), but each example is stored as an approximate magnitude whose fuzziness grows in proportion to its size, as in the scalar variability of the approximate number system. So the size-similarity gradient is sharp around a small example and broad around a large one: a target 10 away from the example 8 feels far, while a target 10 away from 80 feels close, and a target identical to a small example stands out sharply from its neighbours. This predicts fewer "yes" answers beyond the examples when the nearest example is small than when it is large, for the same linear gap.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/scalar_variability_exemplar_traces.py`

### recognition_boost_zigzag_exemplar — no comparison row

**Hypothesis:** Refinement of zigzag_breadth_sequence_exemplar (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, the relaxing criterion, the filled-in span, sequence momentum and zigzag breadth). The single change is a recognition route: when the target is one of the examples people recognise it as an item the program has already produced, and this recognition adds evidence for "yes" over and above graded similarity, in proportion to how well that example is still remembered (strongest for the last-shown one). So a repeated example is endorsed far more than numbers only 1-3 away from it, the boost the critique finds the incumbent under-produces.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/recognition_boost_zigzag_exemplar.py`

## Pruned models (out of the set; narrowest margin first)

### set_size_scaled_magnitude_gradient — pruned (experiment2 end of experiment)

**Margin:** 5.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 14

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that how broadly people generalize in size around each example depends on how many examples the program has shown: they take a larger sample as evidence that the program ranges more widely, so the size gradient around each example widens with the number of examples (a single example is generalized narrowly in size, three or four examples broadly), the reverse of a size-principle narrowing. This changes how generously whole sets are generalized relative to one another, addressing the critique's between-set yes-rate spread.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/set_size_scaled_magnitude_gradient.py`

### memory_limited_span_exemplar — pruned (experiment2 end of experiment)

**Margin:** 5.8 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 14

**Hypothesis:** Refinement of span_filling_size_exemplar: people still infer that the program fills in the stretch of the number line the examples mark out (size mismatch = how far the target falls outside that stretch), still judge final digit, parity and shared factor by recency-weighted similarity to the examples, consult the "multiples of g" and "ends in d" rules, and relax their criterion over the session. The single change is a memory limit on that span inference: the stretch is marked out only by the examples people still remember, and the same fading memory that down-weights older examples makes them more likely to be forgotten altogether, so the inferred stretch often shrinks toward the most recently shown example. Targets near the last example are therefore endorsed more than equally placed targets near the first, which the full-span model cannot produce (the critique's last-versus-first recency gap).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/memory_limited_span_exemplar.py`

### partially_filled_span_exemplar — pruned (experiment2 end of experiment)

**Margin:** 6.2 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 14

**Hypothesis:** Refinement of span_filling_size_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (final digit, parity, graded shared factor, coherence-stretched attention), still consult the "multiples of g" and "ends in d" rules as separate evidence, still relax their criterion over the session, and still treat the span between the smallest and largest example as filled in. The single change is that the span is only partly filled in: a target's size mismatch to each remembered example blends how far it falls outside the span with its own distance from that example, so inside a wide span targets near an example still feel closer to it — and because the most recently shown example is remembered best, size generalization favours targets near the last example over targets near the first (the critique's recency_last_vs_first_example gap, which a purely span-based mismatch cannot produce).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/partially_filled_span_exemplar.py`

### partial_span_filling_exemplar — pruned (experiment2 end of experiment)

**Margin:** 6.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 14

**Hypothesis:** Refinement of span_filling_size_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (final digit, parity, graded shared factor, coherence-stretched attention), still consult the "multiples of g" and "ends in d" rules as separate evidence, and still relax their criterion over the session. The single change is that the span between the examples is only partly filled in: a target's size mismatch with each example is a blend of how far it falls outside the span of the examples and its own distance from that example, with the filling-in share inferred. Because the per-example part of size closeness is weighted by memory, targets inside the span that lie nearer the last-shown example are endorsed more than those nearer the first — the recency-in-size-generalization the critique reports, which a pure span mismatch (identical for every example) cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/partial_span_filling_exemplar.py`

### set_factor_tuned_attention_exemplar — pruned (experiment1 end of experiment)

**Margin:** 10.2 nats behind coherence_sharpened_decision_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 9

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the examples, with attention stretched toward the dimensions on which the examples resemble one another, but when all the examples share a common factor of 3 or more they notice that specific factor and the factor dimension of similarity becomes "is the target a multiple of the examples' shared factor", rather than whether the target shares any factor with each example separately. The single change is this set-specific redefinition of factor similarity, which predicts a sharper "multiples of g" generalisation (e.g. after 12, 18, 30 the target 42 is endorsed but 15 and 21 are not), closing the common-factor rule gap in the critique.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/set_factor_tuned_attention_exemplar.py`

### drifting_criterion_dual_route_exemplar — pruned (experiment2 end of experiment)

**Margin:** 11.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 14

**Hypothesis:** Refinement of dual_route_factor_and_digit_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), and they still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence when they decide. The single change is that the criterion for saying "yes" relaxes steadily over the session: as people answer more and more questions without feedback, they need less evidence before they endorse a target. The same evidence therefore earns more "yes" answers late in the session than early on. This closes the late-minus-early yes-rate drift the critique reports, which a stationary model cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/drifting_criterion_dual_route_exemplar.py`

### dual_route_factor_and_digit_rule_exemplar — pruned (experiment2 end of experiment)

**Margin:** 15.6 nats behind zigzag_breadth_sequence_exemplar (2.2× dse)

**Hypothesis:** Refinement of dual_route_shared_factor_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), and still consult an abstracted explicit rule alongside similarity at the decision, but the rule route is not limited to shared factors. The single change is that when all the examples end in the same digit, people also abstract the explicit rule "numbers ending in that digit" and consult it as separate evidence, raising "yes" for targets ending in that digit and lowering it for the rest regardless of size distance, in proportion to how unlikely that shared ending would be by coincidence (more examples), closing the "ends in d" gap the critique reports (people apply the final-digit rule more sharply than final-digit similarity predicts).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/dual_route_factor_and_digit_rule_exemplar.py`

### iter1_candidate4 — pruned (experiment2 end of experiment)

**Margin:** 15.7 nats behind zigzag_breadth_sequence_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 14

**Hypothesis:** Refinement of dual_route_factor_and_digit_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another) and still consult the abstracted "multiples of g" and "ends in d" rules as separate evidence at the decision. The single change is progressive disengagement over the session: on a growing share of trials as the session wears on, people stop consulting the examples at all and simply guess, with their own guessing bias toward "yes", so answers late in the session are drawn toward that guessing rate for every kind of target alike — closing the critique's session-drift gap (yes rates rise from the first to the last third of the session, by similar amounts for near and far targets) and putting a floor and ceiling on how extreme any item's yes rate can be.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/iter1_candidate4.py`

### coherence_decisive_factor_digit_rules — pruned (experiment2 end of experiment)

**Margin:** 17.0 nats behind zigzag_breadth_sequence_exemplar (2.3× dse)

**Hypothesis:** Refinement of coherence_decisive_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), still consult the abstracted "multiples of g" rule as separate evidence at the decision, and still let set coherence set how decisively the combined evidence is turned into a yes or no. The single change, taken from dual_route_factor_and_digit_rule_exemplar, is that when all the examples end in the same digit people also abstract the explicit rule "numbers ending in that digit" and consult it as separate evidence (in proportion to how unlikely the shared ending would be by coincidence) — and because such sets are coherent, the digit rule is applied near-deterministically, predicting a sharper "ends in d" contrast than either parent model alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coherence_decisive_factor_digit_rules.py`

### coherence_decisive_dual_route_exemplar — pruned (experiment2 end of experiment)

**Margin:** 18.9 nats behind zigzag_breadth_sequence_exemplar (2.3× dse)

**Hypothesis:** Refinement of dual_route_shared_factor_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another) and still consult the abstracted "multiples of g" rule as separate evidence at the decision when the examples share a factor of 3 or more. The single change, taken from coherence_sharpened_decision_exemplar, is that how much the examples resemble one another also sets how decisively people answer: after a coherent set the combined evidence (similarity plus rule, relative to their criterion) is turned into a near-deterministic yes or no, whereas after an incoherent set or a single example answers stay noisy — so responses to the same item are more consistent across people than the incumbent predicts (the cell_response_dispersion gap) and the "ends in d" contrast in coherent same-digit sets is sharper.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coherence_decisive_dual_route_exemplar.py`

### dual_route_shared_factor_rule_exemplar — pruned (experiment2 end of experiment)

**Margin:** 19.7 nats behind zigzag_breadth_sequence_exemplar (2.1× dse)

**Hypothesis:** Refinement of dimension_specific_coherence_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with each dimension's attention stretched by its own sensitivity to how much the examples resemble one another, but when the examples all share a common factor of 3 or more they also abstract the explicit rule "multiples of that factor" and consult it alongside similarity when answering. The single change is this rule as a separate source of evidence at the decision, not a dimension of similarity: it raises "yes" for multiples and lowers it for non-multiples regardless of how far they are in size, in proportion to how unlikely the shared factor would be by coincidence (larger factors and more examples), so far-away multiples of g are endorsed and nearby non-multiples rejected, closing the common-factor rule gap that similarity alone leaves.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/dual_route_shared_factor_rule_exemplar.py`

### upward_skewed_exemplar_gradient — pruned (experiment2 end of experiment)

**Margin:** 20.7 nats behind zigzag_breadth_sequence_exemplar (2.7× dse)

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that the size gradient around each stored example is directional: a number larger than a remembered example feels closer to it than a number smaller by the same amount (as if the program is expected to count upward), so people generalize further upward than downward from the examples, closing the upward-vs-downward generalization gap the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/upward_skewed_exemplar_gradient.py`

### log_compressed_size_dual_route_exemplar — pruned (experiment2 end of experiment)

**Margin:** 24.4 nats behind zigzag_breadth_sequence_exemplar (2.3× dse)

**Hypothesis:** Refinement of drifting_criterion_dual_route_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still relax their yes-criterion over the session. The single change is that closeness in size is judged on a compressed, logarithmic mental number line (by ratio, not by difference), both when comparing the target with each example and when judging how alike the examples are in size. So people generalize narrowly around small examples and broadly around large ones, and further above an example than below it, which predicts the upward-vs-downward generalization gap and the larger set-to-set differences in breadth that the critique reports.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/log_compressed_size_dual_route_exemplar.py`

### capacity_limited_attention_exemplar — pruned (experiment1 end of experiment)

**Margin:** 25.9 nats behind coherence_sharpened_decision_exemplar (3.0× dse)

**Hypothesis:** Refinement of coherence_tuned_attention_exemplar: people still judge a target by its recency-weighted similarity to the remembered examples, with attention stretched toward the dimensions (size, final digit, parity, shared factor) on which the examples resemble one another, but attention is a fixed-capacity resource, as in selective-attention exemplar theory. The single change is that coherence reallocates a fixed total amount of attention rather than adding attention: stretching attention toward the property the examples share withdraws it from the other dimensions, chiefly closeness in size. So after a coherent set (e.g. 12, 18, 30) far-away targets that share the property are endorsed more and nearby targets that lack it less, sharpening the "multiples of g" gap and making set-level yes rates more extreme than the incumbent predicts, while a single example leaves attention at its baseline.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/capacity_limited_attention_exemplar.py`

### upward_biased_size_disengaging_exemplar — pruned (experiment2 end of experiment)

**Margin:** 27.1 nats behind zigzag_breadth_sequence_exemplar (2.5× dse)

**Hypothesis:** Refinement of iter1_candidate4 (dual-route exemplar with progressive disengagement): people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another), still consult the abstracted "multiples of g" and "ends in d" rules as separate evidence, and still disengage and guess on a growing share of trials over the session. The single change is that size similarity is directional: people treat a number larger than a remembered example as psychologically closer to it than an equally distant smaller number (as if the program is expected to keep counting upward), so they generalise more to targets above the examples than below them — closing the critique's upward-versus-downward generalisation gap, which symmetric size distance cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/upward_biased_size_disengaging_exemplar.py`

### set_rule_route_sharpened_exemplar — pruned (experiment2 end of experiment)

**Margin:** 27.3 nats behind zigzag_breadth_sequence_exemplar (3.1× dse)

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening how decisively similarity is turned into an answer. The single change is an abstracted explicit-rule route at the decision: when all the examples share a final digit or a common factor of 3 or more, people extract the narrowest such rule ("ends in d" or "multiples of g") and consult it alongside similarity, raising "yes" for targets that obey it and lowering it for those that do not regardless of size, in proportion to how unlikely the examples would be to obey that rule by coincidence (more examples, and a rule that covers fewer of the numbers 1-100). Unlike the incumbent's factor-only rule, this treats a shared final digit as a rule too, closing the "ends in d" gap the critique reports (people apply it more sharply than similarity predicts).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/set_rule_route_sharpened_exemplar.py`

### feature_distribution_prototype — pruned (experiment1 end of experiment)

**Margin:** 34.6 nats behind coherence_sharpened_decision_exemplar (2.9× dse)

**Hypothesis:** People do not store the examples one by one or infer a rule; they summarise the set as a single statistical category prototype — its central size and spread, and how often each property (final digit, parity, divisibility by 3 to 10) occurs among the examples, smoothed toward how common that property is among numbers in general — and say "yes" in proportion to how typical the target is of that prototype relative to an arbitrary number. Because a property shared by every example becomes more typical the more examples share it, targets with the shared final digit or common factor are endorsed sharply and increasingly so with set size, while the size component broadens with the examples' spread.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/feature_distribution_prototype.py`

### luce_ratio_exemplar_decision — pruned (experiment1 end of experiment)

**Margin:** 40.6 nats behind coherence_sharpened_decision_exemplar (3.5× dse)

**Hypothesis:** People judge a target by its recency-weighted similarity to the remembered examples (closeness in size, shared final digit, shared parity, divisibility), exactly as in the simple exemplar model, but they turn that familiarity into a yes/no answer by a Luce ratio decision rule: they compare the target's familiarity against a fixed internal criterion familiarity and say "yes" with probability equal to the target's share of (familiarity + criterion), each raised to a response-determinism exponent. Because the rule operates on familiarity ratios rather than differences, targets with very low familiarity are rejected almost always and sets whose examples resemble the targets weakly become near-all-"no", producing more extreme set-level yes rates than a linear-in-familiarity criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/luce_ratio_exemplar_decision.py`

### graded_shared_factor_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 50.9 nats behind coherence_sharpened_decision_exemplar (4.2× dse)

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples, but the number-theoretic part of that similarity is graded by how large a factor the target and each example share, not by whether one divides the other. The single change replaces the all-or-none divisibility mismatch with a graded shared-factor mismatch (zero when one number divides the other, one when they share no factor, in between in proportion to the log size of their greatest common divisor relative to the smaller number), so targets that share a big factor with the examples (e.g. 42 after 12, 18, 30) are endorsed more than divisibility alone predicts, which is what the common-factor rule gap in the critique shows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/graded_shared_factor_exemplar_similarity.py`

### recency_weighted_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 53.0 nats behind coherence_sharpened_decision_exemplar (4.4× dse)

**Hypothesis:** People do not infer a hidden rule; they judge a target by its similarity to the remembered examples, as in an exemplar model. Similarity to each example decays exponentially with the psychological distance between the two numbers, a weighted sum of mismatches on how far apart they are, whether they share a final digit, whether they share parity, and whether one divides the other. Memory for the examples fades with how long ago each was shown, so the last-shown example counts most, and the probability of "yes" rises with the target's recency-weighted summed similarity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_exemplar_similarity.py`

### common_factor_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 54.8 nats behind coherence_sharpened_decision_exemplar (4.3× dse)

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples (mismatches on magnitude, final digit, parity and pairwise divisibility), but when several examples share a common factor of 3 or more they also encode that shared factor as a property of the whole set, so a target that is not a multiple of it is psychologically farther from every example. The single change is this one set-level "shared common factor" mismatch dimension in the distance, which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise exemplar similarity alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/common_factor_exemplar_similarity.py`

### step_progression_rule_exemplar — pruned (experiment2 end of experiment)

**Margin:** 55.9 nats behind zigzag_breadth_sequence_exemplar (4.4× dse)

**Hypothesis:** Refinement of dual_route_factor_and_digit_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, attention stretched by how much the examples resemble one another) and still consult an abstracted explicit rule alongside similarity at the decision, but the rules they can abstract are not just "multiples of g" and "ends in d". The single change is the rule space: people entertain every counting-by-steps rule the examples fit ("start from an example and step by s", i.e. numbers leaving the same remainder as the examples when divided by s, for any step s that carries every example to every other), each favoured a priori the smaller its step, with steps of 10 extra natural because of base-10 notation, and weigh them by how unlikely it is that the examples would line up on that step by chance. So after 5, 9 people endorse 13 and 17 and after 83, 77 they endorse 71 and 89, regardless of size, which neither the factor nor the final-digit rule predicts, while "ends in d" (step 10) and "multiples of g" (remainder 0) remain special cases.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/step_progression_rule_exemplar.py`

### conjunctive_worst_fit_exemplar — pruned (experiment1 end of experiment)

**Margin:** 57.6 nats behind coherence_sharpened_decision_exemplar (4.5× dse)

**Hypothesis:** People judge a target by how well it fits the examples as a whole, conjunctively: rather than summing its similarity to each remembered example (so that closeness to any one example suffices), they ask whether the target fits every example, so its familiarity is set by its worst-fitting examples (a soft maximum of its psychological distances to the examples on size, final digit, parity and graded shared factor, weighted by recency). This disagrees most sharply with the best (additive, coherence-tuned) exemplar model on multi-example sets for targets that closely resemble one or two examples but not the others — e.g. a near neighbour of one example that lacks the factor or final digit all the examples share — which it predicts are rejected, so endorsements concentrate on targets sharing the set-wide property (the "multiples of g" gap) without any separate attention tuning.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/conjunctive_worst_fit_exemplar.py`

### running_prototype_delta_rule — pruned (experiment1 end of experiment)

**Margin:** 58.7 nats behind coherence_sharpened_decision_exemplar (4.4× dse)

**Hypothesis:** People do not store the examples separately; they keep a single running summary of what the program's numbers are like (their size, parity, final digit and which small factors they have) and revise it as each example arrives, moving the summary only part of the way toward each new example. A target is endorsed in proportion to how well it fits that final summary, so example order matters through one learning rate: a slow learner stays anchored on the first example (primacy), a fast one is dominated by the last (recency), and properties that every example shares survive the updating at full strength while properties of only one example are diluted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/running_prototype_delta_rule.py`

### residue_progression_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 62.2 nats behind coherence_sharpened_decision_exemplar (4.4× dse)

**Hypothesis:** People are Bayesian concept learners (with partially applied size principle and occasional guessing), but the rules they entertain are generated from one simple grammar rather than a hand-picked list: every arithmetic progression "numbers that leave remainder r when divided by g" (which includes even/odd, multiples of any g, and "ends in d" as step 10), alongside powers, squares, cubes, primes, contained digits and numeric intervals. A progression's prior plausibility falls off as a power of its step size, with extra plausibility for pure multiples (remainder 0) and for decimal steps of 10, so sets sharing an unusual common factor (e.g. 14, 28, 42) or a final digit evoke a sharp "multiples of g" or "ends in d" rule that the fixed rule list lacks.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/residue_progression_concept_learning.py`

### coherence_sharpened_decision_exemplar — pruned (experiment2 end of experiment)

**Margin:** 67.3 nats behind zigzag_breadth_sequence_exemplar (4.4× dse)

**Hypothesis:** Refinement of graded_factor_coherence_tuned_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another, but how much the examples resemble one another also sets how decisively people turn that similarity into an answer. The single change is that set coherence sharpens the decision rule: after a coherent set (examples alike in size, digit, parity or factors) people answer near-deterministically from similarity, endorsing similar targets and rejecting dissimilar ones almost always, whereas after an incoherent set or a single example their answers stay noisy, which makes set-level yes rates more extreme than the incumbent predicts (the between-set yes-rate spread in the critique) and sharpens the "ends in d" and "multiples of g" gaps for coherent sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coherence_sharpened_decision_exemplar.py`

### coherence_sharpened_set_factor_exemplar — pruned (experiment2 end of experiment)

**Margin:** 69.9 nats behind zigzag_breadth_sequence_exemplar (4.4× dse)

**Hypothesis:** Refinement of coherence_sharpened_decision_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another and set coherence sharpening how decisively similarity is turned into an answer. The single change (the set-level factor from set_factor_tuned_attention_exemplar) is that when several examples share a common factor of 3 or more, people notice that factor and judge the factor part of similarity by whether the target is a multiple of the examples' shared factor (graded by how much of that factor the target shares), rather than by the factors it shares with each example separately — so after 12, 18, 30 the target 42 is close and 15 or 20 far, sharpening the "multiples of g" gap the critique reports as still too small.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coherence_sharpened_set_factor_exemplar.py`

### independent_retrieval_exemplar_search — pruned (experiment2 end of experiment)

**Margin:** 76.1 nats behind zigzag_breadth_sequence_exemplar (4.3× dse)

**Hypothesis:** Refinement of graded_factor_coherence_tuned_exemplar: people still compare the target with each remembered example by exponential similarity (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another. The single change is the decision process. People do not add up familiarity and threshold it. They search memory example by example: each example is recalled (less often the longer ago it was shown) and independently accepts the target with probability equal to its similarity, and they answer "yes" as soon as any recalled example accepts it (otherwise they guess "yes" at a low base rate). So one highly similar example is enough for "yes", and every additional example is one more chance to accept, so yes rates rise with set size.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/independent_retrieval_exemplar_search.py`

### tempered_concept_learning_with_guessing — pruned (experiment1 end of experiment)

**Margin:** 85.7 nats behind coherence_sharpened_decision_exemplar (4.9× dse)

**Hypothesis:** Refinement of tempered_size_principle_concept_learning: people do the same Bayesian concept learning over 42 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but on some fraction of trials they do not consult the inferred concept at all and simply guess, answering "yes" with their own inferred guessing bias. The single change is this inferred lapse-to-guessing process, which predicts that even targets the concept clearly excludes get a substantial floor of "yes" answers (and clearly included targets a ceiling of "no" answers) that is the same for every set and target, rather than the near-certain answers the pure concept learner gives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_concept_learning_with_guessing.py`

### log_odds_distorted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 93.5 nats behind coherence_sharpened_decision_exemplar (5.3× dse)

**Hypothesis:** People are Bayesian concept learners over mathematical rules (powers, multiples, final and contained digits, primes, parity, squares, cubes) and numeric intervals with the size principle, but they misperceive their own posterior belief that the target belongs to the concept the way people distort probabilities generally: the belief is compressed or expanded linearly in log-odds around an inferred crossover point before it is turned into a yes/no answer. Weak beliefs are therefore over-endorsed and near-certain ones under-endorsed (or the reverse), while the ordering of targets is still set entirely by the normative posterior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_odds_distorted_concept_learning.py`

### tversky_contrast_feature_similarity — pruned (experiment1 end of experiment)

**Margin:** 93.7 nats behind coherence_sharpened_decision_exemplar (5.4× dse)

**Hypothesis:** People judge a target by Tversky-style featural similarity to the examples: each number is mentally represented as a set of salient arithmetic features (odd/even, being a multiple of 3 through 10, square, cube, prime, power of two, its final digit and its decade), and a target's similarity to an example rises with the features they have in common and falls with the features each has that the other lacks, weighted asymmetrically. The probability of "yes" rises with the target's average featural similarity to the examples, so targets sharing a property with every example (e.g. a common factor) are endorsed sharply more, while closeness in size matters only through sharing a decade.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tversky_contrast_feature_similarity.py`

### tempered_size_principle_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 117.0 nats behind coherence_sharpened_decision_exemplar (5.5× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100, but they apply the size principle only partially — they treat the examples as weaker evidence than a fully strong-sampling learner would, so the evidence from the examples is tempered by one inferred strength (1 recovers the incumbent; below 1 means broader generalization). The single change is this evidence-strength exponent, applied equally to every example regardless of order; it predicts more "yes" answers to targets outside the tightest consistent concept, most of all after a single example, where the incumbent generalises too narrowly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_concept_learning.py`

### recency_forgetting_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 130.3 nats behind coherence_sharpened_decision_exemplar (5.8× dse)

**Hypothesis:** People are Bayesian concept learners (rules and intervals, size principle) who reason only from the examples they still hold in memory: the last-shown example is always remembered, and each earlier example is forgotten with a probability that grows the further back it was shown, so the concept is inferred from a recency-biased subset of the set. This disagrees most with the order-invariant best model on multi-example sets whose early examples point to a different concept than the last one: for targets close to the last example but outside the concept implied by the whole set it predicts many more "yes" answers, and for targets near only the first example, fewer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_forgetting_concept_learning.py`

### mixed_sampling_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 138.6 nats behind coherence_sharpened_decision_exemplar (5.7× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: the same Bayesian concept learner over 42 rules and all intervals of 1-100 (rule weights inferred separately, intervals equally likely, 10% noise examples), but people only partly believe the examples were sampled from the concept: each example is taken as drawn from the concept with an inferred probability (strong sampling, the size principle) and otherwise as an arbitrary number that merely happens to belong to it (weak sampling). The one change is this strong/weak sampling mixture, which weakens the size principle's penalty on large concepts, so people generalise more broadly than the incumbent predicts, most of all from a single example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/mixed_sampling_concept_learning.py`

### spread_scaled_size_exemplar — pruned (experiment2 end of experiment)

**Margin:** 154.5 nats behind zigzag_breadth_sequence_exemplar (6.4× dse)

**Hypothesis:** People estimate how variable the program's output is from how spread out the examples are, and generalize in size accordingly: they judge a target by its recency-weighted similarity to the remembered examples (size, final digit, parity, shared factor), but the size gradient around every example is stretched in proportion to the spread of the examples. A tightly clustered set (66, 78) is generalized narrowly in size and a widely spread set (10, 50, 100) very broadly, both inside and beyond its range, so whole sets differ sharply in how generously they are generalized.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/spread_scaled_size_exemplar.py`

### step_lattice_progression_inference — pruned (experiment1 end of experiment)

**Margin:** 160.8 nats behind coherence_sharpened_decision_exemplar (7.4× dse)

**Hypothesis:** People imagine the hidden program as counting in a fixed step: they infer the largest step size that would carry them from each example to every other (the greatest common divisor of the examples' differences) and reject targets that cannot be reached from the examples in whole steps, so after 13, 23, 43 only numbers ending in 3 qualify and after 12, 18, 30 only multiples of 6. How firmly they hold this stepping rule grows with how unlikely it is that the examples would line up on such a step by chance (more examples beyond the two that define the step, and a larger step); with one or two examples there is no confirmed step and they answer only by how close the target is to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/step_lattice_progression_inference.py`

### recency_decayed_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 178.3 nats behind coherence_sharpened_decision_exemplar (7.0× dse)

**Hypothesis:** People infer the hidden program by Bayesian concept learning over mathematical rules and numeric intervals, but their memory of the examples decays with recency: the most recently shown example counts as full evidence while each earlier example is discounted by a constant factor per position back. Concepts consistent with the later examples therefore dominate, so targets near or sharing structure with the last-shown example get more "yes" answers than those tied to the first-shown one, and evidence from long sets is weaker (broader generalization).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_decayed_concept_learning.py`

### shared_divisor_rule_abstraction — pruned (experiment1 end of experiment)

**Margin:** 192.3 nats behind coherence_sharpened_decision_exemplar (8.6× dse)

**Hypothesis:** People abstract a single explicit rule from the examples: "multiples of g", where g is the greatest common divisor the examples share. They endorse targets that are multiples of g and reject those that are not, and they trust this rule more the less likely it is that the examples would share g by coincidence (more examples and a larger g); when the examples share no factor above 1, people fall back on nothing but how close the target is to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_divisor_rule_abstraction.py`

### shared_property_coincidence_rule — pruned (experiment2 end of experiment)

**Margin:** 208.0 nats behind zigzag_breadth_sequence_exemplar (8.2× dse)

**Hypothesis:** People read the examples for the arithmetic properties they all have in common (same parity, same final digit, same decade, being multiples of the same number) and treat each shared property as evidence for a rule in proportion to how surprising it would be for that many examples to share it by coincidence. They then say "yes" to a target according to the surprise-weighted balance of shared properties it has versus lacks, discounted only by how far the target falls outside the range the examples span — no similarity to individual examples is computed.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/shared_property_coincidence_rule.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 250.2 nats behind coherence_sharpened_decision_exemplar (8.4× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### memoryless_win_stay_lose_shift_rules — pruned (experiment2 end of experiment)

**Margin:** 255.0 nats behind zigzag_breadth_sequence_exemplar (9.0× dse)

**Hypothesis:** People test one candidate rule at a time while reading the examples in order, in a win-stay, lose-shift fashion: the first example suggests a single rule (a final digit, a multiple of some number, odd or even, or a range of sizes, smaller rules being favoured), they keep that rule as long as each new example fits it, and when an example contradicts it they discard it and pick a fresh rule suggested by that contradicting example alone, without re-checking the earlier ones. They then answer "yes" if the target obeys the rule they ended up holding, so the final judgement is driven by the last examples that forced a switch and by whichever rule survived from the start, and the same examples shown in a different order lead to different generalisations.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/memoryless_win_stay_lose_shift_rules.py`

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Margin:** 260.2 nats behind coherence_sharpened_decision_exemplar (7.7× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_interval.py`

### recency_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 284.1 nats behind coherence_sharpened_decision_exemplar (7.7× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and 1-100 intervals (with a midpoint-dependent interval prior and the size principle), but they weigh the examples by recency — each example's evidence is discounted geometrically by how many examples were shown after it (one inferred decay rate, total evidence held fixed), so concepts consistent with the most recently shown examples gain posterior weight and targets near the last example are endorsed more. The single change is this order-dependent evidence weighting, motivated by the critique's recency discrepancy; a decay of 1 recovers the original model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_concept_learning.py`

### factorized_rule_and_range_inference — pruned (experiment2 end of experiment)

**Margin:** 330.7 nats behind zigzag_breadth_sequence_exemplar (10.4× dse)

**Hypothesis:** People are Bayesian concept learners who think the program's numbers obey both an arithmetic rule (a step pattern such as "even", "multiples of 3" or "ends in 7", or no pattern at all) and a numeric range, each inferred from the examples with the size principle, but — as a resource-saving distortion of joint inference — they work out the rule and the range as two separate, independent questions and say "yes" only to the extent that the target passes both. So a target is endorsed when it fits the inferred step pattern AND lies within the inferred range: far-away numbers sharing the examples' final digit or factor are rejected, as are nearby numbers that break the pattern, and the yes-probability is the product of the two separate judgements.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/factorized_rule_and_range_inference.py`

### log_scale_magnitude_range_inference — pruned (experiment2 end of experiment)

**Margin:** 374.7 nats behind zigzag_breadth_sequence_exemplar (10.7× dse)

**Hypothesis:** People do not look for arithmetic rules or compare the target with each example; they infer the range of magnitudes the program produces, reasoning about size on a compressed, logarithmic mental number line. They favour the narrowest range (in ratio terms) that covers all the examples, more strongly the more examples there are, and say "yes" in proportion to their belief that the target falls inside that range — so generalization extends further above the largest example than below the smallest (equal ratios span more numbers upward), and tightly clustered sets are generalized narrowly while widely spread sets are generalized broadly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/pruned/log_scale_magnitude_range_inference.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 512.8 nats behind coherence_sharpened_decision_exemplar (7.1× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
