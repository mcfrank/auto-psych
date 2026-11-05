# Refinement menu

The models you may refine, other than the incumbent `round_number_averaging_exemplar`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### recency_gated_arithmetic_route — rank 1, 20.8 ± 17.0 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -4213.3

**Hypothesis:** People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened when the examples share a suspicious coincidence), but the distance-free arithmetic routes (being a multiple or a factor-kin of an example, and round-number kinship) need the example to be actively held in mind while the target is checked, and examples fade from working memory as later ones arrive. So an example's arithmetic support falls off steadily with how many examples were shown after it: the multiples of the last-shown example (21 after "3, 7", 36 after "5, 9") are endorsed much more than equally far multiples of an earlier example (99 after "3, 33" is not carried by the 3), while ordinary magnitude and digit similarity is unaffected by order.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/recency_gated_arithmetic_route.py`

### round_target_familiarity_bias — rank 2, 51.4 ± 16.1 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4243.9

**Hypothesis:** People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, multiples and factor-kin of an example, all sharpened when the examples share a suspicious coincidence), but the extra appeal of round numbers is a property of the target itself, not a resemblance to round-ish examples: multiples of ten feel like natural outputs of any number-producing program, so people are more willing to say yes to them by a fixed amount whatever the examples are. This disagrees with `round_number_kinship_exemplar` on round targets after sets with no example ending in 0 or 5 (e.g. 30 or 60 after "63, 27, 99, 33" are boosted here, not there), and on round targets after 0/5 sets, which here get no extra boost from having more round-ish examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/round_target_familiarity_bias.py`

### round_number_kinship_exemplar — rank 3, 62.9 ± 16.0 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4255.5

**Hypothesis:** People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, multiples and factor-kin of an example, all sharpened when the examples share a suspicious coincidence), but they also treat round numbers as a familiar class of their own: a target that is a multiple of ten resembles every example ending in 0 or 5 by a fixed amount, at any distance and whatever its final digit. So round targets are endorsed well above what magnitude and digit similarity allow when the examples include numbers ending in 0 or 5 (100 after "5", 50 after "45, 90"), while targets ending in 5, and round targets after examples with no 0 or 5 ending, get no such boost (30 after "63, 27, 99, 33").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/round_number_kinship_exemplar.py`

### weber_scaled_magnitude_exemplar — rank 4, 66.5 ± 16.0 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4259.1

**Hypothesis:** People judge a target by its summed similarity to the examples (final digit, parity, the set's common divisor, multiples, factor-kin and round-number kinship, all sharpened when the examples share a suspicious coincidence), but the magnitude part of that similarity is read off a compressed mental number line that obeys Weber's law: how different two numbers feel depends on their gap relative to their size, not on the raw gap. So a gap of 10 makes a target feel much less like a small example (5 against 15) than like a large one (85 against 95), and generalization by magnitude is narrow around small examples and broad around large ones, with the degree of compression inferred from the data.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/weber_scaled_magnitude_exemplar.py`

### times_table_averaging_exemplar — rank 5, 105.9 ± 20.4 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4298.5

**Hypothesis:** Refinement of `times_table_multiple_route` (multiples route only for times-table examples 2-12), with one component grafted from `similarity_averaging_exemplar`: people pool their evidence across examples by (partly) averaging rather than summing it. A target's support — its similarity to each example (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences) plus the times-table multiples route — is diluted as more examples are shown, not accumulated. So a far target is not carried by adding up several weak resemblances in multi-example sets like "25, 82" or "50, 2", while lone examples like "7" or "5" keep generalizing broadly to their times-table multiples (14, 49, 98; 30, 100) and multiples of large examples (94 after "47") get no boost.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/times_table_averaging_exemplar.py`

### times_table_round_kinship_exemplar — rank 6, 106.5 ± 23.1 nats behind the best (4.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4299.1

**Hypothesis:** Refinement of the incumbent `round_number_kinship_exemplar`: people still judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences), with distance-free routes for factor-kin of an example and for round targets after examples ending in 0 or 5. The one change is that the multiples route is retrieved times-table knowledge, so it exists only for examples that are themselves times-table numbers (2 to 12): after "7" people endorse 14, 49 and 98, but far multiples of two-digit examples (94 after "47", 62 after "31", 75 after "25") get no multiples boost and are judged by ordinary similarity, factor-kinship and roundness alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/times_table_round_kinship_exemplar.py`

### times_table_round_averaging — rank 7, 119.6 ± 19.1 nats behind the best (6.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4312.1

**Hypothesis:** Refinement of `times_table_averaging_exemplar` (similarity to the examples on magnitude, final digit, parity and the set's common divisor, sharpened by suspicious coincidences, plus a distance-free multiples route only for times-table examples 2-12, with support partly averaged rather than summed over examples). The one change, grafted from `round_number_kinship_exemplar`: round numbers are a familiar class of their own, so a target that is a multiple of ten gains a fixed, distance-free amount of similarity to every example ending in 0 or 5. This predicts that round targets are endorsed well above magnitude and digit similarity after sets containing 0/5-ending examples (100 after "5", 50 after "45, 90"), while far multiples of large two-digit examples (94 after "47", 62 after "31") still get no multiples boost, unlike in the current best model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/times_table_round_averaging.py`

## Pruned models (out of the set; narrowest margin first)

### digit_run_strict_criterion_exemplar — pruned (experiment1 end of experiment)

**Margin:** 8.4 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 14

**Hypothesis:** Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, plus a fixed parity penalty), but how much similarity is enough to say "yes" depends on the set. The one change: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and raise their response criterion for every target, so such sets are endorsed less overall, and magnitude-near targets with the wrong digit are rejected especially often, than similarity alone predicts. Other sets keep the shared criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_run_strict_criterion_exemplar.py`

### shared_regularity_exemplar — pruned (experiment1 end of experiment)

**Margin:** 12.5 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 14

**Hypothesis:** Besides generalizing by similarity to the remembered examples (magnitude, final digit and parity, with attention to magnitude and final digit tuned by how variable the examples are), people notice the explicit arithmetic and digit regularities that all of the examples share — multiples of k, powers, squares, cubes, primes, odd/even, ending in a given digit — and weigh them by how surprising they are (the fewer numbers a regularity covers, the more it counts). A target that conforms to the most specific shared regularity is endorsed more, and one that breaks the most specific shared regularity is rejected more, so rule-coherent sets (e.g. "8, 32", "78, 48, 88", "3, 63") generalize to rule members far from every example and reject close non-members, making sets differ more in endorsement than similarity alone allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_regularity_exemplar.py`

### graded_parity_variability_exemplar — pruned (experiment1 end of experiment)

**Margin:** 13.7 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 14

**Hypothesis:** Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change, grafted from `parity_place_value_exemplar_similarity`, is that parity (odd/even) is a fixed, always-attended dimension of the similarity space, not one whose weight depends on the set: every example of the opposite parity to the target contributes less similarity by the same factor, so final digits are compared in a graded way (same digit > same parity > opposite parity). This predicts that same-parity targets far from every example and sharing no final digit are endorsed more after same-parity sets, and also that in mixed-parity sets a target is generalized mostly from its same-parity examples rather than parity being ignored.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/graded_parity_variability_exemplar.py`

### variability_tuned_parity_attention — pruned (experiment1 end of experiment)

**Margin:** 14.0 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 14

**Hypothesis:** People judge a target by its summed similarity to the remembered examples in a psychological space of magnitude, final digit and parity (odd/even), and they set their attention to every one of these dimensions from how variable the examples are on it: scattered examples make magnitude generalization broad, a shared final digit makes a final-digit mismatch heavily penalized, and examples that are all even (or all odd) make an opposite-parity target feel dissimilar. So after an all-even set, even targets far from every example and sharing no final digit with them are still endorsed more than odd ones, while after mixed-parity sets parity plays almost no role.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/variability_tuned_parity_attention.py`

### parity_tuned_exemplar_attention — pruned (experiment1 end of experiment)

**Margin:** 14.8 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 14

**Hypothesis:** Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change is that parity (odd vs even) becomes a further dimension of the same kind: the more uniformly the examples share parity (e.g. all even), the more a target of the opposite parity loses similarity, so after same-parity sets people endorse same-parity targets even far from every example and sharing no final digit, while mixed-parity sets give parity no weight.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_tuned_exemplar_attention.py`

### one_digit_table_factor_kin_route — pruned (experiment2 end of experiment)

**Margin:** 16.8 nats behind round_number_averaging_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 11

**Hypothesis:** Refinement of the incumbent `factor_kin_multiple_route`. People still judge a target by its summed similarity to the examples: magnitude, final digit, parity and divisibility by the set's common divisor, all sharpened when the examples share a suspicious coincidence. A target that shares a factor of 3 or more with an example still gets a small similarity boost at any distance. The one change is to the multiples route. It is retrieved one-digit times-table knowledge, so only one-digit examples (2 to 9) give their multiples a distance-free boost (14, 49 or 98 after "7"; 64 after "4"). Far multiples of two-digit examples (94 after "47", 62 after "31", 75 after "25") get no multiples boost and are judged by ordinary similarity and factor-kinship alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/one_digit_table_factor_kin_route.py`

### one_digit_arithmetic_route — pruned (experiment2 end of experiment)

**Margin:** 16.9 nats behind round_number_averaging_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 11

**Hypothesis:** Refinement of the incumbent `factor_kin_multiple_route`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and the set's common divisor, sharpened when the examples share a suspicious coincidence), with two distance-free arithmetic routes: being a multiple of an example, and sharing a factor of 3 or more with it. The one change is that both arithmetic routes are retrieved times-table knowledge about one-digit numbers, so an example of 10 or more gives both routes a much weaker boost by one common factor: after "7" or "6" people endorse 49, 98 or 9, but after "47" or "15" far targets like 94, 12 or 27 get little arithmetic support and are judged mostly by ordinary similarity.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/one_digit_arithmetic_route.py`

### factor_kin_multiple_route — pruned (experiment2 end of experiment)

**Margin:** 18.6 nats behind round_number_averaging_exemplar (2.2× dse)

**Hypothesis:** Refinement of `multiple_route_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, divisibility by the set's common divisor, all sharpened when the examples share a suspicious coincidence). Being a multiple of an example is still a distance-free route to similarity. The one change is that this arithmetic route is broader than exact multiples. A target that shares a factor of 3 or more with an example also gains a separate, smaller fixed amount of similarity to it, at any distance, even when it is neither a multiple nor a divisor of that example (12 or 27 after "15", 9 after "6", 28 after "42"). So people generalize to the factor-kin of an example, not only to its multiples, while targets sharing no such factor with any example get no boost.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/factor_kin_multiple_route.py`

### problem_size_multiple_retrieval — pruned (experiment2 end of experiment)

**Margin:** 18.6 nats behind round_number_averaging_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 11

**Hypothesis:** People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences, plus a weaker route for factor-kin), and being a multiple of an example gives a distance-free route to a yes only to the extent that the multiplication fact linking them ("7 × 2 = 14") is retrieved from memory. Retrieval of arithmetic facts gets harder as the problem gets bigger (the problem-size effect), so the route weakens steadily with the sum of the two factors: 14 after "7" or 8 after "4" are readily endorsed, 98 after "7" less so, and 94 after "47" or 62 after "31" gain almost nothing, even though each is a small multiple.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/problem_size_multiple_retrieval.py`

### parity_space_digit_tuned_exemplar — pruned (experiment1 end of experiment)

**Margin:** 18.7 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 14

**Hypothesis:** Refinement of `parity_place_value_exemplar_similarity` (one graft from `digit_consistency_attention_exemplar`): people judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity (odd/even), but the attention they pay to the final digit adapts to the examples — the more consistently the examples share a final digit, the more a final-digit mismatch reduces similarity. The parity dimension lets same-parity targets far from every example still feel like they belong (the parity contrast the best model misses), while the adaptive digit attention makes digit-coherent sets like "46, 76, 66" generalize narrowly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_space_digit_tuned_exemplar.py`

### times_table_multiple_route — pruned (experiment2 end of experiment)

**Margin:** 22.0 nats behind round_number_averaging_exemplar (2.2× dse)

**Hypothesis:** Refinement of `multiple_route_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and divisibility by the set's common divisor, sharpened when the examples share a suspicious coincidence), and being a multiple of an example is still a separate route to similarity that works at any distance. The one change is where that route comes from: it is retrieved times-table knowledge, so it exists only for examples that are themselves times-table numbers (2 to 12). After "7" people endorse 14, 49 and 98, but after "47" or "31" the far multiples 94 and 62 get no extra support and are judged by ordinary similarity alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/times_table_multiple_route.py`

### times_table_multiple_route_2 — pruned (experiment2 end of experiment)

**Margin:** 22.1 nats behind round_number_averaging_exemplar (2.3× dse)

**Hypothesis:** Refinement of the incumbent `multiple_route_exemplar`: people still judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences), and being a multiple of an example is still a separate route to a yes that works at any distance. The one change is that this route comes from memorised times tables. So it is strong for one-digit examples (14, 49 or 98 after "7"; 64 after "4") and much weaker for two-digit examples, whose multiples are not familiar facts (94 after "47", 62 after "31"). Far multiples of a two-digit example are therefore endorsed far less than far multiples of a one-digit one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/times_table_multiple_route_2.py`

### variability_tuned_exemplar_attention — pruned (experiment1 end of experiment)

**Margin:** 22.7 nats behind coincidence_sharpened_exemplar (2.4× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples on magnitude and final digit, but they set their attention to each dimension from how variable the examples are on it: widely spread examples make them generalize broadly in magnitude while tightly clustered examples make magnitude matter sharply, and examples that share a final digit make a final-digit mismatch heavily penalized. So the same target can be endorsed after a scattered set and rejected after a homogeneous one, making sets differ strongly in overall endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/variability_tuned_exemplar_attention.py`

### multiple_route_exemplar — pruned (experiment2 end of experiment)

**Margin:** 22.7 nats behind round_number_averaging_exemplar (2.4× dse)

**Hypothesis:** People judge a target by its summed similarity to the examples (closeness in magnitude, same final digit, same parity, and whether it divides by the set's common divisor), sharpened when the examples share a suspicious coincidence. On top of that there is a second, separate way for a target to resemble an example: being one of its multiples (14, 49 or 98 after "7"; 64 after "4"). This multiples route adds a fixed amount of similarity no matter how far apart the two numbers are, instead of only scaling down closeness. So multiples of an example that are far from it and share none of its digits are still often endorsed, especially after single examples and small-number examples, where magnitude similarity alone predicts almost certain rejection.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/multiple_route_exemplar.py`

### common_divisor_membership_route — pruned (experiment2 end of experiment)

**Margin:** 28.6 nats behind round_number_averaging_exemplar (2.6× dse)

**Hypothesis:** Refinement of the incumbent `common_divisor_sharpened_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, and a penalty for not being a multiple of the set's common divisor), sharpened on every dimension when the examples share a suspicious coincidence. The one change is that the set's common divisor of 3 or more (for a single example, the example itself) is also read as a rule, "multiples of that number", which gives the targets it covers a second, distance-free route to a yes: being a member adds a fixed amount to the target's summed similarity, rather than only sparing it a penalty. So multiples of the set's divisor that are far from every example and share none of their digits are often endorsed (e.g. 30 after "5", 14 or 70 after "28, 98", 39 after "63, 27, 99, 33"), while multiples of a single example in a set without that common divisor get no boost.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/common_divisor_membership_route.py`

### serial_position_weighted_exemplar — pruned (experiment1 end of experiment)

**Margin:** 31.0 nats behind coincidence_sharpened_exemplar (3.1× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples (on magnitude and final digit, with attention tuned to how variable the examples are), but the examples are not remembered equally: memory strength changes steadily with serial position, so either the most recent examples (recency) or the first ones (primacy) dominate the judgment. A target close to the last-shown example is therefore endorsed more (under recency) than an equally close target near the first-shown example, with the direction and strength of this order effect inferred from the data.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_exemplar.py`

### digit_consistency_attention_exemplar — pruned (experiment1 end of experiment)

**Margin:** 34.1 nats behind coincidence_sharpened_exemplar (3.1× dse)

**Hypothesis:** Refinement of `place_value_exemplar_similarity`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude and final digit, but how much attention they pay to the final digit adapts to the examples shown. The one change is that the more consistently the examples share a final digit (e.g. "46, 76, 66"), the more heavily a final-digit mismatch reduces similarity, so such sets produce narrow, digit-bound generalization, while sets with varied final digits are generalized mainly by magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_consistency_attention_exemplar.py`

### coherence_driven_digit_attention — pruned (experiment1 end of experiment)

**Margin:** 35.6 nats behind coincidence_sharpened_exemplar (3.2× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples (magnitude and final digit), but how much attention they give to the final digit is set by reading the examples: the more the examples agree with one another on their final digit, the more attention shifts onto it. So a set like "15, 85, 45" makes people say yes almost only to numbers ending in 5, while a set with mixed final digits makes them generalize broadly by magnitude. This makes overall endorsement differ strongly between sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/coherence_driven_digit_attention.py`

### set_size_diluted_multiple_route — pruned (experiment2 end of experiment)

**Margin:** 35.8 nats behind round_number_averaging_exemplar (3.5× dse)

**Hypothesis:** Refinement of `multiple_route_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and divisibility by the set's common divisor, sharpened when the examples share a suspicious coincidence), and being a multiple of an example is still a separate route to similarity that works at any distance. The one change is that this route is something people take mostly from a lone example: each further example shown weakens it by a constant factor, because with more examples people judge from what the examples have in common rather than from one example's multiples. So after the single example "7" or "5", far multiples (14, 49, 98; 30, 100) are often endorsed, but after "50, 2" or "55, 3" far multiples of the small example are seldom endorsed, and in four-example sets the route hardly matters.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/set_size_diluted_multiple_route.py`

### similarity_averaging_exemplar — pruned (experiment2 end of experiment)

**Margin:** 36.5 nats behind round_number_averaging_exemplar (2.6× dse)

**Hypothesis:** Refinement of the incumbent `common_divisor_sharpened_exemplar`. People still judge a target by its similarity to the examples (magnitude, final digit, parity, and the set's common divisor), sharpened on every dimension when the examples share a suspicious coincidence. The one change is how evidence from several examples is pooled: people do not add up the similarity to every example, they (partly) average it, so each further example dilutes rather than adds to a target's support, and a target is judged by how well it fits the examples on average rather than how many it resembles. This predicts that adding examples does not make dissimilar targets more acceptable, that a far target in a sparse pair like "25, 82" is not carried by summing two weak similarities, and that single examples generalize relatively more broadly than multi-example sets, with the degree of averaging (from pure summing to pure averaging) inferred from the data.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/similarity_averaging_exemplar.py`

### times_table_kinship_exemplar — pruned (experiment2 end of experiment)

**Margin:** 39.2 nats behind round_number_averaging_exemplar (2.8× dse)

**Hypothesis:** People judge a target by its summed similarity to the examples (magnitude, final digit, parity, divisibility by the set's common divisor, sharpened when the examples share a suspicious coincidence), but the distance-free arithmetic route is not "being a multiple of the example": it is belonging to the same familiar times table, i.e. sharing a factor of 3 or more with an example, with the route's strength falling off as the smallest such shared factor gets larger (3, 4, 5, 7 tables are well known; a 47 table is not). This disagrees most sharply with the current best model on lone examples: it predicts factor-kin that are not multiples are often endorsed (12 or 27 after "15" or "30", 9 after "6"), while far multiples of large two-digit examples (94 after "47", 62 after "31") get little of the boost that multiples of small examples (49 after "7") get.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/times_table_kinship_exemplar.py`

### divisor_family_exemplar — pruned (experiment2 end of experiment)

**Margin:** 40.1 nats behind round_number_averaging_exemplar (2.5× dse)

**Hypothesis:** Refinement of `common_divisor_sharpened_exemplar`: people still judge a target by its summed similarity to the examples (magnitude, final digit, parity, factor), sharpened by how suspicious the examples' shared coincidence is. The one change is to the space of arithmetic concepts they consider: they entertain "multiples of d" for every divisor d of 3 or more that all the examples share, including the divisors of a single lone example, not only the largest common divisor of a multi-example set. A target that is a multiple of none of these candidate divisors loses similarity to every example by a constant factor, so after a single "7" people favour 14, 49 and 98 over 17 or 47, and after "90, 45" they still accept 30 and 60 (multiples of 15).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/divisor_family_exemplar.py`

### times_table_multiple_route_3 — pruned (experiment2 end of experiment)

**Margin:** 42.3 nats behind round_number_averaging_exemplar (3.0× dse)

**Hypothesis:** Refinement of `set_size_diluted_multiple_route`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and the set's common divisor, sharpened when the examples share a suspicious coincidence). Being a multiple of an example is still a separate route to similarity that works at any distance, and it still weakens with each further example. The one change is where the route comes from: it is retrieved from memorized times tables, so it works only for examples from 2 to 10 (49 or 98 after "7", 30 after "5", 64 after "4"). Multiples of larger examples, such as 94 after "47", 62 after "31" or 75 after "25", get no route and are judged on ordinary similarity alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/times_table_multiple_route_3.py`

### parity_place_value_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 45.1 nats behind coincidence_sharpened_exemplar (3.5× dse)

**Hypothesis:** Refinement of `place_value_exemplar_similarity`: people still judge a target by its summed similarity to the remembered examples, but the psychological space in which numbers are compared has a third dimension, parity — similarity to an example drops by an additional fixed factor when the target and that example differ in being odd or even. The single change is this parity dimension, so after mostly-even (or mostly-odd) examples, same-parity targets far from every example still feel like they "belong", which the magnitude-and-final-digit space cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_place_value_exemplar_similarity.py`

### common_divisor_sharpened_exemplar — pruned (experiment2 end of experiment)

**Margin:** 45.5 nats behind round_number_averaging_exemplar (2.9× dse)

**Hypothesis:** Refinement of the incumbent `coincidence_sharpened_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, and a factor dimension), and they still sharpen this generalization on every dimension when the examples share a suspicious coincidence. The one change is to the factor dimension. People do not check pairwise whether the target shares any factor with each example. Instead they read the common divisor of the whole set (when the examples share one of 3 or more, e.g. 3 in "3, 63" or 10 in "10, 50, 100"), and a target that is not a multiple of it loses similarity to every example by a constant factor. So multiples of the set's common divisor that match no example's final digit are endorsed well above non-multiples, while single examples and sets without such a divisor carry no factor penalty.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/common_divisor_sharpened_exemplar.py`

### digit_consistency_attention_exemplar_2 — pruned (experiment1 end of experiment)

**Margin:** 46.0 nats behind coincidence_sharpened_exemplar (3.6× dse)

**Hypothesis:** Refinement of `exemplar_magnitude_digit_similarity`: people still judge a target by its summed similarity to the remembered examples on magnitude and final digit, and say "yes" by a Luce ratio of that similarity to a fixed criterion. The one change is in attention: how much weight people give the final digit depends on how consistently the examples share final digits. When the examples agree on their last digit (e.g. 78, 48, 88), a mismatching last digit cuts similarity much more sharply. So these sets generalize narrowly to numbers with the shared digit, while sets whose last digits vary generalize mostly by magnitude.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_consistency_attention_exemplar_2.py`

### cohesion_referenced_exemplar — pruned (experiment1 end of experiment)

**Margin:** 48.6 nats behind coincidence_sharpened_exemplar (4.1× dse)

**Hypothesis:** People judge a target by its similarity to the remembered examples (on magnitude, final digit and parity), but they do not compare that similarity to a fixed standard: they compare it to how similar the examples are to one another, asking "does this number fit in with the examples as well as they fit with each other?". So a tightly coherent set (e.g. 46, 76, 66, or 12, 14, 16) sets a high bar and yields narrow generalization and low overall endorsement, while a scattered set sets a low bar and lets many targets in, making overall willingness to say yes vary strongly from set to set.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/cohesion_referenced_exemplar.py`

### similarity_judgment_with_random_lapses — pruned (experiment1 end of experiment)

**Margin:** 49.1 nats behind coincidence_sharpened_exemplar (3.8× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples on magnitude and final digit, but on a fraction of trials they lapse — they do not consult the examples at all and simply guess yes or no at random (a coin flip). So even targets identical or very close to an example sometimes get "no", and targets dissimilar to every example sometimes get "yes", at a floor and ceiling set by one lapse rate rather than by similarity.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/similarity_judgment_with_random_lapses.py`

### feature_distribution_prototype — pruned (experiment1 end of experiment)

**Margin:** 49.1 nats behind coincidence_sharpened_exemplar (3.5× dse)

**Hypothesis:** People do not store examples or test rules; they summarise the examples into a category description — an estimated bell-shaped distribution of magnitudes (centred on the examples' mean, as wide as their spread plus some prior uncertainty) and estimated frequencies of final digits and of odd/even, smoothed toward chance — and they say yes in proportion to how much more probable the target is under that category description than under a random number. Because tight sets yield a narrow, peaked category and digit-coherent sets yield a sharp digit profile, sets differ strongly in how readily any target is accepted, and a magnitude-near target with the wrong final digit is rejected after a digit-homogeneous set.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/feature_distribution_prototype.py`

### place_value_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 53.0 nats behind coincidence_sharpened_exemplar (4.1× dse)

**Hypothesis:** People do not induce a rule; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers is computed in a two-dimensional psychological space of overall magnitude and final digit. Similarity decays exponentially with the numeric distance between the numbers and drops by a fixed factor when their final digits differ, so targets near the examples or sharing an example's last digit feel like they "belong", and the more total similarity a target accrues the more likely the answer is yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/place_value_exemplar_similarity.py`

### common_divisor_rule_exemplar — pruned (experiment2 end of experiment)

**Margin:** 56.8 nats behind round_number_averaging_exemplar (3.5× dse)

**Hypothesis:** Refinement of the incumbent `coincidence_sharpened_exemplar`: people still generalize by summed similarity to the examples (magnitude, final digit, parity and a shared-factor dimension), and they still sharpen it on every dimension when the examples share a suspicious property. The one change is to the shared-factor dimension. When the examples have a common divisor (e.g. 3 in "63, 27, 99, 33", 8 in "8, 32", 10 in "10, 80"), people read the set as "multiples of that number". A target that is not a multiple of it loses similarity to every example, even if it shares some other factor with one of them. Single examples and sets whose only common divisor is 1 keep the old per-example coprimality test. This predicts that far targets with no matching final digit are endorsed when they divide by the set's common divisor (36 after "3, 63") and rejected when they do not.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/common_divisor_rule_exemplar.py`

### exemplar_magnitude_digit_similarity — pruned (experiment1 end of experiment)

**Margin:** 57.7 nats behind coincidence_sharpened_exemplar (4.1× dse)

**Hypothesis:** People do not reason about hidden rules at all; they judge a target by its summed similarity to the remembered example numbers (an exemplar / GCM-style process). Each number is represented psychologically by its magnitude and by its final digit, similarity to an example decays exponentially with attention-weighted city-block distance on those two dimensions, and the probability of saying "yes" is the target's summed similarity relative to a fixed response criterion (Luce choice rule).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_magnitude_digit_similarity.py`

### coincidence_sharpened_exemplar — pruned (experiment2 end of experiment)

**Margin:** 63.0 nats behind round_number_averaging_exemplar (3.5× dse)

**Hypothesis:** People generalize from the examples by summed similarity (magnitude, final digit, parity, shared factors), but how sharply similarity falls off is set by how suspicious a coincidence the examples are: the less likely it is that the examples would share their most specific common property (all the same final digit, all multiples of 5, all the same parity) by chance, the more specific they take the program to be. So they tighten their generalization on every dimension at once. This disagrees most with the best model on coherent sets, e.g. "78, 48, 88" or "15, 85, 45" (near non-matching targets rejected far more), "90, 45" (multiples of 5 favoured more sharply), and on how much sets differ in overall endorsement, while single examples and incoherent sets are generalized as before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coincidence_sharpened_exemplar.py`

### accumulating_regularity_exemplar — pruned (experiment2 end of experiment)

**Margin:** 78.7 nats behind round_number_averaging_exemplar (3.8× dse)

**Hypothesis:** Refinement of the pruned `shared_regularity_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity), and they still notice the most specific arithmetic or digit regularity that all the examples share (e.g. multiples of 3, powers of 2, ending in 8), favouring targets that conform to it and rejecting those that break it. The one change is that rule evidence now builds up with every example, as the size principle says it should: each further example that fits a surprising regularity makes it less likely to be a coincidence. So the push toward rule members and away from rule breakers grows with the number of examples beyond the first, and is no longer a fixed amount per set. Four multiples of 10 therefore support "multiples of 10" (e.g. 20 accepted, 47 rejected) much more strongly than two multiples of 3 support "multiples of 3", while a single example implies no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/accumulating_regularity_exemplar.py`

### digit_numeral_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 84.4 nats behind coincidence_sharpened_exemplar (5.3× dse)

**Hypothesis:** People do not induce a rule or interval at all; they judge the target by its summed similarity to the examples treated as two-digit numerals, where similarity to an example falls off with the distance between tens digits and drops by a fixed factor when the final digits differ. This predicts partial generalization by shared final digit even in mixed sets (e.g. 72 or 12 after "92,82,27,42") and graded interpolation through neighbouring decades, and disagrees most sharply with the rule-based Bayesian model on sets that form a mathematical rule (e.g. powers of 2 or multiples of 7), where it predicts no special endorsement of rule members that share neither tens nor final digits with any example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_numeral_exemplar_similarity.py`

### running_guess_coincidence_exemplar — pruned (experiment2 end of experiment)

**Margin:** 94.2 nats behind round_number_averaging_exemplar (4.6× dse)

**Hypothesis:** People read the examples one at a time and keep a running guess about the program's rule (a shared final digit, multiples of 5, a shared parity): each new example that fits the current guess makes it more convincing, but an example that breaks it makes them drop the guess and start again from that example, forgetting the earlier evidence. They then generalize by summed similarity to the examples (magnitude, final digit, parity, shared factors), sharpened on every dimension by how suspicious a coincidence the guess they end up with is. So order matters: "83, 38, 58, 68" ends on a three-example run of final digit 8 and gives narrow generalization, while the same numbers with the odd one last ("38, 58, 68, 83") would not.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/running_guess_coincidence_exemplar.py`

### digit_run_rule_attention_exemplar — pruned (experiment2 end of experiment)

**Margin:** 106.9 nats behind round_number_averaging_exemplar (4.6× dse)

**Hypothesis:** Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit, parity and shared factor, with attention to magnitude and final digit tuned by how variable the examples are. The one change, adapted from `digit_run_strict_criterion_exemplar` but placed inside the similarity space rather than the response criterion: when three or more examples all end in the same digit, people read the final digit as the program's rule, and their attention to it jumps. A target with a different final digit then loses much more similarity to every example, so after sets like "46, 76, 66" or "15, 85, 45" magnitude-near targets with the wrong digit are rejected far more often than graded digit attention allows. Targets that match the digit are not penalized.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_run_rule_attention_exemplar.py`

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Margin:** 134.1 nats behind coincidence_sharpened_exemplar (5.6× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals of 1-100, but they apply the size principle only partially — the evidence each example provides is tempered by a single inferred exponent, so smaller concepts are favoured less steeply than strong sampling implies and people hedge between broad and narrow concepts. The one change is this free size-principle strength (an exponent on the example likelihood, applied identically to rules and intervals); "yes" is still the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_mixture.py`

### tempered_size_principle_concepts — pruned (experiment1 end of experiment)

**Margin:** 143.3 nats behind coincidence_sharpened_exemplar (5.3× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 45 rules and all intervals of 1-100 with independently weighted rules, but people apply a weakened (or strengthened) size principle — the likelihood of an example inside a concept falls with the concept's size raised to a fitted exponent rather than exactly one over its size. The single change is that exponent; a value below one lets broad intervals spanning scattered examples compete with small coincidental rules, producing more generalization within the examples' range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_concepts.py`

### rule_uncertainty_guessing — pruned (experiment2 end of experiment)

**Margin:** 144.9 nats behind round_number_averaging_exemplar (7.0× dse)

**Hypothesis:** People judge a target by its plain summed similarity to the examples (magnitude, final digit, parity, and the set's common divisor), but whether they actually use that judgment is a decision-level choice that depends on how clear the program's rule looks. When the examples share no suspicious property (a common final digit, multiples of 5, a common divisor, a shared parity), people are unsure what the program does and on many trials just guess, saying yes at their own base rate. The clearer the shared rule, the less they guess. So rule-less sets (e.g. "4, 66, 78, 96") get a broad, flat background of yes answers, while rule-coherent sets (e.g. "80, 70, 90, 100") reject dissimilar targets consistently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/rule_uncertainty_guessing.py`

### common_step_lattice_exemplar — pruned (experiment2 end of experiment)

**Margin:** 150.5 nats behind round_number_averaging_exemplar (6.1× dse)

**Hypothesis:** Refinement of `shared_factor_parity_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factors, with attention tuned to how variable the examples are), but they also read the spacing of the examples. When three or more examples are separated by gaps that are all multiples of one common step of at most 20 (e.g. step 10 in "78, 48, 88", step 6 in "63, 27, 99, 33", step 2 in "12, 14, 16"), people expect the program to move in that step, and a target that cannot be reached from the examples in whole steps feels dissimilar to every example by a fixed factor. So off-step targets near the examples are rejected more than similarity alone predicts, while sets with no such small common step are unaffected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/common_step_lattice_exemplar.py`

### memory_limited_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 152.8 nats behind coincidence_sharpened_exemplar (6.2× dse)

**Hypothesis:** People do Bayesian concept learning over rules (odd, even, primes, multiples, powers, final digits, digits contained) and intervals of 1-100 with the size principle, but under a working-memory limit: each example shown is retained only with some probability, and the learner reasons as if the forgotten examples were never seen. A judgment is therefore an average over the concepts implied by the remembered subsets of examples, so a single stray example (e.g. 27 among 92, 82, 42) often fails to veto a coherent rule, and sets differ sharply in endorsement depending on how robustly their examples point to one concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/memory_limited_concept_learning.py`

### shared_feature_contrast — pruned (experiment2 end of experiment)

**Margin:** 160.8 nats behind round_number_averaging_exemplar (6.1× dse)

**Hypothesis:** People do not compare the target to each example; they abstract the features that all of the examples have in common (odd/even, being a multiple of some number, a final digit, a decade, being a square, cube, power of two or prime) and weight each shared feature by how suspicious a coincidence it is (the rarer the feature among 1-100 and the more examples share it, the more it counts). They say yes to a target in proportion to how much weighted shared-feature evidence it satisfies, minus how much it violates, so sets with many specific common features (e.g. "8, 32", "3, 63", "78, 48, 88") produce sharp, rule-like acceptance of feature-satisfying targets anywhere on the number line and rejection of nearby violators, while sets with few common features are accepted only weakly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/shared_feature_contrast.py`

### capacity_limited_attention_exemplar — pruned (experiment2 end of experiment)

**Margin:** 180.5 nats behind round_number_averaging_exemplar (6.5× dse)

**Hypothesis:** People judge a target by its similarity to the remembered examples on magnitude, final digit, parity and shared factors, but their attention is a fixed, limited budget split across these dimensions: whatever the examples are consistent on (a shared final digit, a shared parity, a tight magnitude range) pulls attention onto that dimension and necessarily away from the others. So after a set that shares a final digit, people reject wrong-digit targets and also become more tolerant of magnitude distance for right-digit targets, because the attention spent on the digit is taken from magnitude, and the reverse holds after scattered-digit sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/capacity_limited_attention_exemplar.py`

### shared_factor_parity_exemplar — pruned (experiment2 end of experiment)

**Margin:** 185.7 nats behind round_number_averaging_exemplar (7.0× dse)

**Hypothesis:** Refinement of `variability_tuned_parity_attention`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity, with attention to each set by how variable the examples are on it. The one change is a further, fixed "shared factor" dimension: a target that has no common divisor with an example (they are coprime) loses similarity to that example by a constant factor. So after "3, 63" people endorse 36 and 39 (they share the factor 3) although these match neither example's final digit nor magnitude, and after "8, 32" they reject 37, unlike targets that share a factor with the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/shared_factor_parity_exemplar.py`

### divisibility_kinship_exemplar — pruned (experiment2 end of experiment)

**Margin:** 188.7 nats behind round_number_averaging_exemplar (7.0× dse)

**Hypothesis:** Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples. Attention to magnitude and final digit is still set by how variable the examples are, and parity is still a fixed dimension. The one change is that the parity dimension is widened into divisibility kinship: an example also contributes less similarity when it and the target have no common factor at all (they are coprime), so two numbers that share a factor such as 3, 5 or 7 feel related even when their parity differs. This predicts that after sets like "3, 63", "6, 66" or "55, 15, 75, 45", targets that share a factor with the examples (30, 36, 39, 60, 50) are endorsed well above what magnitude, final digit and parity alone allow, while targets coprime to the examples (e.g. 37 after "8, 32") are rejected more often.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/divisibility_kinship_exemplar.py`

### digit_run_raised_criterion_exemplar — pruned (experiment2 end of experiment)

**Margin:** 190.2 nats behind round_number_averaging_exemplar (7.1× dse)

**Hypothesis:** Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factor, with attention tuned to how variable the examples are). The one change, grafted from `digit_run_strict_criterion_exemplar`: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and demand more similarity before saying yes to *any* target. So such sets are endorsed less across the board. That includes targets that do match the shared digit, which a sharper digit attention (`digit_run_rule_attention_exemplar`) would leave untouched. Wrong-digit targets near the examples are rejected far more often than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_run_raised_criterion_exemplar.py`

### digit_run_criterion_shared_factor — pruned (experiment2 end of experiment)

**Margin:** 195.6 nats behind round_number_averaging_exemplar (7.2× dse)

**Hypothesis:** Refinement of `digit_run_strict_criterion_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, a fixed parity penalty), and still raise their response criterion when three or more examples all end in the same digit, reading such a set as a narrow rule. The one change, grafted from `shared_factor_parity_exemplar`, is a fixed shared-factor dimension: an example contributes less similarity to a target that shares no common divisor with it (they are coprime). This predicts strict rejection of wrong-digit targets after same-digit runs (e.g. "62, 32, 22") together with endorsement of factor-sharing targets far from the examples (36 and 39 after "3, 63") and rejection of coprime ones (37 after "8, 32").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_run_criterion_shared_factor.py`

### log_odds_distorted_concept_learning — pruned (experiment2 end of experiment)

**Margin:** 202.5 nats behind round_number_averaging_exemplar (7.2× dse)

**Hypothesis:** People infer the hidden program as a Bayesian learner would: they weigh candidate rules (parity, multiples of any number, powers, squares, cubes, primes, a shared final digit) and magnitude ranges by how well each explains the examples, favouring small concepts that make the examples a suspicious coincidence (the size principle), and they derive how probable it is that the target belongs. The one distortion is in how that probability is used: people perceive it in a compressed, log-odds-linear way (as in classic probability-weighting), so near-certain inclusions and exclusions are pulled strongly toward the middle. Targets the posterior rates as almost certain or almost impossible are therefore endorsed or rejected far less consistently than the normative account says.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/log_odds_distorted_concept_learning.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 207.1 nats behind coincidence_sharpened_exemplar (5.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### frequency_weighted_multiple_concepts — pruned (experiment2 end of experiment)

**Margin:** 213.1 nats behind round_number_averaging_exemplar (7.6× dse)

**Hypothesis:** Refinement of `log_odds_distorted_concept_learning`. People still infer the hidden program as a Bayesian learner would: they weigh candidate rules (parity, primes, squares, cubes, powers, multiples of a number, a shared final digit) and magnitude ranges by the size principle, and they perceive the resulting probability that the target belongs in a compressed, log-odds way. The one change is in which arithmetic rules come to mind: "multiples of k" is a familiar idea for small times tables, and it becomes less available the larger k is (its prior falls off as a power of k). So after "47" the tiny rule "multiples of 47" hardly competes, and 94 gets little support, while after "7" the rule "multiples of 7" carries 14, 49 and 98.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/frequency_weighted_multiple_concepts.py`

### interval_width_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 219.0 nats behind coincidence_sharpened_exemplar (5.8× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size-principle sampling with 10% noise, "yes" = posterior probability the concept contains the target), but the one change is that intervals are not equally likely a priori — an interval's prior weight decays exponentially with its width at a rate inferred from the data, so people favour tight ranges hugging the examples and generalize more sharply to numbers inside the example span than to those outside it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_width_prior_concepts.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 271.1 nats behind coincidence_sharpened_exemplar (6.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### times_table_rule_sampling — pruned (experiment2 end of experiment)

**Margin:** 279.2 nats behind round_number_averaging_exemplar (8.1× dse)

**Hypothesis:** People do not compare the target with each example by similarity; on each trial they come up with one guess about the program and answer by whether the target obeys it. The guess is a familiar property that every example has (odd or even, a times-table "multiples of 3 to 12", a shared final digit, squares, cubes or powers of two), or failing that just "numbers in the region the examples cover". Each kind of property comes to mind with its own ease, with smaller times tables easier, and properties become more likely than the region guess as more examples fit them. So after a lone "15" people often say yes to 12 or 9 (the 3- and 5-tables) but after "47" they rarely say yes to 94, and targets inside the examples' span are accepted as fully as the examples themselves.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/pruned/times_table_rule_sampling.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 484.7 nats behind coincidence_sharpened_exemplar (9.1× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
