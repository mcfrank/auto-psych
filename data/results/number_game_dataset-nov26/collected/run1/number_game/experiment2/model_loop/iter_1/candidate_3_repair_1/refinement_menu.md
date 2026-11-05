# Refinement menu

The models you may refine, other than the incumbent `common_divisor_sharpened_exemplar`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### common_divisor_rule_exemplar — rank 1, 11.3 ± 8.1 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -2755.5

**Hypothesis:** Refinement of the incumbent `coincidence_sharpened_exemplar`: people still generalize by summed similarity to the examples (magnitude, final digit, parity and a shared-factor dimension), and they still sharpen it on every dimension when the examples share a suspicious property. The one change is to the shared-factor dimension. When the examples have a common divisor (e.g. 3 in "63, 27, 99, 33", 8 in "8, 32", 10 in "10, 80"), people read the set as "multiples of that number". A target that is not a multiple of it loses similarity to every example, even if it shares some other factor with one of them. Single examples and sets whose only common divisor is 1 keep the old per-example coprimality test. This predicts that far targets with no matching final digit are endorsed when they divide by the set's common divisor (36 after "3, 63") and rejected when they do not.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/common_divisor_rule_exemplar.py`

### coincidence_sharpened_exemplar — rank 2, 17.5 ± 10.2 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -2761.6

**Hypothesis:** People generalize from the examples by summed similarity (magnitude, final digit, parity, shared factors), but how sharply similarity falls off is set by how suspicious a coincidence the examples are: the less likely it is that the examples would share their most specific common property (all the same final digit, all multiples of 5, all the same parity) by chance, the more specific they take the program to be. So they tighten their generalization on every dimension at once. This disagrees most with the best model on coherent sets, e.g. "78, 48, 88" or "15, 85, 45" (near non-matching targets rejected far more), "90, 45" (multiples of 5 favoured more sharply), and on how much sets differ in overall endorsement, while single examples and incoherent sets are generalized as before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/coincidence_sharpened_exemplar.py`

### accumulating_regularity_exemplar — rank 3, 33.2 ± 16.7 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -2777.3

**Hypothesis:** Refinement of the pruned `shared_regularity_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity), and they still notice the most specific arithmetic or digit regularity that all the examples share (e.g. multiples of 3, powers of 2, ending in 8), favouring targets that conform to it and rejecting those that break it. The one change is that rule evidence now builds up with every example, as the size principle says it should: each further example that fits a surprising regularity makes it less likely to be a coincidence. So the push toward rule members and away from rule breakers grows with the number of examples beyond the first, and is no longer a fixed amount per set. Four multiples of 10 therefore support "multiples of 10" (e.g. 20 accepted, 47 rejected) much more strongly than two multiples of 3 support "multiples of 3", while a single example implies no rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/accumulating_regularity_exemplar.py`

### running_guess_coincidence_exemplar — rank 4, 48.7 ± 16.4 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2792.8

**Hypothesis:** People read the examples one at a time and keep a running guess about the program's rule (a shared final digit, multiples of 5, a shared parity): each new example that fits the current guess makes it more convincing, but an example that breaks it makes them drop the guess and start again from that example, forgetting the earlier evidence. They then generalize by summed similarity to the examples (magnitude, final digit, parity, shared factors), sharpened on every dimension by how suspicious a coincidence the guess they end up with is. So order matters: "83, 38, 58, 68" ends on a three-example run of final digit 8 and gives narrow generalization, while the same numbers with the odd one last ("38, 58, 68, 83") would not.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/running_guess_coincidence_exemplar.py`

### digit_run_rule_attention_exemplar — rank 5, 61.4 ± 20.2 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2805.5

**Hypothesis:** Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit, parity and shared factor, with attention to magnitude and final digit tuned by how variable the examples are. The one change, adapted from `digit_run_strict_criterion_exemplar` but placed inside the similarity space rather than the response criterion: when three or more examples all end in the same digit, people read the final digit as the program's rule, and their attention to it jumps. A target with a different final digit then loses much more similarity to every example, so after sets like "46, 76, 66" or "15, 85, 45" magnitude-near targets with the wrong digit are rejected far more often than graded digit attention allows. Targets that match the digit are not penalized.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/digit_run_rule_attention_exemplar.py`

### common_step_lattice_exemplar — rank 6, 105.0 ± 22.6 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2849.2

**Hypothesis:** Refinement of `shared_factor_parity_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factors, with attention tuned to how variable the examples are), but they also read the spacing of the examples. When three or more examples are separated by gaps that are all multiples of one common step of at most 20 (e.g. step 10 in "78, 48, 88", step 6 in "63, 27, 99, 33", step 2 in "12, 14, 16"), people expect the program to move in that step, and a target that cannot be reached from the examples in whole steps feels dissimilar to every example by a fixed factor. So off-step targets near the examples are rejected more than similarity alone predicts, while sets with no such small common step are unaffected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/common_step_lattice_exemplar.py`

### shared_feature_contrast — rank 7, 115.3 ± 24.9 nats behind the best (4.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2859.4

**Hypothesis:** People do not compare the target to each example; they abstract the features that all of the examples have in common (odd/even, being a multiple of some number, a final digit, a decade, being a square, cube, power of two or prime) and weight each shared feature by how suspicious a coincidence it is (the rarer the feature among 1-100 and the more examples share it, the more it counts). They say yes to a target in proportion to how much weighted shared-feature evidence it satisfies, minus how much it violates, so sets with many specific common features (e.g. "8, 32", "3, 63", "78, 48, 88") produce sharp, rule-like acceptance of feature-satisfying targets anywhere on the number line and rejection of nearby violators, while sets with few common features are accepted only weakly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/shared_feature_contrast.py`

### capacity_limited_attention_exemplar — rank 8, 135.0 ± 25.7 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2879.1

**Hypothesis:** People judge a target by its similarity to the remembered examples on magnitude, final digit, parity and shared factors, but their attention is a fixed, limited budget split across these dimensions: whatever the examples are consistent on (a shared final digit, a shared parity, a tight magnitude range) pulls attention onto that dimension and necessarily away from the others. So after a set that shares a final digit, people reject wrong-digit targets and also become more tolerant of magnitude distance for right-digit targets, because the attention spent on the digit is taken from magnitude, and the reverse holds after scattered-digit sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/capacity_limited_attention_exemplar.py`

### shared_factor_parity_exemplar — rank 9, 140.2 ± 23.4 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2884.3

**Hypothesis:** Refinement of `variability_tuned_parity_attention`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity, with attention to each set by how variable the examples are on it. The one change is a further, fixed "shared factor" dimension: a target that has no common divisor with an example (they are coprime) loses similarity to that example by a constant factor. So after "3, 63" people endorse 36 and 39 (they share the factor 3) although these match neither example's final digit nor magnitude, and after "8, 32" they reject 37, unlike targets that share a factor with the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/shared_factor_parity_exemplar.py`

### divisibility_kinship_exemplar — rank 10, 143.2 ± 23.6 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2887.3

**Hypothesis:** Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples. Attention to magnitude and final digit is still set by how variable the examples are, and parity is still a fixed dimension. The one change is that the parity dimension is widened into divisibility kinship: an example also contributes less similarity when it and the target have no common factor at all (they are coprime), so two numbers that share a factor such as 3, 5 or 7 feel related even when their parity differs. This predicts that after sets like "3, 63", "6, 66" or "55, 15, 75, 45", targets that share a factor with the examples (30, 36, 39, 60, 50) are endorsed well above what magnitude, final digit and parity alone allow, while targets coprime to the examples (e.g. 37 after "8, 32") are rejected more often.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/divisibility_kinship_exemplar.py`

### digit_run_raised_criterion_exemplar — rank 11, 144.7 ± 23.6 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2888.8

**Hypothesis:** Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factor, with attention tuned to how variable the examples are). The one change, grafted from `digit_run_strict_criterion_exemplar`: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and demand more similarity before saying yes to *any* target. So such sets are endorsed less across the board. That includes targets that do match the shared digit, which a sharper digit attention (`digit_run_rule_attention_exemplar`) would leave untouched. Wrong-digit targets near the examples are rejected far more often than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/digit_run_raised_criterion_exemplar.py`

### digit_run_criterion_shared_factor — rank 12, 150.1 ± 24.0 nats behind the best (6.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2894.3

**Hypothesis:** Refinement of `digit_run_strict_criterion_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, a fixed parity penalty), and still raise their response criterion when three or more examples all end in the same digit, reading such a set as a narrow rule. The one change, grafted from `shared_factor_parity_exemplar`, is a fixed shared-factor dimension: an example contributes less similarity to a target that shares no common divisor with it (they are coprime). This predicts strict rejection of wrong-digit targets after same-digit runs (e.g. "62, 32, 22") together with endorsement of factor-sharing targets far from the examples (36 and 39 after "3, 63") and rejection of coprime ones (37 after "8, 32").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/digit_run_criterion_shared_factor.py`

### log_odds_distorted_concept_learning — rank 13, 157.0 ± 28.9 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2901.1

**Hypothesis:** People infer the hidden program as a Bayesian learner would: they weigh candidate rules (parity, multiples of any number, powers, squares, cubes, primes, a shared final digit) and magnitude ranges by how well each explains the examples, favouring small concepts that make the examples a suspicious coincidence (the size principle), and they derive how probable it is that the target belongs. The one distortion is in how that probability is used: people perceive it in a compressed, log-odds-linear way (as in classic probability-weighting), so near-certain inclusions and exclusions are pulled strongly toward the middle. Targets the posterior rates as almost certain or almost impossible are therefore endorsed or rejected far less consistently than the normative account says.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/log_odds_distorted_concept_learning.py`

### multiple_route_exemplar — no comparison row

**Hypothesis:** People judge a target by its summed similarity to the examples (closeness in magnitude, same final digit, same parity, and whether it divides by the set's common divisor), sharpened when the examples share a suspicious coincidence. On top of that there is a second, separate way for a target to resemble an example: being one of its multiples (14, 49 or 98 after "7"; 64 after "4"). This multiples route adds a fixed amount of similarity no matter how far apart the two numbers are, instead of only scaling down closeness. So multiples of an example that are far from it and share none of its digits are still often endorsed, especially after single examples and small-number examples, where magnitude similarity alone predicts almost certain rejection.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/multiple_route_exemplar.py`

### rule_uncertainty_guessing — no comparison row

**Hypothesis:** People judge a target by its plain summed similarity to the examples (magnitude, final digit, parity, and the set's common divisor), but whether they actually use that judgment is a decision-level choice that depends on how clear the program's rule looks. When the examples share no suspicious property (a common final digit, multiples of 5, a common divisor, a shared parity), people are unsure what the program does and on many trials just guess, saying yes at their own base rate. The clearer the shared rule, the less they guess. So rule-less sets (e.g. "4, 66, 78, 96") get a broad, flat background of yes answers, while rule-coherent sets (e.g. "80, 70, 90, 100") reject dissimilar targets consistently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/rule_uncertainty_guessing.py`

### divisor_family_exemplar — no comparison row

**Hypothesis:** Refinement of `common_divisor_sharpened_exemplar`: people still judge a target by its summed similarity to the examples (magnitude, final digit, parity, factor), sharpened by how suspicious the examples' shared coincidence is. The one change is to the space of arithmetic concepts they consider: they entertain "multiples of d" for every divisor d of 3 or more that all the examples share, including the divisors of a single lone example, not only the largest common divisor of a multi-example set. A target that is a multiple of none of these candidate divisors loses similarity to every example by a constant factor, so after a single "7" people favour 14, 49 and 98 over 17 or 47, and after "90, 45" they still accept 30 and 60 (multiples of 15).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/divisor_family_exemplar.py`

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

### parity_space_digit_tuned_exemplar — pruned (experiment1 end of experiment)

**Margin:** 18.7 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 14

**Hypothesis:** Refinement of `parity_place_value_exemplar_similarity` (one graft from `digit_consistency_attention_exemplar`): people judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity (odd/even), but the attention they pay to the final digit adapts to the examples — the more consistently the examples share a final digit, the more a final-digit mismatch reduces similarity. The parity dimension lets same-parity targets far from every example still feel like they belong (the parity contrast the best model misses), while the adaptive digit attention makes digit-coherent sets like "46, 76, 66" generalize narrowly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_space_digit_tuned_exemplar.py`

### variability_tuned_exemplar_attention — pruned (experiment1 end of experiment)

**Margin:** 22.7 nats behind coincidence_sharpened_exemplar (2.4× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples on magnitude and final digit, but they set their attention to each dimension from how variable the examples are on it: widely spread examples make them generalize broadly in magnitude while tightly clustered examples make magnitude matter sharply, and examples that share a final digit make a final-digit mismatch heavily penalized. So the same target can be endorsed after a scattered set and rejected after a homogeneous one, making sets differ strongly in overall endorsement.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/variability_tuned_exemplar_attention.py`

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

### parity_place_value_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 45.1 nats behind coincidence_sharpened_exemplar (3.5× dse)

**Hypothesis:** Refinement of `place_value_exemplar_similarity`: people still judge a target by its summed similarity to the remembered examples, but the psychological space in which numbers are compared has a third dimension, parity — similarity to an example drops by an additional fixed factor when the target and that example differ in being odd or even. The single change is this parity dimension, so after mostly-even (or mostly-odd) examples, same-parity targets far from every example still feel like they "belong", which the magnitude-and-final-digit space cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/parity_place_value_exemplar_similarity.py`

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

### exemplar_magnitude_digit_similarity — pruned (experiment1 end of experiment)

**Margin:** 57.7 nats behind coincidence_sharpened_exemplar (4.1× dse)

**Hypothesis:** People do not reason about hidden rules at all; they judge a target by its summed similarity to the remembered example numbers (an exemplar / GCM-style process). Each number is represented psychologically by its magnitude and by its final digit, similarity to an example decays exponentially with attention-weighted city-block distance on those two dimensions, and the probability of saying "yes" is the target's summed similarity relative to a fixed response criterion (Luce choice rule).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_magnitude_digit_similarity.py`

### digit_numeral_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 84.4 nats behind coincidence_sharpened_exemplar (5.3× dse)

**Hypothesis:** People do not induce a rule or interval at all; they judge the target by its summed similarity to the examples treated as two-digit numerals, where similarity to an example falls off with the distance between tens digits and drops by a fixed factor when the final digits differ. This predicts partial generalization by shared final digit even in mixed sets (e.g. 72 or 12 after "92,82,27,42") and graded interpolation through neighbouring decades, and disagrees most sharply with the rule-based Bayesian model on sets that form a mathematical rule (e.g. powers of 2 or multiples of 7), where it predicts no special endorsement of rule members that share neither tens nor final digits with any example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_numeral_exemplar_similarity.py`

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Margin:** 134.1 nats behind coincidence_sharpened_exemplar (5.6× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals of 1-100, but they apply the size principle only partially — the evidence each example provides is tempered by a single inferred exponent, so smaller concepts are favoured less steeply than strong sampling implies and people hedge between broad and narrow concepts. The one change is this free size-principle strength (an exponent on the example likelihood, applied identically to rules and intervals); "yes" is still the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_mixture.py`

### tempered_size_principle_concepts — pruned (experiment1 end of experiment)

**Margin:** 143.3 nats behind coincidence_sharpened_exemplar (5.3× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 45 rules and all intervals of 1-100 with independently weighted rules, but people apply a weakened (or strengthened) size principle — the likelihood of an example inside a concept falls with the concept's size raised to a fitted exponent rather than exactly one over its size. The single change is that exponent; a value below one lets broad intervals spanning scattered examples compete with small coincidental rules, producing more generalization within the examples' range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_concepts.py`

### memory_limited_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 152.8 nats behind coincidence_sharpened_exemplar (6.2× dse)

**Hypothesis:** People do Bayesian concept learning over rules (odd, even, primes, multiples, powers, final digits, digits contained) and intervals of 1-100 with the size principle, but under a working-memory limit: each example shown is retained only with some probability, and the learner reasons as if the forgotten examples were never seen. A judgment is therefore an average over the concepts implied by the remembered subsets of examples, so a single stray example (e.g. 27 among 92, 82, 42) often fails to veto a coherent rule, and sets differ sharply in endorsement depending on how robustly their examples point to one concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/memory_limited_concept_learning.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 207.1 nats behind coincidence_sharpened_exemplar (5.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### interval_width_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 219.0 nats behind coincidence_sharpened_exemplar (5.8× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size-principle sampling with 10% noise, "yes" = posterior probability the concept contains the target), but the one change is that intervals are not equally likely a priori — an interval's prior weight decays exponentially with its width at a rate inferred from the data, so people favour tight ranges hugging the examples and generalize more sharply to numbers inside the example span than to those outside it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_width_prior_concepts.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 271.1 nats behind coincidence_sharpened_exemplar (6.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 484.7 nats behind coincidence_sharpened_exemplar (9.1× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
