# Tried before

63 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### exemplar_magnitude_digit_similarity — pruned (experiment1 end of experiment)

**Outcome:** 57.7 nats behind coincidence_sharpened_exemplar (4.1× dse)

**Hypothesis:** People do not reason about hidden rules at all; they judge a target by its summed similarity to the remembered example numbers (an exemplar / GCM-style process). Each number is represented psychologically by its magnitude and by its final digit, similarity to an example decays exponentially with attention-weighted city-block distance on those two dimensions, and the probability of saying "yes" is the target's summed similarity relative to a fixed response criterion (Luce choice rule).

### digit_numeral_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 84.4 nats behind coincidence_sharpened_exemplar (5.3× dse)

**Hypothesis:** People do not induce a rule or interval at all; they judge the target by its summed similarity to the examples treated as two-digit numerals, where similarity to an example falls off with the distance between tens digits and drops by a fixed factor when the final digits differ. This predicts partial generalization by shared final digit even in mixed sets (e.g. 72 or 12 after "92,82,27,42") and graded interpolation through neighbouring decades, and disagrees most sharply with the rule-based Bayesian model on sets that form a mathematical rule (e.g. powers of 2 or multiples of 7), where it predicts no special endorsement of rule members that share neither tens nor final digits with any example.

### place_value_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 53.0 nats behind coincidence_sharpened_exemplar (4.1× dse)

**Hypothesis:** People do not induce a rule; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers is computed in a two-dimensional psychological space of overall magnitude and final digit. Similarity decays exponentially with the numeric distance between the numbers and drops by a fixed factor when their final digits differ, so targets near the examples or sharing an example's last digit feel like they "belong", and the more total similarity a target accrues the more likely the answer is yes.

### interval_width_prior_concepts — pruned (experiment1 end of experiment)

**Outcome:** 219.0 nats behind coincidence_sharpened_exemplar (5.8× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size-principle sampling with 10% noise, "yes" = posterior probability the concept contains the target), but the one change is that intervals are not equally likely a priori — an interval's prior weight decays exponentially with its width at a rate inferred from the data, so people favour tight ranges hugging the examples and generalize more sharply to numbers inside the example span than to those outside it.

### tempered_size_principle_concepts — pruned (experiment1 end of experiment)

**Outcome:** 143.3 nats behind coincidence_sharpened_exemplar (5.3× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 45 rules and all intervals of 1-100 with independently weighted rules, but people apply a weakened (or strengthened) size principle — the likelihood of an example inside a concept falls with the concept's size raised to a fitted exponent rather than exactly one over its size. The single change is that exponent; a value below one lets broad intervals spanning scattered examples compete with small coincidental rules, producing more generalization within the examples' range.

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Outcome:** 134.1 nats behind coincidence_sharpened_exemplar (5.6× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals of 1-100, but they apply the size principle only partially — the evidence each example provides is tempered by a single inferred exponent, so smaller concepts are favoured less steeply than strong sampling implies and people hedge between broad and narrow concepts. The one change is this free size-principle strength (an exponent on the example likelihood, applied identically to rules and intervals); "yes" is still the posterior probability that the concept contains the target.

### coherence_driven_digit_attention — pruned (experiment1 end of experiment)

**Outcome:** 35.6 nats behind coincidence_sharpened_exemplar (3.2× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples (magnitude and final digit), but how much attention they give to the final digit is set by reading the examples: the more the examples agree with one another on their final digit, the more attention shifts onto it. So a set like "15, 85, 45" makes people say yes almost only to numbers ending in 5, while a set with mixed final digits makes them generalize broadly by magnitude. This makes overall endorsement differ strongly between sets.

### memory_limited_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 152.8 nats behind coincidence_sharpened_exemplar (6.2× dse)

**Hypothesis:** People do Bayesian concept learning over rules (odd, even, primes, multiples, powers, final digits, digits contained) and intervals of 1-100 with the size principle, but under a working-memory limit: each example shown is retained only with some probability, and the learner reasons as if the forgotten examples were never seen. A judgment is therefore an average over the concepts implied by the remembered subsets of examples, so a single stray example (e.g. 27 among 92, 82, 42) often fails to veto a coherent rule, and sets differ sharply in endorsement depending on how robustly their examples point to one concept.

### variability_tuned_exemplar_attention — pruned (experiment1 end of experiment)

**Outcome:** 22.7 nats behind coincidence_sharpened_exemplar (2.4× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples on magnitude and final digit, but they set their attention to each dimension from how variable the examples are on it: widely spread examples make them generalize broadly in magnitude while tightly clustered examples make magnitude matter sharply, and examples that share a final digit make a final-digit mismatch heavily penalized. So the same target can be endorsed after a scattered set and rejected after a homogeneous one, making sets differ strongly in overall endorsement.

### parity_place_value_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 45.1 nats behind coincidence_sharpened_exemplar (3.5× dse)

**Hypothesis:** Refinement of `place_value_exemplar_similarity`: people still judge a target by its summed similarity to the remembered examples, but the psychological space in which numbers are compared has a third dimension, parity — similarity to an example drops by an additional fixed factor when the target and that example differ in being odd or even. The single change is this parity dimension, so after mostly-even (or mostly-odd) examples, same-parity targets far from every example still feel like they "belong", which the magnitude-and-final-digit space cannot produce.

### digit_consistency_attention_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 34.1 nats behind coincidence_sharpened_exemplar (3.1× dse)

**Hypothesis:** Refinement of `place_value_exemplar_similarity`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude and final digit, but how much attention they pay to the final digit adapts to the examples shown. The one change is that the more consistently the examples share a final digit (e.g. "46, 76, 66"), the more heavily a final-digit mismatch reduces similarity, so such sets produce narrow, digit-bound generalization, while sets with varied final digits are generalized mainly by magnitude.

### digit_consistency_attention_exemplar_2 — pruned (experiment1 end of experiment)

**Outcome:** 46.0 nats behind coincidence_sharpened_exemplar (3.6× dse)

**Hypothesis:** Refinement of `exemplar_magnitude_digit_similarity`: people still judge a target by its summed similarity to the remembered examples on magnitude and final digit, and say "yes" by a Luce ratio of that similarity to a fixed criterion. The one change is in attention: how much weight people give the final digit depends on how consistently the examples share final digits. When the examples agree on their last digit (e.g. 78, 48, 88), a mismatching last digit cuts similarity much more sharply. So these sets generalize narrowly to numbers with the shared digit, while sets whose last digits vary generalize mostly by magnitude.

### serial_position_weighted_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 31.0 nats behind coincidence_sharpened_exemplar (3.1× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples (on magnitude and final digit, with attention tuned to how variable the examples are), but the examples are not remembered equally: memory strength changes steadily with serial position, so either the most recent examples (recency) or the first ones (primacy) dominate the judgment. A target close to the last-shown example is therefore endorsed more (under recency) than an equally close target near the first-shown example, with the direction and strength of this order effect inferred from the data.

### variability_tuned_parity_attention — pruned (experiment1 end of experiment)

**Outcome:** 14.0 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 14

**Hypothesis:** People judge a target by its summed similarity to the remembered examples in a psychological space of magnitude, final digit and parity (odd/even), and they set their attention to every one of these dimensions from how variable the examples are on it: scattered examples make magnitude generalization broad, a shared final digit makes a final-digit mismatch heavily penalized, and examples that are all even (or all odd) make an opposite-parity target feel dissimilar. So after an all-even set, even targets far from every example and sharing no final digit with them are still endorsed more than odd ones, while after mixed-parity sets parity plays almost no role.

### similarity_judgment_with_random_lapses — pruned (experiment1 end of experiment)

**Outcome:** 49.1 nats behind coincidence_sharpened_exemplar (3.8× dse)

**Hypothesis:** People judge a target by its summed similarity to the remembered examples on magnitude and final digit, but on a fraction of trials they lapse — they do not consult the examples at all and simply guess yes or no at random (a coin flip). So even targets identical or very close to an example sometimes get "no", and targets dissimilar to every example sometimes get "yes", at a floor and ceiling set by one lapse rate rather than by similarity.

### parity_tuned_exemplar_attention — pruned (experiment1 end of experiment)

**Outcome:** 14.8 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 14

**Hypothesis:** Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change is that parity (odd vs even) becomes a further dimension of the same kind: the more uniformly the examples share parity (e.g. all even), the more a target of the opposite parity loses similarity, so after same-parity sets people endorse same-parity targets even far from every example and sharing no final digit, while mixed-parity sets give parity no weight.

### variability_tuned_parity_attention_2 — rejected (experiment1 round 2 candidate 4 refine incumbent variability_tuned_exemplar_attention)

**Outcome:** predicts like existing model 'parity_tuned_exemplar_attention' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of parity_tuned_exemplar_attention, not a new hypothesis.

**Hypothesis:** Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change is that the same variability-driven attention also reaches parity (odd/even): the more the examples agree in parity (e.g. all even), the more a parity mismatch with an example lowers similarity, so after same-parity sets, same-parity targets far from every example and sharing no final digit are still endorsed more than opposite-parity ones, while mixed-parity sets ignore parity.

### parity_space_digit_tuned_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 18.7 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 14

**Hypothesis:** Refinement of `parity_place_value_exemplar_similarity` (one graft from `digit_consistency_attention_exemplar`): people judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity (odd/even), but the attention they pay to the final digit adapts to the examples — the more consistently the examples share a final digit, the more a final-digit mismatch reduces similarity. The parity dimension lets same-parity targets far from every example still feel like they belong (the parity contrast the best model misses), while the adaptive digit attention makes digit-coherent sets like "46, 76, 66" generalize narrowly.

### graded_parity_variability_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 13.7 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 14

**Hypothesis:** Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change, grafted from `parity_place_value_exemplar_similarity`, is that parity (odd/even) is a fixed, always-attended dimension of the similarity space, not one whose weight depends on the set: every example of the opposite parity to the target contributes less similarity by the same factor, so final digits are compared in a graded way (same digit > same parity > opposite parity). This predicts that same-parity targets far from every example and sharing no final digit are endorsed more after same-parity sets, and also that in mixed-parity sets a target is generalized mostly from its same-parity examples rather than parity being ignored.

### shared_regularity_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 12.5 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 14

**Hypothesis:** Besides generalizing by similarity to the remembered examples (magnitude, final digit and parity, with attention to magnitude and final digit tuned by how variable the examples are), people notice the explicit arithmetic and digit regularities that all of the examples share — multiples of k, powers, squares, cubes, primes, odd/even, ending in a given digit — and weigh them by how surprising they are (the fewer numbers a regularity covers, the more it counts). A target that conforms to the most specific shared regularity is endorsed more, and one that breaks the most specific shared regularity is rejected more, so rule-coherent sets (e.g. "8, 32", "78, 48, 88", "3, 63") generalize to rule members far from every example and reject close non-members, making sets differ more in endorsement than similarity alone allows.

### cohesion_referenced_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 48.6 nats behind coincidence_sharpened_exemplar (4.1× dse)

**Hypothesis:** People judge a target by its similarity to the remembered examples (on magnitude, final digit and parity), but they do not compare that similarity to a fixed standard: they compare it to how similar the examples are to one another, asking "does this number fit in with the examples as well as they fit with each other?". So a tightly coherent set (e.g. 46, 76, 66, or 12, 14, 16) sets a high bar and yields narrow generalization and low overall endorsement, while a scattered set sets a low bar and lets many targets in, making overall willingness to say yes vary strongly from set to set.

### feature_distribution_prototype — pruned (experiment1 end of experiment)

**Outcome:** 49.1 nats behind coincidence_sharpened_exemplar (3.5× dse)

**Hypothesis:** People do not store examples or test rules; they summarise the examples into a category description — an estimated bell-shaped distribution of magnitudes (centred on the examples' mean, as wide as their spread plus some prior uncertainty) and estimated frequencies of final digits and of odd/even, smoothed toward chance — and they say yes in proportion to how much more probable the target is under that category description than under a random number. Because tight sets yield a narrow, peaked category and digit-coherent sets yield a sharp digit profile, sets differ strongly in how readily any target is accepted, and a magnitude-near target with the wrong final digit is rejected after a digit-homogeneous set.

### set_specific_criterion_exemplar — rejected (experiment1 round 3 candidate 3 refine incumbent graded_parity_variability_exemplar)

**Outcome:** p_yes could not be predicted on the 512-stimulus novelty pool (IndexError: index 201 is out of bounds for axis 0 with size 200 Apply node that caused the error: AdvancedSubtensor1(crit_z, set_idx) Toposort index: 24 Inputs types: [TensorType(float64, shape=(200,)), TensorType(int32, shape=(None,))] Inputs shapes: [(200,), (512,)] Inputs strides: [(8,), (4,)] Inputs values: ['not shown', 'not shown'] Outputs clients: [[Composite{clip(sigmoid((i0 + (i1 * i2) + (i3 * (i4 + log(i5))))), 1e-06, 0.999999)}(ExpandDims{axis=0}.0, ExpandDims{axis=0}.0, AdvancedSubtensor1.0, ExpandDims{axis=0}.0, Max{axis=1}.0, Sum{axis=1}.0)]] Backtrace when the node is created (use PyTensor flag traceback__limit=N to make it longer): File "/scr/wurgaft/auto-psych/number_game_dataset-nov26-tiger3/harness_repo/src/pipelines/inner_loop/model_zoo.py", line 822, in _min_prediction_rmse candidate_p = posterior_mean_p_left(model_name) File "/scr/wurgaft/auto-psych/number_game_dataset-nov26-tiger3/harness_repo/src/pipelines/inner_loop/model_zoo.py", line 810, in posterior_mean_p_left fitted = fit_model( File "/scr/wurgaft/auto-psych/number_game_dataset-nov26-tiger3/harness_repo/src/models/pymc_inference.py", line 822, in fit_model fitted = _fit_once_within( File "/scr/wurgaft/auto-psych/number_game_dataset-nov26-tiger3/harness_repo/src/models/pymc_inference.py", line 949, in _fit_once_within return _fit_once(name, models_dir, responses_path, settings, cache_dir) File "/scr/wurgaft/auto-psych/number_game_dataset-nov26-tiger3/harness_repo/src/models/pymc_inference.py", line 989, in _fit_once model = load_pymc_model(name, models_dir) File "/scr/wurgaft/auto-psych/number_game_dataset-nov26-tiger3/harness_repo/src/models/model_loading.py", line 108, in load_pymc_model mod = _exec_model_module(py_path, mod_prefix="_pymc_model_") File "/scr/wurgaft/auto-psych/number_game_dataset-nov26-tiger3/harness_repo/src/models/model_loading.py", line 90, in _exec_model_module exec(compile(source, str(py_path), "exec"), mod.__dict__) File "/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment1/model_loop/models/set_specific_criterion_exemplar.py", line 73, in <module> crit = crit_sd * crit_z[set_idx] HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.). The model must define p_yes (a pm.Deterministic named 'p_yes') with one value per stimulus row, and its compute_features or prepare_observed must work for every stimulus the design can show (e.g. {1} -> 79, {1} -> 83, {2} -> 16), including stimuli unlike the training ones.

**Hypothesis:** Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples, with magnitude and final-digit attention tuned to how variable the examples are and a fixed parity penalty. The one change is that the response criterion is not shared across sets: on reading each example set, people adopt their own threshold for how much similarity is enough to say "yes" (some sets suggest a broad, permissive category, others a strict one), drawn from a population distribution whose spread is inferred. This predicts that sets differ in overall willingness to say yes more than similarity alone produces, while the ordering of targets within a set is still set by similarity.

### divisibility_kinship_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 188.7 nats behind round_number_averaging_exemplar (7.0× dse)

**Hypothesis:** Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples. Attention to magnitude and final digit is still set by how variable the examples are, and parity is still a fixed dimension. The one change is that the parity dimension is widened into divisibility kinship: an example also contributes less similarity when it and the target have no common factor at all (they are coprime), so two numbers that share a factor such as 3, 5 or 7 feel related even when their parity differs. This predicts that after sets like "3, 63", "6, 66" or "55, 15, 75, 45", targets that share a factor with the examples (30, 36, 39, 60, 50) are endorsed well above what magnitude, final digit and parity alone allow, while targets coprime to the examples (e.g. 37 after "8, 32") are rejected more often.

### shared_factor_parity_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 185.7 nats behind round_number_averaging_exemplar (7.0× dse)

**Hypothesis:** Refinement of `variability_tuned_parity_attention`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit and parity, with attention to each set by how variable the examples are on it. The one change is a further, fixed "shared factor" dimension: a target that has no common divisor with an example (they are coprime) loses similarity to that example by a constant factor. So after "3, 63" people endorse 36 and 39 (they share the factor 3) although these match neither example's final digit nor magnitude, and after "8, 32" they reject 37, unlike targets that share a factor with the examples.

### digit_run_strict_criterion_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 8.4 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 14

**Hypothesis:** Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, plus a fixed parity penalty), but how much similarity is enough to say "yes" depends on the set. The one change: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and raise their response criterion for every target, so such sets are endorsed less overall, and magnitude-near targets with the wrong digit are rejected especially often, than similarity alone predicts. Other sets keep the shared criterion.

### coincidence_sharpened_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 63.0 nats behind round_number_averaging_exemplar (3.5× dse)

**Hypothesis:** People generalize from the examples by summed similarity (magnitude, final digit, parity, shared factors), but how sharply similarity falls off is set by how suspicious a coincidence the examples are: the less likely it is that the examples would share their most specific common property (all the same final digit, all multiples of 5, all the same parity) by chance, the more specific they take the program to be. So they tighten their generalization on every dimension at once. This disagrees most with the best model on coherent sets, e.g. "78, 48, 88" or "15, 85, 45" (near non-matching targets rejected far more), "90, 45" (multiples of 5 favoured more sharply), and on how much sets differ in overall endorsement, while single examples and incoherent sets are generalized as before.

### common_step_lattice_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 150.5 nats behind round_number_averaging_exemplar (6.1× dse)

**Hypothesis:** Refinement of `shared_factor_parity_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factors, with attention tuned to how variable the examples are), but they also read the spacing of the examples. When three or more examples are separated by gaps that are all multiples of one common step of at most 20 (e.g. step 10 in "78, 48, 88", step 6 in "63, 27, 99, 33", step 2 in "12, 14, 16"), people expect the program to move in that step, and a target that cannot be reached from the examples in whole steps feels dissimilar to every example by a fixed factor. So off-step targets near the examples are rejected more than similarity alone predicts, while sets with no such small common step are unaffected.

### capacity_limited_attention_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 180.5 nats behind round_number_averaging_exemplar (6.5× dse)

**Hypothesis:** People judge a target by its similarity to the remembered examples on magnitude, final digit, parity and shared factors, but their attention is a fixed, limited budget split across these dimensions: whatever the examples are consistent on (a shared final digit, a shared parity, a tight magnitude range) pulls attention onto that dimension and necessarily away from the others. So after a set that shares a final digit, people reject wrong-digit targets and also become more tolerant of magnitude distance for right-digit targets, because the attention spent on the digit is taken from magnitude, and the reverse holds after scattered-digit sets.

### digit_run_rule_attention_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 106.9 nats behind round_number_averaging_exemplar (4.6× dse)

**Hypothesis:** Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit, parity and shared factor, with attention to magnitude and final digit tuned by how variable the examples are. The one change, adapted from `digit_run_strict_criterion_exemplar` but placed inside the similarity space rather than the response criterion: when three or more examples all end in the same digit, people read the final digit as the program's rule, and their attention to it jumps. A target with a different final digit then loses much more similarity to every example, so after sets like "46, 76, 66" or "15, 85, 45" magnitude-near targets with the wrong digit are rejected far more often than graded digit attention allows. Targets that match the digit are not penalized.

### digit_run_sharpened_exemplar — rejected (experiment1 round 4 candidate 4 refine incumbent shared_factor_parity_exemplar)

**Outcome:** predicts like existing model 'digit_run_rule_attention_exemplar' (p_yes RMSE 0.00102 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of digit_run_rule_attention_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of `shared_factor_parity_exemplar`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit, parity and shared factors, with attention tuned to how variable the examples are. The one change is that when three or more examples all end in the same digit, people take the final digit to be the program's rule and attend to it much more sharply, so a target with a different final digit loses far more similarity to every example than graded digit variability alone allows. Targets that match the shared digit are unaffected. After "46, 76, 66" or "55, 15, 75, 45", a target close in magnitude but with a different final digit (e.g. 50 or 47) is rejected much more often than the incumbent predicts.

### digit_run_criterion_shared_factor — pruned (experiment2 end of experiment)

**Outcome:** 195.6 nats behind round_number_averaging_exemplar (7.2× dse)

**Hypothesis:** Refinement of `digit_run_strict_criterion_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, a fixed parity penalty), and still raise their response criterion when three or more examples all end in the same digit, reading such a set as a narrow rule. The one change, grafted from `shared_factor_parity_exemplar`, is a fixed shared-factor dimension: an example contributes less similarity to a target that shares no common divisor with it (they are coprime). This predicts strict rejection of wrong-digit targets after same-digit runs (e.g. "62, 32, 22") together with endorsement of factor-sharing targets far from the examples (36 and 39 after "3, 63") and rejection of coprime ones (37 after "8, 32").

### digit_run_raised_criterion_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 190.2 nats behind round_number_averaging_exemplar (7.1× dse)

**Hypothesis:** Refinement of the incumbent `shared_factor_parity_exemplar`. People still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity and shared factor, with attention tuned to how variable the examples are). The one change, grafted from `digit_run_strict_criterion_exemplar`: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and demand more similarity before saying yes to *any* target. So such sets are endorsed less across the board. That includes targets that do match the shared digit, which a sharper digit attention (`digit_run_rule_attention_exemplar`) would leave untouched. Wrong-digit targets near the examples are rejected far more often than the incumbent predicts.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 484.7 nats behind coincidence_sharpened_exemplar (9.1× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 271.1 nats behind coincidence_sharpened_exemplar (6.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 207.1 nats behind coincidence_sharpened_exemplar (5.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### log_odds_distorted_concept_learning — pruned (experiment2 end of experiment)

**Outcome:** 202.5 nats behind round_number_averaging_exemplar (7.2× dse)

**Hypothesis:** People infer the hidden program as a Bayesian learner would: they weigh candidate rules (parity, multiples of any number, powers, squares, cubes, primes, a shared final digit) and magnitude ranges by how well each explains the examples, favouring small concepts that make the examples a suspicious coincidence (the size principle), and they derive how probable it is that the target belongs. The one distortion is in how that probability is used: people perceive it in a compressed, log-odds-linear way (as in classic probability-weighting), so near-certain inclusions and exclusions are pulled strongly toward the middle. Targets the posterior rates as almost certain or almost impossible are therefore endorsed or rejected far less consistently than the normative account says.

### shared_feature_contrast — pruned (experiment2 end of experiment)

**Outcome:** 160.8 nats behind round_number_averaging_exemplar (6.1× dse)

**Hypothesis:** People do not compare the target to each example; they abstract the features that all of the examples have in common (odd/even, being a multiple of some number, a final digit, a decade, being a square, cube, power of two or prime) and weight each shared feature by how suspicious a coincidence it is (the rarer the feature among 1-100 and the more examples share it, the more it counts). They say yes to a target in proportion to how much weighted shared-feature evidence it satisfies, minus how much it violates, so sets with many specific common features (e.g. "8, 32", "3, 63", "78, 48, 88") produce sharp, rule-like acceptance of feature-satisfying targets anywhere on the number line and rejection of nearby violators, while sets with few common features are accepted only weakly.

### running_guess_coincidence_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 94.2 nats behind round_number_averaging_exemplar (4.6× dse)

**Hypothesis:** People read the examples one at a time and keep a running guess about the program's rule (a shared final digit, multiples of 5, a shared parity): each new example that fits the current guess makes it more convincing, but an example that breaks it makes them drop the guess and start again from that example, forgetting the earlier evidence. They then generalize by summed similarity to the examples (magnitude, final digit, parity, shared factors), sharpened on every dimension by how suspicious a coincidence the guess they end up with is. So order matters: "83, 38, 58, 68" ends on a three-example run of final digit 8 and gives narrow generalization, while the same numbers with the odd one last ("38, 58, 68, 83") would not.

### common_divisor_sharpened_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 45.5 nats behind round_number_averaging_exemplar (2.9× dse)

**Hypothesis:** Refinement of the incumbent `coincidence_sharpened_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, and a factor dimension), and they still sharpen this generalization on every dimension when the examples share a suspicious coincidence. The one change is to the factor dimension. People do not check pairwise whether the target shares any factor with each example. Instead they read the common divisor of the whole set (when the examples share one of 3 or more, e.g. 3 in "3, 63" or 10 in "10, 50, 100"), and a target that is not a multiple of it loses similarity to every example by a constant factor. So multiples of the set's common divisor that match no example's final digit are endorsed well above non-multiples, while single examples and sets without such a divisor carry no factor penalty.

### common_divisor_rule_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 56.8 nats behind round_number_averaging_exemplar (3.5× dse)

**Hypothesis:** Refinement of the incumbent `coincidence_sharpened_exemplar`: people still generalize by summed similarity to the examples (magnitude, final digit, parity and a shared-factor dimension), and they still sharpen it on every dimension when the examples share a suspicious property. The one change is to the shared-factor dimension. When the examples have a common divisor (e.g. 3 in "63, 27, 99, 33", 8 in "8, 32", 10 in "10, 80"), people read the set as "multiples of that number". A target that is not a multiple of it loses similarity to every example, even if it shares some other factor with one of them. Single examples and sets whose only common divisor is 1 keep the old per-example coprimality test. This predicts that far targets with no matching final digit are endorsed when they divide by the set's common divisor (36 after "3, 63") and rejected when they do not.

### accumulating_regularity_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 78.7 nats behind round_number_averaging_exemplar (3.8× dse)

**Hypothesis:** Refinement of the pruned `shared_regularity_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude, final digit, parity), and they still notice the most specific arithmetic or digit regularity that all the examples share (e.g. multiples of 3, powers of 2, ending in 8), favouring targets that conform to it and rejecting those that break it. The one change is that rule evidence now builds up with every example, as the size principle says it should: each further example that fits a surprising regularity makes it less likely to be a coincidence. So the push toward rule members and away from rule breakers grows with the number of examples beyond the first, and is no longer a fixed amount per set. Four multiples of 10 therefore support "multiples of 10" (e.g. 20 accepted, 47 rejected) much more strongly than two multiples of 3 support "multiples of 3", while a single example implies no rule.

### multiple_route_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 22.7 nats behind round_number_averaging_exemplar (2.4× dse)

**Hypothesis:** People judge a target by its summed similarity to the examples (closeness in magnitude, same final digit, same parity, and whether it divides by the set's common divisor), sharpened when the examples share a suspicious coincidence. On top of that there is a second, separate way for a target to resemble an example: being one of its multiples (14, 49 or 98 after "7"; 64 after "4"). This multiples route adds a fixed amount of similarity no matter how far apart the two numbers are, instead of only scaling down closeness. So multiples of an example that are far from it and share none of its digits are still often endorsed, especially after single examples and small-number examples, where magnitude similarity alone predicts almost certain rejection.

### rule_uncertainty_guessing — pruned (experiment2 end of experiment)

**Outcome:** 144.9 nats behind round_number_averaging_exemplar (7.0× dse)

**Hypothesis:** People judge a target by its plain summed similarity to the examples (magnitude, final digit, parity, and the set's common divisor), but whether they actually use that judgment is a decision-level choice that depends on how clear the program's rule looks. When the examples share no suspicious property (a common final digit, multiples of 5, a common divisor, a shared parity), people are unsure what the program does and on many trials just guess, saying yes at their own base rate. The clearer the shared rule, the less they guess. So rule-less sets (e.g. "4, 66, 78, 96") get a broad, flat background of yes answers, while rule-coherent sets (e.g. "80, 70, 90, 100") reject dissimilar targets consistently.

### divisor_family_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 40.1 nats behind round_number_averaging_exemplar (2.5× dse)

**Hypothesis:** Refinement of `common_divisor_sharpened_exemplar`: people still judge a target by its summed similarity to the examples (magnitude, final digit, parity, factor), sharpened by how suspicious the examples' shared coincidence is. The one change is to the space of arithmetic concepts they consider: they entertain "multiples of d" for every divisor d of 3 or more that all the examples share, including the divisors of a single lone example, not only the largest common divisor of a multi-example set. A target that is a multiple of none of these candidate divisors loses similarity to every example by a constant factor, so after a single "7" people favour 14, 49 and 98 over 17 or 47, and after "90, 45" they still accept 30 and 60 (multiples of 15).

### shared_divisor_factor_exemplar — rejected (experiment2 round 1 candidate 3 refine incumbent common_divisor_sharpened_exemplar)

**Outcome:** predicts like existing model 'divisor_family_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of divisor_family_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `common_divisor_sharpened_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and a factor dimension), with generalization sharpened on every dimension when the examples share a suspicious coincidence. The one change is to how the factor dimension is read. People take the common divisor of the examples (for a single example, the example itself) as a source of candidate "multiples of k" rules, and they accept any factor of 3 or more of it as such a rule. A target loses similarity to every example only when it shares no factor of 3 or more with that common divisor. So after a single example such as "7", multiples of 7 keep full similarity and other numbers are penalized. After "90, 45", 30 and 60 escape the penalty because they share 15 with the divisor 45. After "8, 64", even multiples of 4 that are not multiples of 8 also escape it.

### factor_kinship_sharpened_exemplar — rejected (experiment2 round 1 candidate 4 refine incumbent common_divisor_sharpened_exemplar)

**Outcome:** predicts like existing model 'divisor_family_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of divisor_family_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `common_divisor_sharpened_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, and a set-level factor dimension), sharpened on every dimension by how suspicious the examples' shared coincidence is. The one change is to the factor test. People do not demand that the target be an exact multiple of the set's common divisor. They ask only whether the target shares a factor of 3 or more with it, and a lone example counts as its own common divisor. So after "7" people endorse 14, 49 and 98 above other far targets, after "90, 45" they accept 30 and 60 (which share 15 with 45), and a target that shares no such factor loses similarity to every example by a constant amount.

### common_divisor_kinship_exemplar — rejected (experiment2 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'divisor_family_exemplar' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of divisor_family_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of `common_divisor_rule_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and a factor dimension), sharpened on every dimension when the examples share a suspicious coincidence, and they still read the factor dimension at the level of the whole set, from the examples' common divisor (for a single example, that number itself). The one change is that the factor test is kinship, not strict divisibility. When the set's common divisor is 3 or more, a target loses similarity to every example only if it shares no factor of 3 or more with that divisor. So 30 and 60 still fit "90, 45", 25 fits "80, 70, 90, 100", and after the single example "7" its multiples 14, 49 and 98 are endorsed while 15 is not, whereas a target that shares only the factor 2, or nothing, is penalized.

### common_divisor_membership_route — pruned (experiment2 end of experiment)

**Outcome:** 28.6 nats behind round_number_averaging_exemplar (2.6× dse)

**Hypothesis:** Refinement of the incumbent `common_divisor_sharpened_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, and a penalty for not being a multiple of the set's common divisor), sharpened on every dimension when the examples share a suspicious coincidence. The one change is that the set's common divisor of 3 or more (for a single example, the example itself) is also read as a rule, "multiples of that number", which gives the targets it covers a second, distance-free route to a yes: being a member adds a fixed amount to the target's summed similarity, rather than only sparing it a penalty. So multiples of the set's divisor that are far from every example and share none of their digits are often endorsed (e.g. 30 after "5", 14 or 70 after "28, 98", 39 after "63, 27, 99, 33"), while multiples of a single example in a set without that common divisor get no boost.

### similarity_averaging_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 36.5 nats behind round_number_averaging_exemplar (2.6× dse)

**Hypothesis:** Refinement of the incumbent `common_divisor_sharpened_exemplar`. People still judge a target by its similarity to the examples (magnitude, final digit, parity, and the set's common divisor), sharpened on every dimension when the examples share a suspicious coincidence. The one change is how evidence from several examples is pooled: people do not add up the similarity to every example, they (partly) average it, so each further example dilutes rather than adds to a target's support, and a target is judged by how well it fits the examples on average rather than how many it resembles. This predicts that adding examples does not make dissimilar targets more acceptable, that a far target in a sparse pair like "25, 82" is not carried by summing two weak similarities, and that single examples generalize relatively more broadly than multi-example sets, with the degree of averaging (from pure summing to pure averaging) inferred from the data.

### set_size_diluted_multiple_route — pruned (experiment2 end of experiment)

**Outcome:** 35.8 nats behind round_number_averaging_exemplar (3.5× dse)

**Hypothesis:** Refinement of `multiple_route_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and divisibility by the set's common divisor, sharpened when the examples share a suspicious coincidence), and being a multiple of an example is still a separate route to similarity that works at any distance. The one change is that this route is something people take mostly from a lone example: each further example shown weakens it by a constant factor, because with more examples people judge from what the examples have in common rather than from one example's multiples. So after the single example "7" or "5", far multiples (14, 49, 98; 30, 100) are often endorsed, but after "50, 2" or "55, 3" far multiples of the small example are seldom endorsed, and in four-example sets the route hardly matters.

### times_table_multiple_route — pruned (experiment2 end of experiment)

**Outcome:** 22.0 nats behind round_number_averaging_exemplar (2.2× dse)

**Hypothesis:** Refinement of `multiple_route_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and divisibility by the set's common divisor, sharpened when the examples share a suspicious coincidence), and being a multiple of an example is still a separate route to similarity that works at any distance. The one change is where that route comes from: it is retrieved times-table knowledge, so it exists only for examples that are themselves times-table numbers (2 to 12). After "7" people endorse 14, 49 and 98, but after "47" or "31" the far multiples 94 and 62 get no extra support and are judged by ordinary similarity alone.

### times_table_rule_sampling — pruned (experiment2 end of experiment)

**Outcome:** 279.2 nats behind round_number_averaging_exemplar (8.1× dse)

**Hypothesis:** People do not compare the target with each example by similarity; on each trial they come up with one guess about the program and answer by whether the target obeys it. The guess is a familiar property that every example has (odd or even, a times-table "multiples of 3 to 12", a shared final digit, squares, cubes or powers of two), or failing that just "numbers in the region the examples cover". Each kind of property comes to mind with its own ease, with smaller times tables easier, and properties become more likely than the region guess as more examples fit them. So after a lone "15" people often say yes to 12 or 9 (the 3- and 5-tables) but after "47" they rarely say yes to 94, and targets inside the examples' span are accepted as fully as the examples themselves.

### times_table_kinship_exemplar — pruned (experiment2 end of experiment)

**Outcome:** 39.2 nats behind round_number_averaging_exemplar (2.8× dse)

**Hypothesis:** People judge a target by its summed similarity to the examples (magnitude, final digit, parity, divisibility by the set's common divisor, sharpened when the examples share a suspicious coincidence), but the distance-free arithmetic route is not "being a multiple of the example": it is belonging to the same familiar times table, i.e. sharing a factor of 3 or more with an example, with the route's strength falling off as the smallest such shared factor gets larger (3, 4, 5, 7 tables are well known; a 47 table is not). This disagrees most sharply with the current best model on lone examples: it predicts factor-kin that are not multiples are often endorsed (12 or 27 after "15" or "30", 9 after "6"), while far multiples of large two-digit examples (94 after "47", 62 after "31") get little of the boost that multiples of small examples (49 after "7") get.

### factor_kin_multiple_route — pruned (experiment2 end of experiment)

**Outcome:** 18.6 nats behind round_number_averaging_exemplar (2.2× dse)

**Hypothesis:** Refinement of `multiple_route_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, divisibility by the set's common divisor, all sharpened when the examples share a suspicious coincidence). Being a multiple of an example is still a distance-free route to similarity. The one change is that this arithmetic route is broader than exact multiples. A target that shares a factor of 3 or more with an example also gains a separate, smaller fixed amount of similarity to it, at any distance, even when it is neither a multiple nor a divisor of that example (12 or 27 after "15", 9 after "6", 28 after "42"). So people generalize to the factor-kin of an example, not only to its multiples, while targets sharing no such factor with any example get no boost.

### times_table_multiple_route_2 — pruned (experiment2 end of experiment)

**Outcome:** 22.1 nats behind round_number_averaging_exemplar (2.3× dse)

**Hypothesis:** Refinement of the incumbent `multiple_route_exemplar`: people still judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences), and being a multiple of an example is still a separate route to a yes that works at any distance. The one change is that this route comes from memorised times tables. So it is strong for one-digit examples (14, 49 or 98 after "7"; 64 after "4") and much weaker for two-digit examples, whose multiples are not familiar facts (94 after "47", 62 after "31"). Far multiples of a two-digit example are therefore endorsed far less than far multiples of a one-digit one.

### times_table_multiple_route_3 — pruned (experiment2 end of experiment)

**Outcome:** 42.3 nats behind round_number_averaging_exemplar (3.0× dse)

**Hypothesis:** Refinement of `set_size_diluted_multiple_route`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and the set's common divisor, sharpened when the examples share a suspicious coincidence). Being a multiple of an example is still a separate route to similarity that works at any distance, and it still weakens with each further example. The one change is where the route comes from: it is retrieved from memorized times tables, so it works only for examples from 2 to 10 (49 or 98 after "7", 30 after "5", 64 after "4"). Multiples of larger examples, such as 94 after "47", 62 after "31" or 75 after "25", get no route and are judged on ordinary similarity alone.

### problem_size_multiple_retrieval — pruned (experiment2 end of experiment)

**Outcome:** 18.6 nats behind round_number_averaging_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 11

**Hypothesis:** People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences, plus a weaker route for factor-kin), and being a multiple of an example gives a distance-free route to a yes only to the extent that the multiplication fact linking them ("7 × 2 = 14") is retrieved from memory. Retrieval of arithmetic facts gets harder as the problem gets bigger (the problem-size effect), so the route weakens steadily with the sum of the two factors: 14 after "7" or 8 after "4" are readily endorsed, 98 after "7" less so, and 94 after "47" or 62 after "31" gain almost nothing, even though each is a small multiple.

### frequency_weighted_multiple_concepts — pruned (experiment2 end of experiment)

**Outcome:** 213.1 nats behind round_number_averaging_exemplar (7.6× dse)

**Hypothesis:** Refinement of `log_odds_distorted_concept_learning`. People still infer the hidden program as a Bayesian learner would: they weigh candidate rules (parity, primes, squares, cubes, powers, multiples of a number, a shared final digit) and magnitude ranges by the size principle, and they perceive the resulting probability that the target belongs in a compressed, log-odds way. The one change is in which arithmetic rules come to mind: "multiples of k" is a familiar idea for small times tables, and it becomes less available the larger k is (its prior falls off as a power of k). So after "47" the tiny rule "multiples of 47" hardly competes, and 94 gets little support, while after "7" the rule "multiples of 7" carries 14, 49 and 98.

### one_digit_table_factor_kin_route — pruned (experiment2 end of experiment)

**Outcome:** 16.8 nats behind round_number_averaging_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 11

**Hypothesis:** Refinement of the incumbent `factor_kin_multiple_route`. People still judge a target by its summed similarity to the examples: magnitude, final digit, parity and divisibility by the set's common divisor, all sharpened when the examples share a suspicious coincidence. A target that shares a factor of 3 or more with an example still gets a small similarity boost at any distance. The one change is to the multiples route. It is retrieved one-digit times-table knowledge, so only one-digit examples (2 to 9) give their multiples a distance-free boost (14, 49 or 98 after "7"; 64 after "4"). Far multiples of two-digit examples (94 after "47", 62 after "31", 75 after "25") get no multiples boost and are judged by ordinary similarity and factor-kinship alone.

### one_digit_arithmetic_route — pruned (experiment2 end of experiment)

**Outcome:** 16.9 nats behind round_number_averaging_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 11

**Hypothesis:** Refinement of the incumbent `factor_kin_multiple_route`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity and the set's common divisor, sharpened when the examples share a suspicious coincidence), with two distance-free arithmetic routes: being a multiple of an example, and sharing a factor of 3 or more with it. The one change is that both arithmetic routes are retrieved times-table knowledge about one-digit numbers, so an example of 10 or more gives both routes a much weaker boost by one common factor: after "7" or "6" people endorse 49, 98 or 9, but after "47" or "15" far targets like 94, 12 or 27 get little arithmetic support and are judged mostly by ordinary similarity.

### drifting_criterion_exemplar — rejected (experiment3 round 0 candidate 0 lens 8)

**Outcome:** predicts like existing model 'round_number_averaging_exemplar' (p_yes RMSE 0.00065 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of round_number_averaging_exemplar, not a new hypothesis.

**Hypothesis:** People judge each target by its similarity to the examples exactly as in the current best account, but the criterion they hold for saying "yes" is not fixed: it relaxes (or tightens) steadily over the course of the session, so the same set and target is endorsed more readily late in the session than early on. This is a decision-level shift in response bias over time, not a change in how the examples are generalized, and the direction and size of the drift are inferred from the data.

### round_set_kinship_averaging — rejected (experiment3 round 0 candidate 3 refine incumbent round_number_averaging_exemplar)

**Outcome:** predicts like existing model 'whole_set_round_category' (p_yes RMSE 0.00179 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of whole_set_round_category, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `round_number_averaging_exemplar`. Support is still pooled by partly averaging similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences), with distance-free routes for multiples and factor-kin. The one change is that people read round-number kinship from the set as a whole, not from any single example: a multiple-of-ten target gets the distance-free round boost only in proportion to how round the whole set looks (the share of its examples ending in 0 or 5, with how steeply this matters inferred from the data). So round sets ("45, 90", "5", "80, 70, 90, 100") still carry far round targets, but a single 5-ending example in a mixed set ("15, 11", "6, 25") lends 30, 60 or 100 almost no support.
