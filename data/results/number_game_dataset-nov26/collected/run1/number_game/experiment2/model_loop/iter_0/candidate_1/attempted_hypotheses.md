# Tried before

28 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### digit_run_strict_criterion_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 8.4 nats behind coincidence_sharpened_exemplar; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 14

**Hypothesis:** Refinement of `graded_parity_variability_exemplar`: people still judge a target by its summed similarity to the remembered examples (magnitude and final-digit attention tuned to how variable the examples are, plus a fixed parity penalty), but how much similarity is enough to say "yes" depends on the set. The one change: when three or more examples all end in the same digit, people read the set as coming from a narrow, rule-bound program and raise their response criterion for every target, so such sets are endorsed less overall, and magnitude-near targets with the wrong digit are rejected especially often, than similarity alone predicts. Other sets keep the shared criterion.

### digit_run_sharpened_exemplar — rejected (experiment1 round 4 candidate 4 refine incumbent shared_factor_parity_exemplar)

**Outcome:** predicts like existing model 'digit_run_rule_attention_exemplar' (p_yes RMSE 0.00102 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of digit_run_rule_attention_exemplar, not a new hypothesis.

**Hypothesis:** Refinement of `shared_factor_parity_exemplar`: people still judge a target by its summed similarity to the remembered examples in a space of magnitude, final digit, parity and shared factors, with attention tuned to how variable the examples are. The one change is that when three or more examples all end in the same digit, people take the final digit to be the program's rule and attend to it much more sharply, so a target with a different final digit loses far more similarity to every example than graded digit variability alone allows. Targets that match the shared digit are unaffected. After "46, 76, 66" or "55, 15, 75, 45", a target close in magnitude but with a different final digit (e.g. 50 or 47) is rejected much more often than the incumbent predicts.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 484.7 nats behind coincidence_sharpened_exemplar (9.1× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 271.1 nats behind coincidence_sharpened_exemplar (6.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 207.1 nats behind coincidence_sharpened_exemplar (5.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.
