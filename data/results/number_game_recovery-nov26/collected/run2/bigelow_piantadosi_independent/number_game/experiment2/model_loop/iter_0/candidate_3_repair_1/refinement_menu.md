# Refinement menu

The models you may refine, other than the incumbent `capacity_limited_evidence_integration`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### exposure_weakened_size_principle — rank 1, 10.2 ± 5.3 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -2150.4

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but the strength with which they apply the size principle is not fixed — each further question about the same set of examples within a session weakens it (a fitted power-law decay with the number of earlier trials on that set), so the learner who is repeatedly probed about one set comes to treat its examples as less diagnostic of a small concept and generalizes more broadly. The single change is making the sampling strength depend on the set's exposure history, motivated by the critique that first encounters of a set are judged less generously than later ones, which an exposure-blind learner cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/exposure_weakened_size_principle.py`

### familiarity_relaxed_size_principle — rank 2, 14.5 ± 5.8 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2154.7

**Hypothesis:** People do the same recency-weighted Bayesian concept learning over rules and numerical ranges as the best model, but how strongly they apply the size principle depends on how familiar the example set is: the first time a participant meets a set they read its examples as tightly sampled from a small concept, and each further time they are asked about that same set (having already been probed with other targets for it) they discount that sampling evidence more, so their generalization broadens. This predicts fewer yeses on a set's first encounter than on its repeats, concentrated on targets outside the tightest range or rule covering the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/familiarity_relaxed_size_principle.py`

### tempered_learned_rule_prior — rank 3, 19.7 ± 32.2 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -2159.9

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free sampling-strength exponent (1 = full strong sampling, 0 = weak sampling). The single change is grafting this tempered size principle (from `tempered_size_principle_mixture`) onto the learned-rule-prior model, predicting that broad favoured rules and wide intervals keep more posterior weight after a few examples than an ideal strong-sampling learner allows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/tempered_learned_rule_prior.py`

### recency_family_prior_sharpened_response — rank 4, 22.7 ± 6.7 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2162.9

**Hypothesis:** Refinement of `recency_weighted_even_odd_family_prior`: people do the same recency-weighted Bayesian concept learning over the same 45 rules (separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not answer by probability matching to their belief that the target is in the concept — they turn that belief into a yes/no answer more (or less) decisively, following the belief's log-odds scaled by a free decisiveness exponent. The single change is grafting this sharpened belief-to-response mapping (from `response_sharpened_tempered_rule_prior`) onto the incumbent, predicting that targets the learner judges probably in (or probably out of) the concept are answered yes (or no) more consistently across people than the graded posterior itself implies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/recency_family_prior_sharpened_response.py`

### response_sharpened_tempered_rule_prior — rank 5, 23.4 ± 32.4 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2163.5

**Hypothesis:** Refinement of `tempered_learned_rule_prior`: people do the same Bayesian concept learning over 45 rules (each with its own learned prior plausibility) and all intervals of 1-100 with a tempered size principle, but they turn their belief that the target is in the concept into a yes/no answer more decisively than probability matching — their answer follows the belief's log-odds scaled by a free decisiveness exponent, so targets the learner thinks are probably in (or probably out of) the concept are answered yes (or no) more consistently than the belief itself. The single change is this sharpened belief-to-response mapping (a one-parameter log-odds scaling, from `llo_distorted_concept_posterior`) grafted onto `tempered_learned_rule_prior`, motivated by the critique that people's repeated answers to the same set and target are more unanimous than the graded posterior predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/response_sharpened_tempered_rule_prior.py`

### learned_rule_prior_mixture — rank 6, 64.2 ± 38.2 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -2204.4

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people learn concepts by the same Bayesian inference over 45 mathematical rules and all intervals of 1-100 with the size principle, but they do not hold the rules as equally likely a priori — each rule (e.g. odd, even, ends in a digit) carries its own prior plausibility, inferred from the data. The single change is replacing the uniform prior over rules with a learned, non-uniform one (the core idea of Bigelow & Piantadosi's prior inference), so favoured rules such as parity and shared final digits can dominate interval-based similarity when the examples support them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/learned_rule_prior_mixture.py`

### magnitude_privileged_size_principle — no comparison row

**Hypothesis:** People infer the hidden concept by Bayesian learning over mathematical rules (with separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, with capacity-limited integration of the examples, but they mis-weight the two kinds of evidence: the "suspicious coincidence" that the examples all fall in a narrow stretch of the number line is applied more strongly than the coincidence that they all obey an arithmetic rule, because magnitude is perceived effortlessly while rule membership must be computed. This predicts that when no rule fits well (for example sets mixing odd and even numbers) people confine their yeses to the span of the examples more sharply than a learner who applies one sampling strength to every concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/magnitude_privileged_size_principle.py`

### kind_specific_sampling_assumption — no comparison row

**Hypothesis:** People infer the hidden concept by Bayesian learning over mathematical rules (with separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, but they make different sampling assumptions for the two kinds of concept: they treat the examples as deliberately drawn from a magnitude range (a strong size principle, so the tightest range covering the examples wins sharply), while they treat a mathematical rule as merely compatible with the examples, discounting how suspiciously the examples fit a small rule (a separately fitted, weaker size principle). This predicts sharp yes/no edges at the examples' smallest and largest values when no rule explains them (e.g. mixed-parity sets), while broad rules such as parity still survive when the examples do fit them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/kind_specific_sampling_assumption.py`

### surprise_weighted_sequential_updating — no comparison row

**Hypothesis:** People read the examples one at a time, revising a running Bayesian guess about the concept (over mathematical rules, with separate prior plausibilities for even, odd and the other kinds of rule, and numerical ranges), and how much each new example counts depends on how surprising it is given the guess formed from the examples before it: an example the current guess predicted well is skimmed over, while an example that violates it (breaks the parity pattern, falls outside the range seen so far) captures attention and is weighted more heavily. Because the weight depends on the preceding examples, the same set shown in a different order is generalized differently, and pattern-breaking sets such as mixed-parity ones end up with their range taken more sharply than an order-blind learner would allow.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/surprise_weighted_sequential_updating.py`

## Pruned models (out of the set; narrowest margin first)

### recency_weighted_even_odd_family_prior — pruned (experiment1 end of experiment)

**Margin:** 2.8 nats behind familiarity_relaxed_size_principle (2.1× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not weigh every example equally — they read the examples in order and each example's evidence counts more the later it was shown (a fitted order gradient, with the total amount of evidence held fixed). The single change is grafting this recency weighting of the examples (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that targets near the last-shown example get more yeses than targets near the first-shown one, which an order-blind learner cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_even_odd_family_prior.py`

### even_odd_split_family_prior — pruned (experiment1 end of experiment)

**Margin:** 4.1 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a tempered size principle and a prior over kinds of rule, but they do not treat "even" and "odd" as one kind of rule with a single shared plausibility — "even numbers" and "odd numbers" are separate kinds, each with its own prior plausibility inferred from the data. The single change is splitting the parity family in two, predicting that an all-even set of examples triggers parity generalization (yeses to distant same-parity targets) more readily than an all-odd set does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/even_odd_split_family_prior.py`

### recency_decay_even_odd_family_prior — pruned (experiment1 end of experiment)

**Margin:** 4.8 nats behind familiarity_relaxed_size_principle (2.1× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples one at a time and their memory of earlier examples fades, so each example's evidence counts less the more examples were shown after it. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that people say yes more to targets near the last-shown example than to targets near the first-shown one, which the order-blind incumbent cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_decay_even_odd_family_prior.py`

### stochastic_example_recall_sampling — pruned (experiment1 end of experiment)

**Margin:** 5.6 nats behind familiarity_relaxed_size_principle (2.7× dse)

**Hypothesis:** People do not reason from all the examples they were shown, but from the examples they can recall when judging the target: the most recent example is always remembered, while each earlier example is recalled only by chance, less often the further back it was shown, and the learner infers the hidden concept (by Bayesian rule-and-range inference) from whichever examples were recalled. Because the answer averages over these all-or-none recall outcomes, targets near the last example are endorsed more than targets near the first, and sets shown earlier in a sequence partly behave like smaller, broader-generalizing sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/stochastic_example_recall_sampling.py`

### arithmetic_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.9 nats behind familiarity_relaxed_size_principle (2.8× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning (tempered size principle, separate even and odd kinds, a prior over kinds of rule), but their space of concepts also includes regularly spaced sequences that do not start at zero — "every m-th number starting from r" (numbers leaving the same remainder r when divided by m, for spacings 3–12, e.g. 1, 5, 9, 13, …), a new kind of rule with its own prior plausibility inferred from the data. The single change is adding this arithmetic-progression kind, predicting that sets whose examples share a common spacing (such as 5, 9, 17, 33) earn yeses for distant targets continuing that spacing, which no multiple, parity or final-digit rule covers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arithmetic_progression_concepts.py`

### previous_set_intrusion_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 7.0 nats behind familiarity_relaxed_size_principle (3.2× dse)

**Hypothesis:** People infer the hidden concept by Bayesian rule-and-range inference (tempered size principle, a prior over kinds of rule with separate even and odd kinds), but the examples of the previously shown, different set linger in memory and intrude as extra, down-weighted evidence about the current set's concept (proactive interference between trials). This disagrees most sharply with the order-blind, trial-independent best model whenever a set follows an unrelated set: the pooled, partly conflicting examples push belief toward broad ranges and broad rules (more yeses to distant targets), or, when the previous set shares the current set's pattern (e.g. both even), toward that shared rule — so the same set and target get different answers depending on what was shown just before.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/previous_set_intrusion_concept_learning.py`

### rule_family_prior_tempered_mixture — pruned (experiment1 end of experiment)

**Margin:** 19.6 nats behind familiarity_relaxed_size_principle (4.0× dse)

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but they do not treat every rule as equally plausible a priori — they hold a prior over *kinds* of rule (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each kind having its own plausibility, inferred from the data, and the rules within a kind sharing it equally. The single change is this family-structured prior in place of the uniform rule prior: it predicts that whole kinds of rule (e.g. "contains the digit 2" or "multiples of 4") are rarely entertained even when examples fit them, while parity and final-digit rules compete strongly with numerical intervals.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_family_prior_tempered_mixture.py`

### erlang_interval_family_prior — pruned (experiment1 end of experiment)

**Margin:** 20.6 nats behind familiarity_relaxed_size_principle (3.8× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is grafting this size-graded interval prior (from `learned_rule_prior_erlang_intervals`) onto the incumbent in place of its uniform interval prior, predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them instead of collapsing onto the tightest covering range or spreading over the whole 1-100 line.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_family_prior.py`

### recency_weighted_rule_family_prior — pruned (experiment1 end of experiment)

**Margin:** 21.1 nats behind familiarity_relaxed_size_principle (4.0× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples in the order shown and their memory of earlier examples fades, so each example's evidence counts less the further back it was shown and the most recent examples dominate which concept is inferred. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the family-prior model, motivated by the critique that targets near the last-shown example are endorsed more than targets near the first-shown one, which no order-blind learner can produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_rule_family_prior.py`

### recency_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 47.2 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** People infer the hidden concept by Bayesian inference over mathematical rules (each with its own prior plausibility) and numerical intervals, but they read the examples one at a time and their memory of earlier examples fades: each example's evidence counts less the further back in the sequence it was shown, so the most recent examples dominate which concept is inferred. This predicts that the same set shown in a different order yields different generalization, with targets that fit the last examples (but not the first ones) endorsed more than an ideal, order-blind learner would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_concept_learning.py`

### first_example_win_stay_lose_sample — pruned (experiment1 end of experiment)

**Margin:** 50.6 nats behind familiarity_relaxed_size_principle (6.1× dse)

**Hypothesis:** People do not weigh all examples together; they form a guess about the concept from the first example they see and then test it against the later examples ("win-stay, lose-sample"). If every later example fits the guess formed from the first example, they keep it; only when an example contradicts it do they abandon it and re-infer the concept from all the examples. Because the first example's guess survives whenever it is consistent, broad concepts suggested by the first example (e.g. "even", a wide range around it) persist far more than an ideal learner allows, and the same set in a different order can yield different generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/first_example_win_stay_lose_sample.py`

### learned_rule_prior_erlang_intervals — pruned (experiment1 end of experiment)

**Margin:** 58.4 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100 with the full size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is replacing the incumbent's uniform prior over intervals with this size-graded interval prior (taken from `tenenbaum_rules_similarity`), predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them rather than collapsing onto the tightest range that covers them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/learned_rule_prior_erlang_intervals.py`

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Margin:** 160.0 nats behind familiarity_relaxed_size_principle (7.2× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free "sampling strength" exponent (1 = the incumbent's full strong sampling, 0 = weak sampling). With weaker sampling, broad rules such as odd/even and digit rules keep more posterior weight against tight intervals, predicting stronger parity generalization, more yeses to far targets and less pure numerical-proximity generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_mixture.py`

### llo_distorted_concept_posterior — pruned (experiment1 end of experiment)

**Margin:** 171.0 nats behind familiarity_relaxed_size_principle (7.9× dse)

**Hypothesis:** People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — but they misread their own belief when turning it into a yes/no answer: the posterior probability that the target is in the concept is distorted linearly in log-odds (Zhang & Maloney-style probability weighting), compressed toward a reference point. As a result, near-certain members get fewer yeses and near-certain non-members more yeses than the ideal learner predicts, while the ordering of targets is left intact.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/llo_distorted_concept_posterior.py`

### lapse_guess_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 200.8 nats behind familiarity_relaxed_size_principle (9.7× dse)

**Hypothesis:** People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — and normally answer by probability matching to their belief that the target is in the concept, but on a fixed fraction of trials they do not consult that belief at all and instead guess, with their own overall bias toward "yes" or "no". This lapse-and-guess decision rule predicts a floor of yeses for clear non-members and a ceiling below one for clear members (including the examples themselves), both set by the same lapse rate and guessing bias and independent of how certain the inference is.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/lapse_guess_concept_learning.py`

### tempered_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 207.7 nats behind familiarity_relaxed_size_principle (7.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but update conservatively — they under-weight the evidence of the examples relative to their prior (the example likelihood is raised to a fitted power below one, a weakened size principle). As a result broad concepts like "even" or "odd" and wide intervals survive a few examples better than for an ideal learner, producing more yeses to distant parity-matching targets and less sharply peaked generalization to numbers right next to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_rules_similarity.py`

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 211.1 nats behind familiarity_relaxed_size_principle (8.5× dse)

**Hypothesis:** People do not induce a hidden rule; they judge a target by its featural similarity to the remembered examples (exemplar generalization over symbolic number features). A target is accepted in proportion to its average similarity to the examples, where similarity decays with the number of salient surface features it fails to share with each example — its parity, its final digit, and its decade — rather than with its numerical distance.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_exemplar_similarity.py`

### digit_parity_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 219.9 nats behind familiarity_relaxed_size_principle (9.0× dse)

**Hypothesis:** People do not induce a hidden rule at all; they judge a target by its exemplar similarity to the examples shown, where two numbers are similar to the extent they share surface features (same parity, same final digit) and are close in magnitude, and "yes" grows with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian best model on targets that match only some examples' final digit or share the examples' parity without fitting any whole rule (graded yeses where the rule model says no), and on near-in-magnitude targets that break the examples' parity/digit pattern (fewer yeses than interval hypotheses predict).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_parity_exemplar_similarity.py`

### ones_digit_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 244.0 nats behind familiarity_relaxed_size_principle (9.4× dse)

**Hypothesis:** People judge a target by comparing its final (ones) digit with the final digit of each example, ignoring magnitude and mathematical rules: a target is endorsed in proportion to how many examples it resembles in surface form, where an example with the identical final digit counts fully and an example whose final digit merely has the same parity (odd/even) counts partially. This graded, example-by-example digit matching predicts yeses for targets that share a final digit with only some of the examples and for parity-matching targets far from every example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/ones_digit_exemplar_similarity.py`

### recency_weighted_category_exemplar — pruned (experiment1 end of experiment)

**Margin:** 246.8 nats behind familiarity_relaxed_size_principle (9.4× dse)

**Hypothesis:** People do not infer a hidden concept; they judge a target by its summed similarity to the remembered examples (an exemplar account), where two numbers are similar to the extent they are close in magnitude and share membership in stored number categories (even, odd, primes, squares, powers, multiples, final digit, contained digit), with membership in small, distinctive categories counting more than in broad ones. The examples are held in memory with strengths that fade with how far back in the sequence they were shown, so the most recently seen examples dominate the comparison — predicting more yeses to targets resembling the last example than to targets resembling only the first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_weighted_category_exemplar.py`

### divisor_overlap_similarity — pruned (experiment1 end of experiment)

**Margin:** 350.0 nats behind familiarity_relaxed_size_principle (10.7× dse)

**Hypothesis:** People judge a target by its arithmetic similarity to the examples rather than by magnitude: two numbers are similar to the extent that they share divisors (the overlap of their sets of divisors greater than one), and the willingness to say "yes" rises smoothly with the target's average divisor-overlap similarity to the examples. Because even numbers all share the divisor 2 while odd numbers share no common divisor, this predicts strong parity-like generalization from even sets, much weaker generalization from odd sets, and graded yeses for targets sharing factors (e.g. multiples of 3 or 5) with some of the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/divisor_overlap_similarity.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 393.7 nats behind familiarity_relaxed_size_principle (8.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 536.0 nats behind familiarity_relaxed_size_principle (9.0× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`

### interval_only_size_principle — pruned (experiment1 end of experiment)

**Margin:** 659.8 nats behind familiarity_relaxed_size_principle (17.5× dse)

**Hypothesis:** People generalize from the examples purely by numerical magnitude, ignoring every arithmetic or digit rule: they infer which contiguous range of 1-100 the program draws from, favouring tighter ranges that still cover all the examples (the size principle) under a prior preference for ranges of a typical width, and say "yes" with the probability that the target lies inside the inferred range. This predicts near-certain yeses between the smallest and largest example, a smooth fall-off just outside them that sharpens as more examples are shown, and no special boost for targets that share the examples' parity, final digit or multiplicative structure.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_only_size_principle.py`
