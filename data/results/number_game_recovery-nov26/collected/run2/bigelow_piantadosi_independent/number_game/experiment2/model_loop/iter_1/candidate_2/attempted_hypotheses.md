# Tried before

31 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 211.1 nats behind familiarity_relaxed_size_principle (8.5× dse)

**Hypothesis:** People do not induce a hidden rule; they judge a target by its featural similarity to the remembered examples (exemplar generalization over symbolic number features). A target is accepted in proportion to its average similarity to the examples, where similarity decays with the number of salient surface features it fails to share with each example — its parity, its final digit, and its decade — rather than with its numerical distance.

### digit_parity_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 219.9 nats behind familiarity_relaxed_size_principle (9.0× dse)

**Hypothesis:** People do not induce a hidden rule at all; they judge a target by its exemplar similarity to the examples shown, where two numbers are similar to the extent they share surface features (same parity, same final digit) and are close in magnitude, and "yes" grows with the target's average similarity to the examples. This disagrees most sharply with the rule-based Bayesian best model on targets that match only some examples' final digit or share the examples' parity without fitting any whole rule (graded yeses where the rule model says no), and on near-in-magnitude targets that break the examples' parity/digit pattern (fewer yeses than interval hypotheses predict).

### ones_digit_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 244.0 nats behind familiarity_relaxed_size_principle (9.4× dse)

**Hypothesis:** People judge a target by comparing its final (ones) digit with the final digit of each example, ignoring magnitude and mathematical rules: a target is endorsed in proportion to how many examples it resembles in surface form, where an example with the identical final digit counts fully and an example whose final digit merely has the same parity (odd/even) counts partially. This graded, example-by-example digit matching predicts yeses for targets that share a final digit with only some of the examples and for parity-matching targets far from every example.

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Outcome:** 160.0 nats behind familiarity_relaxed_size_principle (7.2× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — they discount the evidence that the examples were sampled from a small concept, so the likelihood of the examples is tempered by a free "sampling strength" exponent (1 = the incumbent's full strong sampling, 0 = weak sampling). With weaker sampling, broad rules such as odd/even and digit rules keep more posterior weight against tight intervals, predicting stronger parity generalization, more yeses to far targets and less pure numerical-proximity generalization.

### hierarchical_rule_prior_mixture — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_mixture)

**Outcome:** MCMC did not converge (3 divergent transitions of 2000), even at target_accept 0.99 (the fitter already refits a near miss with smaller NUTS steps, so raising target_accept will not help). Change the model's geometry instead: non-centred parameterisations for hierarchical or scale parameters, tighter (weakly informative) priors on parameters the data barely constrain, fewer weakly identified parameters (drop or merge parameters that trade off against each other), and no hard thresholds or discontinuities in the likelihood.

**Hypothesis:** Refinement of bigelow_piantadosi_mixture: people learn concepts by the same Bayesian inference over 45 mathematical rules and all intervals of 1-100 with the size principle, but they do not hold the rules as equally likely a priori — each rule (e.g. odd, even, ends in a digit) carries its own prior plausibility, inferred from the data under a shared hierarchical spread. The single change is replacing the uniform prior over rules with a learned, non-uniform one (the core idea of Bigelow & Piantadosi's prior inference), so favoured rules such as parity and shared final digits can dominate interval-based similarity when the examples support them.

### tempered_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 207.7 nats behind familiarity_relaxed_size_principle (7.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but update conservatively — they under-weight the evidence of the examples relative to their prior (the example likelihood is raised to a fitted power below one, a weakened size principle). As a result broad concepts like "even" or "odd" and wide intervals survive a few examples better than for an ideal learner, producing more yeses to distant parity-matching targets and less sharply peaked generalization to numbers right next to the examples.

### recency_weighted_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 47.2 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** People infer the hidden concept by Bayesian inference over mathematical rules (each with its own prior plausibility) and numerical intervals, but they read the examples one at a time and their memory of earlier examples fades: each example's evidence counts less the further back in the sequence it was shown, so the most recent examples dominate which concept is inferred. This predicts that the same set shown in a different order yields different generalization, with targets that fit the last examples (but not the first ones) endorsed more than an ideal, order-blind learner would.

### llo_distorted_concept_posterior — pruned (experiment1 end of experiment)

**Outcome:** 171.0 nats behind familiarity_relaxed_size_principle (7.9× dse)

**Hypothesis:** People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — but they misread their own belief when turning it into a yes/no answer: the posterior probability that the target is in the concept is distorted linearly in log-odds (Zhang & Maloney-style probability weighting), compressed toward a reference point. As a result, near-certain members get fewer yeses and near-certain non-members more yeses than the ideal learner predicts, while the ordering of targets is left intact.

### interval_only_size_principle — pruned (experiment1 end of experiment)

**Outcome:** 659.8 nats behind familiarity_relaxed_size_principle (17.5× dse)

**Hypothesis:** People generalize from the examples purely by numerical magnitude, ignoring every arithmetic or digit rule: they infer which contiguous range of 1-100 the program draws from, favouring tighter ranges that still cover all the examples (the size principle) under a prior preference for ranges of a typical width, and say "yes" with the probability that the target lies inside the inferred range. This predicts near-certain yeses between the smallest and largest example, a smooth fall-off just outside them that sharpens as more examples are shown, and no special boost for targets that share the examples' parity, final digit or multiplicative structure.

### tempered_learned_rule_prior_2 — rejected (experiment1 round 1 candidate 4 refine incumbent learned_rule_prior_mixture)

**Outcome:** predicts like existing model 'tempered_learned_rule_prior' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of tempered_learned_rule_prior, not a new hypothesis.

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people do the same Bayesian concept learning over the 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100, but they apply the size principle only partially — the evidence that the examples were sampled from a small concept is tempered by a free "sampling strength" exponent (1 = full strong sampling, 0 = weak sampling). The single change is grafting this tempered likelihood from `tempered_size_principle_mixture` onto the learned rule prior, predicting that broad rules and wide intervals keep more posterior weight after a few examples than an ideal learner allows, so generalization is less sharply confined to the tightest consistent concept.

### rule_family_prior_tempered_mixture — pruned (experiment1 end of experiment)

**Outcome:** 19.6 nats behind familiarity_relaxed_size_principle (4.0× dse)

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but they do not treat every rule as equally plausible a priori — they hold a prior over *kinds* of rule (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each kind having its own plausibility, inferred from the data, and the rules within a kind sharing it equally. The single change is this family-structured prior in place of the uniform rule prior: it predicts that whole kinds of rule (e.g. "contains the digit 2" or "multiples of 4") are rarely entertained even when examples fit them, while parity and final-digit rules compete strongly with numerical intervals.

### learned_rule_prior_erlang_intervals — pruned (experiment1 end of experiment)

**Outcome:** 58.4 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** Refinement of `learned_rule_prior_mixture`: people learn concepts by the same Bayesian inference over 45 mathematical rules (each with its own learned prior plausibility) and all intervals of 1-100 with the full size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is replacing the incumbent's uniform prior over intervals with this size-graded interval prior (taken from `tenenbaum_rules_similarity`), predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them rather than collapsing onto the tightest range that covers them.

### first_example_win_stay_lose_sample — pruned (experiment1 end of experiment)

**Outcome:** 50.6 nats behind familiarity_relaxed_size_principle (6.1× dse)

**Hypothesis:** People do not weigh all examples together; they form a guess about the concept from the first example they see and then test it against the later examples ("win-stay, lose-sample"). If every later example fits the guess formed from the first example, they keep it; only when an example contradicts it do they abandon it and re-infer the concept from all the examples. Because the first example's guess survives whenever it is consistent, broad concepts suggested by the first example (e.g. "even", a wide range around it) persist far more than an ideal learner allows, and the same set in a different order can yield different generalization.

### divisor_overlap_similarity — pruned (experiment1 end of experiment)

**Outcome:** 350.0 nats behind familiarity_relaxed_size_principle (10.7× dse)

**Hypothesis:** People judge a target by its arithmetic similarity to the examples rather than by magnitude: two numbers are similar to the extent that they share divisors (the overlap of their sets of divisors greater than one), and the willingness to say "yes" rises smoothly with the target's average divisor-overlap similarity to the examples. Because even numbers all share the divisor 2 while odd numbers share no common divisor, this predicts strong parity-like generalization from even sets, much weaker generalization from odd sets, and graded yeses for targets sharing factors (e.g. multiples of 3 or 5) with some of the examples.

### lapse_guess_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 200.8 nats behind familiarity_relaxed_size_principle (9.7× dse)

**Hypothesis:** People infer the hidden concept as an ideal Bayesian learner would — over the same 45 mathematical rules (equally likely a priori) and all intervals of 1-100, with the size principle — and normally answer by probability matching to their belief that the target is in the concept, but on a fixed fraction of trials they do not consult that belief at all and instead guess, with their own overall bias toward "yes" or "no". This lapse-and-guess decision rule predicts a floor of yeses for clear non-members and a ceiling below one for clear members (including the examples themselves), both set by the same lapse rate and guessing bias and independent of how certain the inference is.

### even_odd_split_family_prior — pruned (experiment1 end of experiment)

**Outcome:** 4.1 nats behind familiarity_relaxed_size_principle (2.2× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a tempered size principle and a prior over kinds of rule, but they do not treat "even" and "odd" as one kind of rule with a single shared plausibility — "even numbers" and "odd numbers" are separate kinds, each with its own prior plausibility inferred from the data. The single change is splitting the parity family in two, predicting that an all-even set of examples triggers parity generalization (yeses to distant same-parity targets) more readily than an all-odd set does.

### even_odd_split_family_prior_2 — rejected (experiment1 round 2 candidate 4 refine incumbent rule_family_prior_tempered_mixture)

**Outcome:** predicts like existing model 'even_odd_split_family_prior' (p_yes RMSE 0.00062 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of even_odd_split_family_prior, not a new hypothesis.

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a tempered size principle and a prior over kinds of rule, but "even numbers" and "odd numbers" are not one kind sharing a single plausibility — each has its own prior plausibility inferred from the data. The single change is splitting the parity family into separate even and odd kinds, addressing the critique that people generalize the parity pattern more from all-even examples than from all-odd ones; it predicts stronger same-parity generalization (more yeses to distant even targets) after even sets than after comparable odd sets.

### erlang_interval_family_prior — pruned (experiment1 end of experiment)

**Outcome:** 20.6 nats behind familiarity_relaxed_size_principle (3.8× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not consider every numerical range equally plausible a priori — they expect ranges of a typical, intermediate width (an Erlang prior on interval size with a fitted scale, as in Tenenbaum's number game), so very narrow and very broad ranges are discounted before any examples are seen. The single change is grafting this size-graded interval prior (from `learned_rule_prior_erlang_intervals`) onto the incumbent in place of its uniform interval prior, predicting that magnitude-based generalization from one or two examples extends to a characteristic distance around them instead of collapsing onto the tightest covering range or spreading over the whole 1-100 line.

### arithmetic_progression_concepts — pruned (experiment1 end of experiment)

**Outcome:** 5.9 nats behind familiarity_relaxed_size_principle (2.8× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning (tempered size principle, separate even and odd kinds, a prior over kinds of rule), but their space of concepts also includes regularly spaced sequences that do not start at zero — "every m-th number starting from r" (numbers leaving the same remainder r when divided by m, for spacings 3–12, e.g. 1, 5, 9, 13, …), a new kind of rule with its own prior plausibility inferred from the data. The single change is adding this arithmetic-progression kind, predicting that sets whose examples share a common spacing (such as 5, 9, 17, 33) earn yeses for distant targets continuing that spacing, which no multiple, parity or final-digit rule covers.

### recency_weighted_category_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 246.8 nats behind familiarity_relaxed_size_principle (9.4× dse)

**Hypothesis:** People do not infer a hidden concept; they judge a target by its summed similarity to the remembered examples (an exemplar account), where two numbers are similar to the extent they are close in magnitude and share membership in stored number categories (even, odd, primes, squares, powers, multiples, final digit, contained digit), with membership in small, distinctive categories counting more than in broad ones. The examples are held in memory with strengths that fade with how far back in the sequence they were shown, so the most recently seen examples dominate the comparison — predicting more yeses to targets resembling the last example than to targets resembling only the first.

### stochastic_example_recall_sampling — pruned (experiment1 end of experiment)

**Outcome:** 5.6 nats behind familiarity_relaxed_size_principle (2.7× dse)

**Hypothesis:** People do not reason from all the examples they were shown, but from the examples they can recall when judging the target: the most recent example is always remembered, while each earlier example is recalled only by chance, less often the further back it was shown, and the learner infers the hidden concept (by Bayesian rule-and-range inference) from whichever examples were recalled. Because the answer averages over these all-or-none recall outcomes, targets near the last example are endorsed more than targets near the first, and sets shown earlier in a sequence partly behave like smaller, broader-generalizing sets.

### recency_weighted_even_odd_family_prior — pruned (experiment1 end of experiment)

**Outcome:** 2.8 nats behind familiarity_relaxed_size_principle (2.1× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not weigh every example equally — they read the examples in order and each example's evidence counts more the later it was shown (a fitted order gradient, with the total amount of evidence held fixed). The single change is grafting this recency weighting of the examples (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that targets near the last-shown example get more yeses than targets near the first-shown one, which an order-blind learner cannot produce.

### recency_decay_even_odd_family_prior — pruned (experiment1 end of experiment)

**Outcome:** 4.8 nats behind familiarity_relaxed_size_principle (2.1× dse)

**Hypothesis:** Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples one at a time and their memory of earlier examples fades, so each example's evidence counts less the more examples were shown after it. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that people say yes more to targets near the last-shown example than to targets near the first-shown one, which the order-blind incumbent cannot produce.

### recency_weighted_rule_family_prior — pruned (experiment1 end of experiment)

**Outcome:** 21.1 nats behind familiarity_relaxed_size_principle (4.0× dse)

**Hypothesis:** Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules (with a prior over kinds of rule) and all intervals of 1-100 with a tempered size principle, but they read the examples in the order shown and their memory of earlier examples fades, so each example's evidence counts less the further back it was shown and the most recent examples dominate which concept is inferred. The single change is grafting this recency-weighted evidence (from `recency_weighted_concept_learning`) onto the family-prior model, motivated by the critique that targets near the last-shown example are endorsed more than targets near the first-shown one, which no order-blind learner can produce.

### previous_set_intrusion_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 7.0 nats behind familiarity_relaxed_size_principle (3.2× dse)

**Hypothesis:** People infer the hidden concept by Bayesian rule-and-range inference (tempered size principle, a prior over kinds of rule with separate even and odd kinds), but the examples of the previously shown, different set linger in memory and intrude as extra, down-weighted evidence about the current set's concept (proactive interference between trials). This disagrees most sharply with the order-blind, trial-independent best model whenever a set follows an unrelated set: the pooled, partly conflicting examples push belief toward broad ranges and broad rules (more yeses to distant targets), or, when the previous set shares the current set's pattern (e.g. both even), toward that shared rule — so the same set and target get different answers depending on what was shown just before.

### exposure_relaxed_size_principle — rejected (experiment1 round 4 candidate 3 refine incumbent recency_weighted_even_odd_family_prior)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `recency_weighted_even_odd_family_prior`: people do the same recency-weighted Bayesian concept learning over the same 45 rules (separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but the strength of the size principle is not fixed — it changes as a participant keeps judging targets for the same set of examples, so the "suspicious coincidence" that favours tight concepts fades (or hardens) with each repeated encounter of the set. The single change is making the sampling strength depend on how many times that participant has already seen the set (a fitted exposure gradient), motivated by the critique that first encounters of a set get fewer yeses than later ones, which predicts that generalization broadens toward wider ranges and broad rules on repeated exposures.

### exposure_tempered_recency_family_prior — rejected (experiment1 round 4 candidate 4 refine incumbent recency_weighted_even_odd_family_prior)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** Refinement of `recency_weighted_even_odd_family_prior`: people do the same recency-weighted Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but how strongly they apply the size principle depends on whether they are meeting a set for the first time: on the first question about a set they treat the examples as strong evidence for a small, tight concept, while on later questions about an already-familiar set the examples' evidence weighs less (a fitted change in sampling strength), so they generalize more broadly. The single change is this exposure-dependent sampling strength, motivated by the critique that a set's first trial gets fewer yeses than its later trials, which the history-blind incumbent cannot produce.

### recency_family_prior_decisive_response — rejected (experiment1 round 4 candidate 4 refine incumbent recency_weighted_even_odd_family_prior repair 1)

**Outcome:** predicts like existing model 'recency_family_prior_sharpened_response' (p_yes RMSE 0.00073 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of recency_family_prior_sharpened_response, not a new hypothesis.

**Hypothesis:** Refinement of `recency_weighted_even_odd_family_prior`: people do the same recency-weighted Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not answer by probability matching to their belief that the target is in the concept — they answer more decisively, following the belief's log-odds scaled by a fitted decisiveness exponent, so targets they think are probably in (or probably out) get yes (or no) more consistently than the belief itself implies. The single change is grafting this sharpened belief-to-response mapping (from `response_sharpened_tempered_rule_prior`) onto the incumbent, predicting fewer yeses for weakly supported targets and more for strongly supported ones, with the ordering of targets unchanged.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 536.0 nats behind familiarity_relaxed_size_principle (9.0× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 393.7 nats behind familiarity_relaxed_size_principle (8.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### kind_specific_sampling_strength — rejected (experiment2 round 0 candidate 3 refine incumbent capacity_limited_evidence_integration)

**Outcome:** predicts like existing model 'magnitude_privileged_size_principle' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of magnitude_privileged_size_principle, not a new hypothesis.

**Hypothesis:** Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian learning over rules (separate prior plausibilities for even, odd and the other kinds of rule) and numerical ranges, with each added example contributing less evidence, but they apply the size principle with different strength to the two kinds of concept — examples are read as much more (or less) diagnostic of a tight numerical range than of a small mathematical rule. The single change is a separate, fitted sampling strength for range hypotheses, motivated by the critique that for sets mixing odd and even numbers (where ranges dominate) people separate targets inside versus outside the examples' range more sharply than the incumbent predicts.
