# Tried before

67 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### bigelow_piantadosi_independent — dropped (experiment1)

**Outcome:** MCMC fit failed on CV fold 3 (AssertionError: Apply node that caused the error: Composite{((i0 * i1) + (eq(i6, i7) * i2) + switch(i3, 0.0, (i4 * i5)))}(IncSubtensor{:, i}.0, Exp.0, Composite{(i0 + (i2 * i3) + i1)}.0, Isinf.0, Composite{...}.1, Composite{...}.1, ExpandDims{axis=1}.0, Join.0) Toposort index: 128 Inputs types: [TensorType(float64, shape=(None, 46)), TensorType(float64, shape=(None, 46)), TensorType(float64, shape=(None, 1)), TensorType(bool, shape=(None, 1)), TensorType(float64, shape=(None, 1)), TensorType(float64, shape=(None, 46)), TensorType(float64, shape=(None, 1)), TensorType(float64, shape=(None, 46))] HINT: Use a linker other than the C linker to print the inputs' shapes and strides. HINT: Re-running with most PyTensor optimizations disabled could provide a back-trace showing when this node was created. This can be done by setting the PyTensor flag 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'. HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### distinctive_feature_matching — pruned (experiment1 end of experiment)

**Outcome:** 441.3 nats behind powers_own_family_erlang_interval (11.7× dse)

**Hypothesis:** People do not entertain explicit hypotheses about the hidden program; instead they judge a target by feature matching: they notice which arithmetic and digit properties (parity, primality, being a square or a power, being a multiple of k, final digit, contained digits, decade) all the examples have in common, and say "yes" in proportion to how much of that common-feature profile the target also shows, with rarer (more distinctive) shared properties counting more than common ones. Numerical closeness matters only insofar as it produces shared features, so a target missing a distinctive common property is rejected even if it is adjacent to an example.

### shared_property_profile_match — pruned (experiment1 end of experiment)

**Outcome:** 424.5 nats behind powers_own_family_erlang_interval (11.6× dse)

**Hypothesis:** People do not average over candidate concepts or generalise by numerical closeness; instead they note which arithmetic and digit properties (parity, primality, squares, powers, multiples, final digit, contained digit) every example shares, and say "yes" in proportion to how much of that shared-property profile the target also has, with rarer (more distinctive) properties counting more. This disagrees most sharply with the best model on targets just next to an example that break the shared properties (e.g. 17 for the set 16, 8, 64), which it predicts are usually rejected because there is no interval or magnitude-based generalisation at all.

### shared_property_matching — pruned (experiment1 end of experiment)

**Outcome:** 383.7 nats behind powers_own_family_erlang_interval (12.6× dse)

**Hypothesis:** People do not weigh competing concepts by how well they explain the examples; instead they note every salient number property that all the examples share (parity, final digit, tens digit, being a multiple or power of something, prime, square, containing a digit) and say "yes" to a target in proportion to how many of those shared properties it also has. A target is rejected to the degree it breaks the examples' common properties, regardless of how close it lies in magnitude to any example.

### family_weighted_concept_prior — pruned (experiment1 end of experiment)

**Outcome:** 11.9 nats behind powers_own_family_erlang_interval (2.4× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same rules and intervals with the size principle, but they do not treat all rules as equally likely a priori. Instead, each family of concepts — parity (odd/even), special mathematical sets (primes, squares, cubes, powers), multiples of k, "ends in digit d", "contains digit d", and numeric intervals — has its own prior probability, shared equally by that family's members. So a broad but natural rule like "even" or "multiples of 3" can outweigh a tighter but implausible one like "contains the digit 2", which the size principle alone would favor.

### concept_learning_endorsement_ceiling — pruned (experiment1 end of experiment)

**Outcome:** 272.5 nats behind powers_own_family_erlang_interval (10.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over rules and intervals (size principle, rule-vs-interval prior lambda_0), but they do not say "yes" whenever they believe the concept contains the target — they endorse a target only with a fixed probability (an endorsement ceiling below 1), so "yes" is that ceiling times the posterior probability of membership. The one change is this conservative endorsement ceiling, which lowers yes rates uniformly for every believed member (near-example, same-parity, same-digit targets alike) instead of letting the posterior map straight onto yes.

### conservative_endorsement_concept_learning — rejected (experiment1 round 0 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'concept_learning_endorsement_ceiling' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of concept_learning_endorsement_ceiling, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people infer the concept exactly as in that model (Bayesian concept learning over 45 equally likely rules and all intervals, with the size principle), but they are conservative endorsers — even when the inferred concept contains the target, they commit to "yes" only on a fixed fraction of occasions and otherwise answer "no". The one change is this no-biased response stage, which scales every posterior membership probability down by a single commitment probability (so even targets shown as examples get "yes" only about three times in four, and partial-belief targets near examples or off a parity/last-digit rule are endorsed less).

### learned_example_reliability_ceiling — pruned (experiment1 end of experiment)

**Outcome:** 261.7 nats behind powers_own_family_erlang_interval (9.6× dse)

**Hypothesis:** Refinement of `concept_learning_endorsement_ceiling`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals (size principle, rule-vs-interval prior), and endorse a believed member only up to a fixed ceiling, but they do not assume a fixed 10% of the examples are random noise — how much they trust the examples to come from the hidden concept is itself a free parameter. The one change is this learned example-reliability: if people trust the examples more than the original model assumes, a single example that breaks a rule (an odd number among evens, a different final digit) rules that concept out more decisively, sharpening rejection of parity- and digit-violating targets.

### posterior_sharpened_concept_selection — pruned (experiment1 end of experiment)

**Outcome:** 12.9 nats behind powers_own_family_erlang_interval (2.7× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people weigh the same rules and intervals by prior plausibility and the size principle, but they do not probability-match over the whole posterior — when settling on what the program does, they favour the most convincing concepts more strongly than Bayes would (or less, if their selection is noisy), as if choosing among concepts with a sharpening exponent on each concept's posterior weight. The one change is this inferred decision temperature: a sharpening exponent above one lets a clearly winning rule like "ends in 3" crowd out the many loosely fitting intervals, so in-range targets that break the shared final digit are rejected more often.

### evidence_overweighted_size_principle — pruned (experiment1 end of experiment)

**Outcome:** 368.3 nats behind powers_own_family_erlang_interval (9.7× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (equally likely) and numeric intervals with the size principle, but they over-weight the evidence: each example counts as if it were seen more than once, so the likelihood of the examples is raised to a power above one (an "evidence-overweighting" distortion of the normative posterior). This makes tight rules that fit every example (e.g. "ends in 3" for 13, 23, 43) crowd out broader intervals more decisively than the normative model allows, so in-range targets that break such a rule are rejected more sharply.

### rule_only_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 398.2 nats behind powers_own_family_erlang_interval (12.3× dse)

**Hypothesis:** People generalize only through arithmetic and digit rules, never through numerical magnitude: they weigh the 45 candidate rules (parity, primes, squares, cubes, powers, multiples of k, "ends in d", "contains d") by how well each explains the examples under the size principle, and say "yes" with the posterior probability that the target obeys the rule, with no interval or closeness-based hypotheses at all. A target lying right between the examples but breaking every rule they share (e.g. 23 for 21, 41, 81) should therefore be rejected, not endorsed as nearby.

### family_weighted_learned_reliability — pruned (experiment1 end of experiment)

**Outcome:** 14.7 nats behind powers_own_family_erlang_interval (2.9× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass for each concept family, but they do not assume a fixed 10% of the examples are random noise — how much they trust the examples to come from the hidden concept is a free, inferred parameter (the component taken from `learned_example_reliability_ceiling`). The one change is this learned example reliability: if people trust the examples more, tight intervals that cover only one example and write the others off as noise lose their support, so for sets whose examples share a final digit people reject in-range targets that break that digit more firmly than the incumbent predicts.

### size_principle_strength_family_prior — pruned (experiment1 end of experiment)

**Outcome:** 11.8 nats behind powers_own_family_erlang_interval (2.4× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with a separate prior weight per concept family, but they treat the examples as more (or less) diagnostic than single independent random draws — the evidence the examples provide is raised to a learned "sampling strength" that sharpens or flattens the size principle. The one change is this learned strength of the size principle: if people over-weight how suspicious it is that every example ends in the same digit, a tight rule such as "ends in 3" decisively beats a broad interval spanning the examples, so in-range targets that break the shared final digit are rejected.

### family_prior_reliability_ceiling — pruned (experiment1 end of experiment)

**Outcome:** 17.8 nats behind powers_own_family_erlang_interval (3.3× dse)

**Hypothesis:** Refinement of `learned_example_reliability_ceiling`: people do Bayesian concept learning over the same 45 rules and all intervals with the size principle, trust the examples to a learned degree, and endorse a believed member only up to a ceiling — but they do not treat all rules as equally likely a priori. The one change is grafting the family-weighted concept prior: each family of concepts (parity, special mathematical sets, multiples of k, "ends in digit d", "contains digit d", intervals) has its own prior mass shared by its members, so a natural rule like "ends in 3" can outweigh the interval spanning the examples, which sharpens rejection of in-range targets that break a shared final digit.

### order_weighted_evidence_concept_learning — rejected (experiment1 round 2 candidate 0 lens 6)

**Outcome:** predicts like existing model 'family_weighted_concept_prior' (p_yes RMSE 0.00040 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of family_weighted_concept_prior, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over rules and numeric intervals with the size principle and a separate prior weight per concept family, but they do not weigh the examples equally: how strongly each example counts as evidence about the concept rises or falls steadily with its position in the sequence shown, so the last-seen (recency) or first-seen (primacy) examples dominate which concepts survive. The one change is this learned order gradient on evidence weight, with the average weight per example held at one, so a rule that fits the heavily weighted examples but not a lightly weighted one is penalised less than an equal-weight learner would.

### log_ratio_adaptive_similarity_gradient — pruned (experiment1 end of experiment)

**Outcome:** 473.7 nats behind powers_own_family_erlang_interval (15.2× dse)

**Hypothesis:** People judge a target by its similarity in magnitude to the examples, measured as the ratio distance (difference in log magnitude) to the nearest example, so a fixed absolute gap matters more among small numbers than large ones. Their willingness to say "yes" falls off smoothly with that ratio distance, and the tolerance of the gradient widens in proportion to how spread out (in log magnitude) the examples are, so tightly clustered examples yield a sharp generalization gradient and widely scattered examples a broad one.

### membership_confidence_threshold — pruned (experiment1 end of experiment)

**Outcome:** 333.1 nats behind powers_own_family_erlang_interval (10.9× dse)

**Hypothesis:** People infer the hidden concept as in plain Bayesian concept learning (equally likely rules versus intervals, with the size principle), but they do not answer "yes" with the probability that the concept contains the target; instead they apply a noisy confidence criterion, saying "yes" only when their belief that the target belongs clears a learned threshold. Targets supported by only part of the posterior — such as 37 for powers-of-2 examples, which only the ad-hoc "powers of 2, or 37" rule contains — therefore fall below the criterion and are rejected far more often than probability matching predicts, while targets believed members are endorsed nearly always.

### exception_rule_family_prior — pruned (experiment2 end of experiment)

**Outcome:** 20.7 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior weight per concept family and a learned strength of the size principle, but they do not treat ad-hoc "exception" rules (powers of 2 except 32, powers of 2 plus 37) as being as natural as clean mathematical sets. The one change is to give these exception rules their own concept family with its own inferred prior mass, so people can find "powers of 2, or 37" implausible and reject the 37 lure for sets of powers of 2 while still generalising to the other powers of 2.

### exception_rule_family_size_principle — rejected (experiment1 round 2 candidate 4 refine incumbent size_principle_strength_family_prior)

**Outcome:** predicts like existing model 'exception_rule_family_prior' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of exception_rule_family_prior, not a new hypothesis.

**Hypothesis:** Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior mass per concept family and a learned strength of the size principle, but they do not treat gerrymandered "exception" rules (powers of 2 without 32, powers of 2 plus 37) as just as natural as plain mathematical sets like powers of 2 or squares. The one change is that these ad-hoc exception rules form their own concept family with their own inferred (expectedly small) prior mass, so after seeing powers of 2, people rarely say the program would produce the lure 37.

### exception_rule_family_prior_2 — pruned (experiment2 end of experiment)

**Outcome:** 21.3 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass per concept family, but the two gerrymandered exception rules ("powers of 2 except 32" and "powers of 2 plus 37") are not natural mathematical sets — people give them their own, separately inferred (and expectedly tiny) prior mass rather than an equal share of the "special sets" family alongside primes, squares and powers. The one change is this separate family for exception rules, so for sets of powers of 2 people should reject the lure 37 nearly always while still endorsing the other powers of 2.

### first_example_anchored_then_filtered — pruned (experiment1 end of experiment)

**Outcome:** 613.9 nats behind powers_own_family_erlang_interval (11.3× dse)

**Hypothesis:** People form their guess about the program from the first example alone — weighing rules and numeric intervals by family-level prior plausibility and by how suspicious it would be to draw that first number from each concept (the size principle) — and then use each later example only as a consistency check: a concept that fails to contain a later example is (nearly) discarded, but a later example that fits does not make tight concepts more convincing than broad ones. So the first example sets how specific the guess is, and the rest merely prune it; e.g. after 16, 8, 64 a broad rule like "even" keeps the share it had after seeing 16 alone instead of being crowded out by "powers of 2".

### family_prior_confidence_threshold — pruned (experiment1 end of experiment)

**Outcome:** 31.6 nats behind powers_own_family_erlang_interval (4.5× dse)

**Hypothesis:** Refinement of `size_principle_strength_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior weight per concept family and a learned strength of the size principle, but they do not answer "yes" with the probability that the concept contains the target; instead they apply a noisy confidence criterion, saying "yes" only when their belief that the target belongs clears a learned threshold (the decision stage taken from `membership_confidence_threshold`). The one change is this thresholded decision rule: targets backed by only a minority of the posterior — such as 37 after powers of 2, which only the ad-hoc "powers of 2, or 37" rule contains — fall below the criterion and are rejected far more often than probability matching predicts, while confidently believed members are endorsed nearly always.

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Outcome:** 2.6 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 12

**Hypothesis:** People learn the hidden program's concept by Bayesian inference over natural rules and numeric ranges (with the size principle and a separate prior plausibility for each family of concepts), and their space of candidate rules also includes arithmetic progressions that do not start at zero — "every k-th number starting from r", such as 4, 11, 18, 25, … or the odd multiples of 3 — as their own family with its own prior plausibility. When the examples are evenly spaced by a common step, people therefore endorse targets that continue the same progression and reject nearby targets that fall off it.

### digit_parity_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 514.2 nats behind powers_own_family_erlang_interval (13.5× dse)

**Hypothesis:** People do not infer a hidden rule; they store the shown examples as exemplars and judge a target by its summed similarity to them, as in an exemplar (generalized context) model. Each number is represented on three psychological dimensions — its magnitude, its parity, and its final digit — and similarity to an example decays exponentially with magnitude distance and with each mismatch in parity or final digit; the probability of "yes" is the summed similarity relative to a fixed background response criterion.

### generative_production_likelihood — pruned (experiment1 end of experiment)

**Outcome:** 127.5 nats behind powers_own_family_erlang_interval (5.2× dse)

**Hypothesis:** People do not answer by asking whether the target belongs to the concept they have inferred; they answer by how readily they can imagine the program actually producing that target next — the probability of the target as the program's next output, which is high for targets in small, tight concepts (a lone powers-of-2 value, a narrow interval around clustered examples) and low for targets that belong only to broad concepts (all even numbers, a wide interval) even when membership is certain. Their yes rate rises smoothly with this predicted production probability, relative to what a program emitting numbers at random would give.

### erlang_interval_size_family_prior — pruned (experiment2 end of experiment)

**Outcome:** 6.6 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals, with the size principle and a separate prior mass per concept family (exception rules in their own family), but they do not regard all numeric intervals as equally plausible a priori. The one change is an Erlang-shaped prior over interval width with a learned characteristic size (as in Tenenbaum's number game): very narrow and very wide ranges are a priori unlikely, so after a single example or tightly clustered examples people generalise to a band of moderate width around them rather than collapsing onto the tightest interval, and after widely spread examples broad ranges are penalised relative to natural rules.

### erlang_interval_size_family_prior_2 — dropped (experiment2)

**Outcome:** MCMC fit failed on CV fold 2 (AssertionError: Apply node that caused the error: Composite{((i0 * i1) + (eq(i6, i7) * i2) + switch(i3, 0.0, (i4 * i5)))}(AdvancedIncSubtensor.0, Exp.0, Composite{(i0 + (i2 * i3) + i1)}.0, Isinf.0, Composite{...}.1, Composite{...}.1, ExpandDims{axis=1}.0, Join.0) Toposort index: 111 Inputs types: [TensorType(float64, shape=(None, 145)), TensorType(float64, shape=(None, 145)), TensorType(float64, shape=(None, 1)), TensorType(bool, shape=(None, 1)), TensorType(float64, shape=(None, 1)), TensorType(float64, shape=(None, 145)), TensorType(float64, shape=(None, 1)), TensorType(float64, shape=(None, 145))] HINT: Use a linker other than the C linker to print the inputs' shapes and strides. HINT: Re-running with most PyTensor optimizations disabled could provide a back-trace showing when this node was created. This can be done by setting the PyTensor flag 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'. HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.)

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules (with a prior mass per concept family, the two exception rules in their own family) and numeric intervals with the size principle, but they do not treat every interval of 1-100 as equally likely a priori — they expect a magnitude-based concept to span a characteristic, moderate range, so their prior over intervals follows an Erlang distribution over interval width with an inferred typical size (Tenenbaum's interval prior). The one change is this size-dependent interval prior: it penalises both the hairline intervals that hug the examples and the very broad ones, which changes how far people generalise by magnitude beyond the examples (e.g. to targets 10-20 away from a single example) relative to the incumbent's flat interval prior.

### erlang_interval_size_exception_prior — pruned (experiment2 end of experiment)

**Outcome:** 7.7 nats behind first_example_evidence_weight (2.6× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior`: people do the same Bayesian concept learning over rules and intervals, with a prior mass per concept family (exception rules as their own family) and a learned strength of the size principle, but they do not consider every numeric interval equally likely a priori. The one change is an Erlang prior on interval size with a learned typical size, as in Tenenbaum's number game: people expect a magnitude-based program to cover a range of moderate width, so very narrow intervals hugging the examples and very broad ones are both discounted, which reshapes how far "yes" answers extend beyond the examples in magnitude.

### mixed_weak_strong_sampling — pruned (experiment1 end of experiment)

**Outcome:** 2.6 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 12

**Hypothesis:** People do Bayesian concept learning over the same rules and numeric intervals (with a prior mass per concept family, the exception rules in their own family), but they do not assume every example was deliberately drawn from the concept: they believe each example is, with a learned probability, merely a number picked at random that happened to pass the program's test (weak sampling) rather than one the program generated (strong sampling, the size principle). The one change is this inferred mix of weak and strong sampling, which blunts the size principle in proportion to how large a concept is, so it disagrees most with the best model on multi-example sets consistent with both a tight and a broad natural rule — e.g. other even numbers after 16, 8, 64, or in-range targets that break a shared final digit — which it predicts are endorsed more often.

### digit_vocabulary_concepts — pruned (experiment1 end of experiment)

**Outcome:** 3.1 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 12

**Hypothesis:** People learn the hidden program's concept by Bayesian inference over natural number rules and numeric ranges (with the size principle and a prior plausibility per concept family), but they also entertain "digit vocabulary" concepts — the program writes numbers using only a particular set of digits — as their own family. When all the examples are spelled with a small set of digits, that coincidence makes such a concept convincing, so people endorse targets that recombine the examples' digits (e.g. 76 after 27, 37, 67) more than neighbouring targets that introduce a new digit.

### leaky_working_memory_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 3.4 nats behind powers_own_family_erlang_interval; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 12

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules and numeric intervals, the size principle, a prior mass per concept family with the exception rules in their own family), but they reason only from the examples they still hold in working memory: each shown example is independently retained with an inferred probability and forgotten otherwise, and a forgotten example simply does not enter the inference. Their "yes" rate is therefore the average, over which examples happened to be remembered, of the posterior probability that the concept contains the target — so multi-example sets generalise more broadly and less rule-tightly than a learner using every example would predict.

### power_base_family_prior — pruned (experiment2 end of experiment)

**Outcome:** 21.0 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals with the size principle, a separate prior mass per concept family and the exception rules in their own family, but they do not lump "powers of a base" (powers of 2, 3, ... 10) together with primes, squares and cubes as one "special sets" family. The one change is to give powers-of-a-base rules their own family with its own inferred prior mass: people readily think of a set as successive powers (or repeated multiplication) of a number, while primes, squares and cubes each have a separate, independently inferred plausibility, so generalisation after power-like sets (e.g. 8, 64) and after square- or prime-like sets (e.g. 81, 9 or 3, 31) can differ in strength instead of being tied to a shared per-rule prior.

### power_sequence_family_prior — rejected (experiment1 round 4 candidate 4 refine incumbent exception_rule_family_prior_2)

**Outcome:** predicts like existing model 'power_base_family_prior' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of power_base_family_prior, not a new hypothesis.

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals, with the size principle and a separate prior mass per concept family (the exception rules in their own family), but they do not regard "powers of a base" (2, 4, 8, 16…; 3, 9, 27…; 8, 64) as the same kind of concept as the classic named sets primes, squares and cubes. The one change is splitting the "special sets" family in two — power sequences and the named sets primes/squares/cubes — each with its own inferred prior mass, so people can readily generalise along a power sequence while finding a "primes" (or "squares") explanation less natural, which mainly lowers endorsement of other primes after examples such as 2, 3, 5, 7 or 11, 13, 17.

### powers_own_family_erlang_interval — pruned (experiment2 end of experiment)

**Outcome:** 7.2 nats behind first_example_evidence_weight (2.4× dse)

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval width, but they do not lump "powers of m" (2, 4, 8, 16, …; 3, 9, 27, 81; 8, 64; …) together with primes, squares and cubes. The one change is that repeated-multiplication rules form their own concept family with a separately inferred prior mass, so if people find powers an especially natural kind of program, examples like 8 and 64 are read as "powers of 8" (rejecting 2, 4, 16 and 32), and sets of powers of 2 crowd out "even" more strongly than the current model predicts.

### parity_bounded_range_concepts — pruned (experiment2 end of experiment)

**Outcome:** 21.8 nats behind first_example_evidence_weight (3.2× dse)

**Hypothesis:** Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle and a prior mass per concept family (the exception rules in their own family), but their hypothesis space also contains conjunctive "parity within a range" concepts — "even numbers between 20 and 60", "odd numbers from 9 to 69" — as a new family with its own inferred prior mass. The one change is this composed rule-plus-magnitude family: when examples are all even (or all odd) and clustered in magnitude, people generalise to same-parity numbers near the examples while rejecting both nearby numbers of the other parity and same-parity numbers far outside the examples' range, which neither a pure parity rule nor a pure interval predicts.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 330.3 nats behind powers_own_family_erlang_interval (8.5× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### weber_noisy_interval_membership — pruned (experiment2 end of experiment)

**Outcome:** 10.0 nats behind first_example_evidence_weight (3.0× dse)

**Hypothesis:** People do Bayesian concept learning over natural rules and numeric ranges as in the best model (size principle, a prior mass per concept family, an Erlang prior on range width), but they check whether a target falls inside a magnitude range using their approximate, noisy sense of number rather than exact arithmetic. The one distortion is this Weber-like perceptual noise, which grows in proportion to a number's size: symbolic rules (even, ends in 3, powers of 2) are checked exactly, but the edges of a range are blurred, so large targets just outside a range are often accepted and large targets just inside are sometimes rejected, while range boundaries among small numbers stay sharp.

### log_number_line_ranges — pruned (experiment2 end of experiment)

**Outcome:** 308.7 nats behind first_example_evidence_weight (8.9× dse)

**Hypothesis:** People do Bayesian concept learning over number rules (with a prior mass per concept family, the exception rules in their own family) and numeric ranges with the size principle, but they represent magnitude on a compressed, logarithmic mental number line: a range is a band on that line, so ranges are a priori equally likely wherever they sit on the log line and their plausibility depends on their ratio width (e.g. 5-10 is as natural as 50-100), not their width in units. As a result, magnitude-based generalisation scales with the examples' size — after small examples people generalise only to nearby small numbers, after large examples to a proportionally much wider band.

### win_stay_lose_sample_concept — rejected (experiment2 round 0 candidate 2 lens 6)

**Outcome:** predicts like existing model 'erlang_interval_size_family_prior' (p_yes RMSE 0.00049 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_family_prior, not a new hypothesis.

**Hypothesis:** People do not compute the full Bayesian posterior over concepts; they hold a single running guess about the program (rules and numeric ranges, weighed by family plausibility, an Erlang prior on range width and the size principle) and revise it example by example in the order shown, win-stay/lose-sample style: after each new example they keep their current guess with a probability that grows with how well it explains that example, and otherwise draw a fresh guess from what all the examples so far suggest. A learned stickiness controls how readily a guess formed on the early examples survives later ones, so the same examples shown in a different order (e.g. 16, 8 vs 8, 16) can yield different generalisation, with early-formed broad guesses persisting more than an ideal Bayesian learner allows.

### weber_scaled_interval_size_prior — pruned (experiment2 end of experiment)

**Outcome:** 7.4 nats behind first_example_evidence_weight (2.6× dse)

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval size, but they judge how big a range is relative to its magnitude (Weber's law) rather than in absolute units. The one change is that the Erlang prior is placed on the interval's width divided by a learned power of its location on the number line, so a range of ten numbers around 85 counts as "narrower" than one around 8; people therefore generalise to wider bands of nearby numbers after large examples than after small ones, with the strength of this magnitude scaling inferred from the data (zero recovers the incumbent).

### erlang_interval_learned_reliability — rejected (experiment2 round 0 candidate 4 refine incumbent erlang_interval_size_family_prior)

**Outcome:** predicts like existing model 'erlang_interval_size_family_prior' (p_yes RMSE 0.00053 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_family_prior, not a new hypothesis.

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over interval width, but how much they trust the examples to come from the hidden concept is not fixed at 90%; it is an inferred parameter. The one change is this learned example reliability (the component from `family_weighted_learned_reliability`, now added to the Erlang-interval model): the less people trust the examples, the more readily a concept that misses one example survives, so they generalise from the consistent majority of a set and hesitate even on targets that were themselves shown as examples.

### primes_own_family_erlang_interval — pruned (experiment2 end of experiment)

**Outcome:** 8.9 nats behind first_example_evidence_weight (2.5× dse)

**Hypothesis:** Refinement of `powers_own_family_erlang_interval`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules and powers of m each in their own family) and an Erlang prior over interval width, but they do not treat "prime numbers" as the same kind of program as squares and cubes. The one change is that primes form their own concept family with a separately inferred prior mass, so how readily people read examples such as 3, 7 or 53, 2 as "primes" (endorsing 29 or 59 and rejecting nearby composites) is no longer tied to how readily they read 81, 25 as "squares".

### newest_example_size_principle_wsls — pruned (experiment2 end of experiment)

**Outcome:** 313.1 nats behind first_example_evidence_weight (11.7× dse)

**Hypothesis:** People hold a single running guess about the program (a rule or a numeric range, weighed by family plausibility and an Erlang prior on range width) and revise it example by example in the order shown: they keep their current guess with a probability that grows with how well it explains the newest example, and otherwise draw a fresh guess — but when they redraw they judge the "suspicious coincidence" (the size principle) only for the newest example, merely requiring the guess to cover the earlier ones. So evidence for tight concepts does not compound across examples as in an ideal Bayesian learner, and which example drives specificity depends on the order shown (the last examples, plus whatever guess has survived from the first).

### threshold_concepts_erlang_interval — pruned (experiment2 end of experiment)

**Outcome:** 4.8 nats behind first_example_evidence_weight (2.1× dse)

**Hypothesis:** Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over the width of bounded ranges, but they treat one-sided magnitude thresholds ("numbers above 60", "numbers below 20") as a distinct kind of program rather than as just very wide intervals that happen to touch 1 or 100. The one change is that these threshold concepts form their own family with a separately inferred prior mass, shared equally among them and exempt from the width prior, so when all the examples are large (or all small) people extend "yes" all the way to the end of the number line, not just to a band of moderate width around the examples.

### average_digit_magnitude_similarity — pruned (experiment2 end of experiment)

**Outcome:** 793.1 nats behind first_example_evidence_weight (19.5× dse)

**Hypothesis:** People do not infer a rule; they judge a target by its average similarity to the examples, where two numbers are similar if they are close in magnitude and share their final digit, with similarity decaying exponentially along the number line and dropping by a fixed factor when the final digits differ. Because the similarity is averaged rather than summed over examples, "yes" depends on the proportion of the examples the target resembles, not on how many examples were shown, so adding dissimilar examples to a set dilutes generalisation to a target near one of them.

### any_of_sampled_concepts_endorsement — rejected (experiment2 round 1 candidate 1 lens 8)

**Outcome:** predicts like existing model 'threshold_concepts_erlang_interval' (p_yes RMSE 0.00068 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of threshold_concepts_erlang_interval, not a new hypothesis.

**Hypothesis:** People infer the hidden program exactly as in the best current model (Bayesian concept learning over number rules, bounded ranges and one-sided thresholds), but they answer by recall rather than by reading off their belief: a few candidate programs come to mind, sampled from their beliefs, and they say "yes" if any one of them would produce the target. The one change is this liberal any-of-several decision rule, with the effective number of programs considered inferred from the data. It raises "yes" most for targets that only a minority of plausible programs contain, and leaves targets that every or no plausible program contains unchanged; one program considered recovers probability matching.

### digit_sum_concepts_threshold_erlang — pruned (experiment2 end of experiment)

**Outcome:** 7.8 nats behind first_example_evidence_weight (2.5× dse)

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided thresholds, with the size principle and a prior mass per concept family. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50 for s = 5) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. When the examples happen to share a digit sum (especially a single example with a small, rare sum), people endorse other numbers with that digit sum more, and neighbours with a different digit sum less, than the incumbent predicts.

### threshold_concepts_primes_family — dropped (experiment3)

**Outcome:** MCMC fit failed on CV fold 1 (AssertionError: Apply node that caused the error: Composite{((i1 + (eq(i6, i7) * i2) + switch(i3, 0.0, (i4 * i5))) * i0)}([ 1. 2. ... . 97. 98.], Sum{axis=0}.0, Composite{(i0 + (i2 * i3) + i1)}.0, Isinf.0, Composite{...}.1, Composite{...}.1, ExpandDims{axis=0}.0, Composite{...}.0) Toposort index: 166 Inputs types: [TensorType(float64, shape=(98,)), TensorType(float64, shape=(98,)), TensorType(float64, shape=(1,)), TensorType(bool, shape=(1,)), TensorType(float64, shape=(1,)), TensorType(float64, shape=(98,)), TensorType(float64, shape=(1,)), TensorType(float64, shape=(98,))] HINT: Use a linker other than the C linker to print the inputs' shapes and strides. HINT: Re-running with most PyTensor optimizations disabled could provide a back-trace showing when this node was created. This can be done by setting the PyTensor flag 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'. HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.)

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided magnitude thresholds (their own family), with the size principle and a prior mass per concept family, but they do not treat "prime numbers" as just one more member of a shared "special sets" family with squares, cubes and powers. The one change is that primes form their own concept family with a separately inferred prior mass, so examples such as 3, 7 or 53, 2 can be read as "primes" strongly enough that people endorse other primes (13, 29, 5, 11) and reject nearby composites, independently of how readily they read 81, 25 as "squares" or 8, 64 as "powers".

### set_size_amplified_size_principle — pruned (experiment2 end of experiment)

**Outcome:** 5.7 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over rules, bounded numeric ranges (with an Erlang prior on width) and one-sided thresholds, with the size principle and a prior mass per concept family, but the "suspicious coincidence" does not accumulate additively over examples. The one change is that each example's evidential weight scales with a learned power of how many examples were shown, so the more examples a set has, the more decisively each one favours tight concepts over broad ones (zero recovers the incumbent): generalisation narrows faster from one to four examples than an ideal learner predicts.

### recognition_shortcut_for_examples — pruned (experiment2 end of experiment)

**Outcome:** 35.8 nats behind first_example_evidence_weight (3.1× dse)

**Hypothesis:** People infer the hidden program as in the best current model (Bayesian concept learning over number rules, bounded ranges and one-sided thresholds) and answer most targets by probability matching on that belief, but when the target is one of the numbers they were just shown they do not consult their inferred concept at all: they answer from recognition, saying "yes" at a single fixed rate that does not depend on the set. So the yes rate for a repeated example is the same whether one example or four were shown and whatever concept they suggest, unlike the best model, where it falls as tight concepts that exclude random-looking examples gain belief.

### shared_category_exemplar_averaging — pruned (experiment2 end of experiment)

**Outcome:** 774.5 nats behind first_example_evidence_weight (18.4× dse)

**Hypothesis:** People do not infer one program that explains all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored examples, where two numbers are similar to the degree that they fall together in the familiar number categories people know (parity, multiples, final or contained digits, primes, squares, powers, magnitude ranges and thresholds), with small, distinctive shared categories counting more than broad ones (Shepard's additive-clustering view of number similarity). Because each example generalises on its own and the similarities are averaged rather than multiplied, the "suspicious coincidence" of several examples sharing a tight category never accumulates: a target similar to only one example of a set gets partial "yes" support, and adding examples does not sharpen generalisation the way it does for a Bayesian concept learner.

### pedagogical_teacher_sampling — pruned (experiment2 end of experiment)

**Outcome:** 107.0 nats behind first_example_evidence_weight (6.1× dse)

**Hypothesis:** People assume the program's examples were chosen to be helpful, as a teacher would choose them, rather than drawn at random from its concept (rational pedagogy): a teacher holding a concept picks examples in proportion to how strongly each one, on its own, would point a naive learner (one with no preference among kinds of concept) to that concept. People judge each candidate rule, range or threshold by how likely such a teacher would be to have shown exactly these examples, so examples that would point equally well to many other concepts count as weak evidence and a concept is favoured when the examples are the ones most diagnostic of it; they then say "yes" with the probability that the inferred concept contains the target.

### blunted_range_size_principle — rejected (experiment2 round 2 candidate 2 lens 1)

**Outcome:** too slow to fit: the fit of 'blunted_range_size_principle' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, but the "suspicious coincidence" (size principle) is weaker for magnitude concepts than for rules. The one change is that, while examples count fully against broad rules such as "even" versus "powers of 2", their evidence against a wide range is blunted by a learned exponent on the range's size (one recovers the incumbent, zero means a range is judged only by whether it covers the examples), so ranges do not shrink onto the examples as fast as an ideal learner's would. The sharpest disagreement with the best model is for multi-example sets that are clustered in magnitude (e.g. 11, 15 or 42, 62): this model keeps saying "yes" to targets well beyond the examples' extremes, whereas the incumbent narrows onto a tight range around them, while rule-based generalisation (powers of 2, a shared final digit) is unchanged.

### range_size_principle_strength — pruned (experiment2 end of experiment)

**Outcome:** 5.2 nats behind first_example_evidence_weight (2.2× dse)

**Hypothesis:** Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded numeric ranges (with an Erlang prior on width) and one-sided magnitude thresholds (their own family), with a prior mass per concept family, but they do not apply the "suspicious coincidence" of the size principle to magnitude concepts as strongly as to arithmetic and digit rules. The one change is that, for ranges and thresholds only, how much an example's probability shrinks with the concept's width is governed by a learned exponent (one recovers the incumbent, zero means ranges are treated as mere filters the examples pass), so how tightly people's magnitude-based generalisation hugs the examples, relative to rule-based generalisation, is inferred from the data rather than fixed by the ideal learner.

### centre_weighted_ranges_with_thresholds_2 — rejected (experiment2 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'centre_weighted_ranges_with_thresholds' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of centre_weighted_ranges_with_thresholds, not a new hypothesis.

**Hypothesis:** Refinement of `centre_weighted_range_sampling` (rank 1): people do the same Bayesian concept learning over number rules and bounded numeric ranges, with the size principle, a prior mass per concept family (exception rules in their own family), an Erlang prior over range width, and the expectation that a bounded range's outputs cluster toward its middle. The one change, taken from `threshold_concepts_erlang_interval`, is that one-sided magnitude thresholds ("numbers above 60", "numbers below 20") are a distinct kind of program with their own inferred prior mass, exempt from the width prior and from centre-weighting (a threshold has no middle, so it emits its numbers uniformly). So the "edge-is-suspicious" pressure that pushes centred bounded ranges past the examples' extremes acts only on genuine ranges, while all-large or all-small sets can still be read as open-ended thresholds that run to the end of the number line.

### primes_family_set_size_amplified — pruned (experiment2 end of experiment)

**Outcome:** 7.6 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `set_size_amplified_size_principle`, adding one component from `threshold_concepts_primes_family`. People do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family and a size principle whose per-example weight grows with how many examples were shown. The one change is that "prime numbers" leaves the shared special-sets family (squares, cubes, powers) and becomes its own family with a separately inferred prior mass, so prime-like sets such as 3, 7 or 53, 2 are read as "primes" (endorsing other primes, rejecting nearby composites) independently of how plausible squares or powers are, and this reading tightens with more prime examples faster than an ideal learner's would.

### gaussian_prototype_magnitude_concepts — pruned (experiment2 end of experiment)

**Outcome:** 97.3 nats behind first_example_evidence_weight (6.6× dse)

**Hypothesis:** People do Bayesian concept learning over number rules and one-sided thresholds as in the best model, but they do not represent a magnitude-based program as a sharp-edged range; they represent it as a graded prototype — a typical value on the number line with a spread — that produces numbers mostly near its centre and fewer the further away they lie. A target then belongs to such a concept to the degree it is close to the prototype's centre, so magnitude generalisation falls off smoothly with distance from the examples instead of dropping at a boundary.

### limited_family_consideration_set — pruned (experiment2 end of experiment)

**Outcome:** 57.8 nats behind first_example_evidence_weight (5.4× dse)

**Hypothesis:** People do not search the whole space of possible programs on every trial: each kind of program (parity, special sets such as primes, squares and powers, multiples, final digit, contained digit, exception rules, bounded ranges, one-sided thresholds) comes to mind only with its own probability, and people do Bayesian concept learning with the size principle only over the kinds that came to mind, giving each considered kind an equal share of their prior. When the kind of program that best fits the examples is not thought of, they fall back on whatever kind they did consider, so "yes" is the average, over which kinds came to mind, of the probability that the inferred program produces the target. How available each kind is to recall replaces a fixed prior plausibility per kind.

### without_replacement_digit_sum_concepts — pruned (experiment2 end of experiment)

**Outcome:** 6.7 nats behind first_example_evidence_weight (2.5× dse)

**Hypothesis:** Refinement of `without_replacement_size_principle`, adding one component from `digit_sum_concepts_threshold_erlang`: people do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family and examples believed to be drawn from the concept without replacement, but their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. The one change is this digit-sum family: because these concepts are tiny, examples that happen to share a digit sum are an especially suspicious coincidence when drawn without replacement, so people endorse other numbers with that digit sum, and reject neighbours with a different sum, more than the incumbent predicts.

### set_size_amplified_digit_sum_concepts — pruned (experiment2 end of experiment)

**Outcome:** 8.7 nats behind first_example_evidence_weight (2.6× dse)

**Hypothesis:** Refinement of `set_size_amplified_size_principle`, adding one component from `digit_sum_concepts_threshold_erlang`. People do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family and a size principle whose per-example weight grows with how many examples were shown. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass shared equally among the 18 sums, so when the examples share a digit sum people endorse other numbers with that sum (and reject neighbours with a different sum) more than the parent predicts, and this reading sharpens with more examples faster than an ideal learner's would.

### simplicity_description_length_prior — pruned (experiment2 end of experiment)

**Outcome:** 910.8 nats behind first_example_evidence_weight (13.5× dse)

**Hypothesis:** People infer the hidden program as a Bayesian concept learner would — weighing number rules, bounded ranges and one-sided thresholds by how well they explain the examples, with the size principle and examples drawn without repetition — but their sense of how plausible a program is a priori comes from a single simplicity bias rather than learned preferences for particular kinds of concept: every program is penalised by the same amount for each symbol it takes to state it ("odd" is one, "multiples of 7" two, "between 23 and 41" three, "powers of 2 except 32" four). So how readily people read examples as a rule versus a range versus a threshold is fixed by description length alone, with one inferred cost per symbol.

### conjunctive_category_similarity — pruned (experiment2 end of experiment)

**Outcome:** 734.8 nats behind first_example_evidence_weight (17.9× dse)

**Hypothesis:** People do not infer a program; they judge a target by its similarity to the examples, where two numbers are similar to the degree that they fall together in familiar number categories (parity, multiples, final or contained digit, primes, squares, powers, magnitude ranges and thresholds), small distinctive shared categories counting more than broad ones. The similarities are combined conjunctively, as a geometric mean over the examples, so a target must resemble every example: being very dissimilar to any single example drags "yes" down sharply, unlike an average in which closeness to one example is enough. The willingness to say "yes" rises smoothly (logistically) with this geometric-mean similarity on a log scale.

### round_number_ranges_digit_sums — pruned (experiment2 end of experiment)

**Outcome:** 6.5 nats behind first_example_evidence_weight (2.4× dse)

**Hypothesis:** Refinement of `round_number_range_boundaries`, adding one component from `digit_sum_concepts_threshold_erlang`: people do the same Bayesian concept learning over number rules, bounded ranges (Erlang width prior, with a learned round-number bias on ranges whose boundaries fall on multiples of ten) and one-sided thresholds, with a prior mass per concept family and examples drawn without replacement. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. Because these concepts are tiny, examples that share a digit sum are a suspicious coincidence, so people endorse other numbers with that sum and reject neighbours with a different sum more than the incumbent predicts, while the round-number bias still shapes where magnitude-based "yes" answers stop.

### without_replacement_primes_digit_sums — pruned (experiment2 end of experiment)

**Outcome:** 8.2 nats behind first_example_evidence_weight (2.3× dse)

**Hypothesis:** Refinement of `without_replacement_primes_family`, adding one component from `digit_sum_concepts_threshold_erlang`: people do the same Bayesian concept learning over number rules (with "prime numbers" as its own family), bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, and they believe the program never repeats itself, so examples are drawn from the concept without replacement. The one change is that their space of candidate programs also includes digit-sum concepts — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a new family with its own inferred prior mass, shared equally among the 18 possible sums. When the examples happen to share a digit sum, people endorse other numbers with that sum and reject neighbours with a different sum more than the parent predicts, and this works independently of how readily they read prime-like sets as "primes".

### target_as_evidence_positive_test — rejected (experiment3 round 1 candidate 0 lens 0)

**Outcome:** too slow to fit: the fit of 'target_as_evidence_positive_test' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the hidden program as a Bayesian concept learner over number rules (including digit-sum rules) and numeric ranges, but when they judge a target they test it positively: they imagine the target as one more output of the program and let it count, with a partial weight inferred from the data, as evidence for the programs that would produce it. So at the moment of judgement, programs that contain the target gain belief — the more so the smaller they are — and targets that fit a tight concept are endorsed more than the examples alone justify, while targets that fit only broad concepts gain little.

### large_divisor_multiples_concepts — dropped (experiment3 round 1)

**Outcome:** fold 0 too slow to fit: the fit of 'large_divisor_multiples_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the hidden program as a Bayesian concept learner over number rules (including digit-sum rules) and numeric ranges, as in the best current model, but their space of programs also includes "multiples of n" for unusual, larger divisors n (13 up to 50, e.g. 13, 26, 39, 52 or 28, 56, 84), a family of programs judged a priori less plausible than the familiar multiples of 3 to 12 by an amount inferred from the data. Because such programs contain only a handful of numbers, examples that are all multiples of an unusual number (e.g. 13, 91 or 28, 84) are a suspicious coincidence, so people endorse other multiples of that number and reject nearby non-multiples more than the best model predicts.

### graded_digit_sums_round_edge_ranges — rejected (experiment3 round 1 candidate 4 refine incumbent graded_prior_digit_sum_concepts repair 1)

**Outcome:** too slow to fit: the fit of 'graded_digit_sums_round_edge_ranges' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `graded_prior_digit_sum_concepts`, adding one component from `round_number_range_boundaries`: people do the same Bayesian concept learning over number rules, digit-sum rules (with a prior over sums that changes smoothly with the size of the sum) and numeric ranges (Erlang width prior, outputs clustered toward the range's middle), with the size principle and a prior mass per concept family. The one change is that multiples of ten act as reference points when people judge how plausible a range is a priori: a range whose lower or upper edge falls exactly on a multiple of ten is judged more or less likely than an otherwise identical range, with the direction and strength of the bias inferred from the data. This shifts where magnitude-based "yes" answers stop near a round number (e.g. whether 40 is endorsed after examples in the low thirties), while digit-sum and other rule-based generalisation are unchanged.
