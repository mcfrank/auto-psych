# Tried before

75 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### tenenbaum_rules_similarity — dropped (experiment1)

**Outcome:** MCMC fit failed on CV fold 3 (AssertionError: Apply node that caused the error: Mul(Assert{msg='Could not broadcast dimensi...'}.0, Shape_i{2}.0) Toposort index: 54 Inputs types: [TensorType(int64, shape=()), TensorType(int64, shape=())] HINT: Use a linker other than the C linker to print the inputs' shapes and strides. HINT: Re-running with most PyTensor optimizations disabled could provide a back-trace showing when this node was created. This can be done by setting the PyTensor flag 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'. HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### exemplar_digit_magnitude_gcm — pruned (experiment1 end of experiment)

**Outcome:** 269.8 nats behind sticky_guess_power_law_ranges (10.6× dse)

**Hypothesis:** People do not infer a hidden rule; they judge a target by its summed similarity to the remembered examples (an exemplar, generalized-context-model process). Similarity to each example decays exponentially with a city-block psychological distance built from how far apart the two numbers are in magnitude and whether they differ in their tens digit and in their ones digit, and the probability of "yes" grows with the target's total similarity to the examples relative to a response criterion.

### exemplar_similarity_number_line — pruned (experiment1 end of experiment)

**Outcome:** 299.6 nats behind sticky_guess_power_law_ranges (10.8× dse)

**Hypothesis:** People do not induce any rule or interval: they judge a target by its summed perceptual similarity on the number line to the examples shown, with similarity decaying exponentially with numeric distance (an exemplar / GCM account), and say "yes" in proportion to that summed similarity relative to a fixed response criterion. This disagrees most sharply with the rule-and-interval learner in the gaps between widely spaced examples (e.g. 50 for the set 92,14,20,5, where an interval spanning the examples predicts "yes" but similarity predicts "no") and for targets just outside the examples' range (where the interval predicts "no" but similarity predicts "yes").

### spread_scaled_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 265.3 nats behind sticky_guess_power_law_ranges (9.0× dse)

**Hypothesis:** People do not infer any mathematical rule; they generalize by numerical similarity alone, judging a target more likely to be produced the closer it lies to the individual examples on the number line. The breadth of that similarity gradient is set by how spread out the examples are: tightly clustered examples yield a narrow gradient, widely scattered examples a broad one, so the same distance counts for less when the examples cover a wide range.

### inferred_sampling_reliability_concepts — pruned (experiment1 end of experiment)

**Outcome:** 7.4 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100, with the size principle, but how strongly they believe the examples were drawn from the concept (rather than at random from 1-100) is a free parameter inferred from the data instead of being fixed at 90%. This one change sets how sharply the size principle favours small, specific concepts over broad ones, and so how tightly people generalise from a few examples.

### bayesian_concepts_logodds_readout — pruned (experiment1 end of experiment)

**Outcome:** 7.1 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, 10% noisy examples), but they do not report the posterior probability of membership directly; instead they answer "yes" through a graded, possibly biased decision readout in which the log-odds of membership is scaled by a free confidence factor and shifted by a free yes/no criterion. This lets near-certain posteriors (e.g. for the examples themselves, or for numbers far outside the concept) map onto less extreme, empirically observed yes rates, and lets judges be more or less liberal than the posterior.

### partial_strong_sampling_rules_intervals — pruned (experiment1 end of experiment)

**Outcome:** 94.8 nats behind sticky_guess_power_law_ranges (4.3× dse)

**Hypothesis:** Refinement of the dropped `tenenbaum_rules_similarity` model (Bayesian concept learning over 33 mathematical rules and all intervals of 1-100, rules sharing a prior fraction equally and intervals weighted by an Erlang density on their size). The one change: learners do not assume every example was deliberately sampled from the concept (pure strong sampling, the size principle); they treat each example as drawn from the concept with probability theta (likelihood 1/|concept|) and otherwise as an arbitrary number that merely happens to fall in it (weak sampling, likelihood independent of concept size), with theta fitted — so the size principle is only partly applied, and generalisation is broader, e.g. beyond the examples' range and to larger rules, than strong sampling predicts. (The original's crash on cross-validation folds, a data-shape dependence on the fold, is also fixed.)

### leaky_memory_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 8.3 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 20 of 21

**Hypothesis:** People do Bayesian concept learning over mathematical rules and intervals (with the size principle), but they read the examples one at a time and their memory of earlier examples fades: each example's evidence is discounted by a free forgetting factor for every example shown after it, so the most recent example counts fully and the first one of a long set counts least. As a result, concepts that fit only the later examples (and rules or intervals violated by an early example) remain plausible, and generalisation from long sets is broader and recency-biased.

### serial_position_weighted_concepts — pruned (experiment1 end of experiment)

**Outcome:** 9.7 nats behind sticky_guess_power_law_ranges (2.1× dse)

**Hypothesis:** People do Bayesian concept learning over the same 45 mathematical rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not weigh the examples equally: because the examples are seen in sequence, each example's evidence is weighted by its serial position, with a single fitted order-bias parameter that shifts weight toward the most recent examples (recency) or toward the first ones (primacy) while keeping the total amount of evidence fixed. So the same examples shown in different orders lead to different generalisations, favouring concepts that fit the weighted-up examples.

### map_single_concept_selection — pruned (experiment1 end of experiment)

**Outcome:** 236.5 nats behind sticky_guess_power_law_ranges (10.2× dse)

**Hypothesis:** People do not average over all the concepts consistent with the examples: they commit to the single most probable concept (a rule such as "powers of 2" or "multiples of 10", or a numeric interval, chosen by a fixed prior and the size principle) and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. Generalisation is therefore all-or-none relative to the winning concept, with no graded uncertainty from runner-up concepts.

### erlang_interval_prior_concepts — pruned (experiment1 end of experiment)

**Outcome:** 3.6 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule), but they do not regard every interval as equally plausible a priori. Instead, their prior over intervals follows an Erlang density on the interval's size with a fitted characteristic size, so very narrow ranges hugging the examples are a priori implausible and ranges of a typical intermediate width are favoured; this one change controls how far beyond the examples' span people generalise when they think in terms of a numerical range.

### erlang_interval_prior_concepts_2 — rejected (experiment1 round 1 candidate 4 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'erlang_interval_prior_concepts' (p_yes RMSE 0.00067 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight inferred for each rule), but they do not consider every interval equally likely a priori; instead their prior over intervals follows an Erlang density on the interval's size with a fitted scale, so very narrow intervals hugging the examples are a priori implausible and intervals of a characteristic, moderate breadth are preferred. This one change to the interval prior sets how far beyond and between the examples people extend a magnitude-based concept.

### erlang_width_interval_concepts — pruned (experiment1 end of experiment)

**Outcome:** 6.5 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 21

**Hypothesis:** Refinement of bayesian_concepts_logodds_readout: people do the same Bayesian concept learning over the 45 mathematical rules and all intervals of 1-100 (size principle, 10% noisy examples) and answer through the same scaled, shifted log-odds readout, but they do not consider all interval sizes equally plausible a priori. The one change: an interval's prior depends on its width through an Erlang density with a fitted typical width, so people expect the program's range of numbers to have a characteristic breadth, which sets how far they generalise beyond the span of the examples independently of the size principle.

### power_law_interval_endpoints — pruned (experiment1 end of experiment)

**Outcome:** 5.0 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight per rule), but when they imagine the program picking a numerical range they do not treat every endpoint as equally likely: the two endpoints are drawn from a power-law distribution over 1-100 whose exponent is fitted, so a range's prior plausibility depends on where on the number line it lies (among small versus large numbers), not just on whether it exists. This magnitude-dependent prior on interval endpoints is the one change; an exponent of zero recovers the incumbent's uniform endpoints. It changes how strongly range concepts compete with rules for small-number versus large-number examples.

### first_guess_win_stay_concepts — pruned (experiment2 end of experiment)

**Outcome:** 4.2 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 20

**Hypothesis:** People do not weigh all examples together at once; they hold a single working guess about the concept (a rule or a numerical range) that they form from the first example by Bayesian concept learning, and as each further example arrives they keep that guess, with a fitted stickiness, as long as the new example fits it, reconsidering the concept from all examples seen so far only when the new example contradicts the guess (or when they do not stick). Because the guess is anchored on what fitted the early examples, a broad concept chosen after the first example survives later examples that would have favoured a narrower one, so the same examples in a different order produce different generalisations (a primacy effect).

### tversky_feature_contrast_similarity — pruned (experiment1 end of experiment)

**Outcome:** 310.6 nats behind sticky_guess_power_law_ranges (12.0× dse)

**Hypothesis:** People do not infer a hidden rule or range; they judge a target by Tversky-style feature similarity to the examples. Each number is represented by its salient properties (parity, primality, being a square, cube or power, multiples, final digit, digits it contains, and which band of ten it falls in), each property weighted by how distinctive it is (rarer properties count more), and the target's similarity to the examples is the weighted features it shares with them relative to the features the examples have that it lacks and the features it has that the examples lack (Tversky's ratio contrast model); the probability of "yes" rises with that similarity.

### lapse_biased_guess_concepts — pruned (experiment1 end of experiment)

**Outcome:** 9.7 nats behind sticky_guess_power_law_ranges (2.3× dse)

**Hypothesis:** People do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 as the simplest model (size principle, 10% noisy examples, a prior weight per rule), and on most trials they answer by reporting the posterior probability that the concept contains the target; but on a fitted fraction of trials they lapse — they do not consult the inferred concept at all and instead guess, answering "yes" with a fitted guessing bias. This all-or-none attentional lapse with a biased guess is the one change: it puts a floor and a ceiling on yes rates that are the same for every stimulus, rather than graded softening of confident posteriors.

### round_endpoint_interval_concepts — pruned (experiment2 end of experiment)

**Outcome:** 39.2 nats behind coincidence_triggered_guess_rethink (4.9× dse)

**Hypothesis:** Refinement of erlang_interval_prior_concepts: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule, an Erlang prior on interval size), but when they imagine the program picking a numerical range they favour ranges with round-number boundaries (starting at 1 or a multiple of ten, ending at a multiple of ten, e.g. "20 to 40" or "1 to 50") over arbitrary ones like "17 to 43". The one change is a fitted prior boost for each round endpoint, so generalisation from clustered examples extends to, and then drops off sharply at, the nearest decade boundaries rather than fading smoothly with width alone.

### decade_aligned_interval_concepts — rejected (experiment1 round 2 candidate 4 refine incumbent erlang_interval_prior_concepts)

**Outcome:** predicts like existing model 'erlang_interval_prior_concepts' (p_yes RMSE 0.00154 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of erlang_interval_prior_concepts: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule, an Erlang prior on interval size), but when they think of the program as producing a numerical range they favour ranges bounded by round numbers — decade boundaries such as "1 to 10", "20 to 29", "between 10 and 50", "51 to 100" — over ranges with arbitrary endpoints. The one change is a fitted prior boost for decade-aligned intervals, so people generalise from examples out to the nearest decade boundaries (e.g. from 24 and 27 to the whole of the twenties) rather than smoothly beyond the examples' span.

### power_law_endpoints_erlang_width — pruned (experiment2 end of experiment)

**Outcome:** 34.9 nats behind coincidence_triggered_guess_rethink (4.6× dse)

**Hypothesis:** Refinement of power_law_interval_endpoints: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight per rule) with interval endpoints whose prior plausibility follows a fitted power law over the number line, but the one change is that they also expect a numerical range to have a characteristic width: the interval prior is additionally weighted by an Erlang density on the interval's size with a fitted typical size (the component taken from erlang_interval_prior_concepts). So where a range lies on the number line and how broad it is both shape the prior, which jointly sets how far people generalise beyond the examples' span for small- versus large-number sets.

### similarity_route_erlang_concepts — pruned (experiment1 end of experiment)

**Outcome:** 5.3 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 21

**Hypothesis:** Refinement of erlang_interval_prior_concepts: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule, an Erlang prior on interval size), but on a fitted fraction of judgements they do not consult the inferred concept at all and instead answer by number-line similarity, saying "yes" more the closer the target is to the nearest example (a component taken from the exemplar-similarity models). The one change is this similarity route: it makes people endorse far-away numbers that fit a rule only through a surface property (such as sharing an example's last digit) less than pure concept learning predicts, and near neighbours of the examples more.

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Outcome:** 9.0 nats behind sticky_guess_power_law_ranges (2.1× dse)

**Hypothesis:** People do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 as the Erlang-interval model (size principle, 10% noisy examples, a prior weight per rule, an Erlang prior on interval size), but their space of rules also includes "same-step" arithmetic progressions with any starting offset — every class of numbers that are a fixed step apart without being multiples of it (e.g. 2, 7, 12, ... 97 for 82, 67, 72; or 1, 5, 9, ... for 25, 17) — for steps 3 to 12. This family of offset progressions takes a single fitted share of the rule prior, split equally among its members, so examples that share a common spacing are generalised to other numbers on the same step even when no familiar rule (multiples, last digit) fits them.

### category_exemplar_similarity_retrieval — pruned (experiment1 end of experiment)

**Outcome:** 9.2 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 21 of 21

**Hypothesis:** People do not work out which concept could have generated the examples. Instead they retrieve familiar number categories stored in memory as sets of remembered members: the familiar rules (odd numbers, powers of 2, multiples of 10, numbers ending in 7, ...) and familiar ranges bounded by multiples of five (the 20s, 1 to 50, 60 to 75, ...). Each category is activated by how similar the examples are, on average, to its stored members, with similarity falling off exponentially with numerical distance at a fitted rate (a generalized-context-model comparison against category exemplars). The target is endorsed in proportion to how similar it is to the nearest member of each activated category, so near-misses partly activate a category and partly earn endorsement (e.g. 31 after 10, 20, 40).

### pedagogical_teacher_sampling — pruned (experiment1 end of experiment)

**Outcome:** 7.0 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 21

**Hypothesis:** People treat the examples as having been chosen on purpose by a helpful teacher (pedagogical sampling): when they ask how likely an example is under a candidate concept (a rule such as "powers of 2" or a numerical range), they expect a teacher who held that concept to pick examples from it that are diagnostic of it — numbers that would lead a naive learner to that concept rather than to others — and not just any member at random. Bayesian concept learning over rules and ranges with this teacher-based likelihood, whose strength of pedagogical expectation is fitted (zero recovers ordinary random sampling from the concept), sets how people generalise to a target.

### sticky_guess_rule_specific_priors — pruned (experiment1 end of experiment)

**Outcome:** 4.5 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 21

**Hypothesis:** Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning, keep it with a fitted stickiness while new examples fit it, and reconsider from all examples seen so far only when an example contradicts it (or they do not stick), producing a primacy effect. The one change is that rules are not equally plausible a priori: each rule (e.g. "powers of 2" versus "ends in 7") has its own fitted prior weight (the component taken from bigelow_piantadosi_independent / erlang_interval_prior_concepts), so which guess people first latch onto — and therefore which guess survives later examples — depends on how natural each rule is to them.

### sticky_guess_power_law_ranges — pruned (experiment2 end of experiment)

**Outcome:** 4.4 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 20

**Hypothesis:** Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept, with a fitted stickiness, as later examples arrive as long as each new example fits it, reconsidering from all examples so far only when it is contradicted (or they do not stick). The one change, the component taken from power_law_interval_endpoints: when they consider a numerical range they do not treat all locations on the number line alike — the range's endpoints have a prior plausibility that follows a fitted power law over 1-100 (on top of the incumbent's Erlang preference for a typical width), so whether a range guess is formed and kept depends on whether the examples are small or large numbers.

### magnitude_compressed_sampling_concepts — pruned (experiment2 end of experiment)

**Outcome:** 34.4 nats behind coincidence_triggered_guess_rethink (4.6× dse)

**Hypothesis:** Refinement of power_law_endpoints_erlang_width (same 45 rules and all intervals of 1-100, size principle, 10% noisy examples, a prior weight per rule, power-law endpoint prior and Erlang width prior). The one change is in how people think the examples were sampled: they imagine the program drawing numbers uniformly on a compressed, logarithmic-like mental number line rather than uniformly in raw value, with the degree of compression fitted, so within a range or rule smaller numbers are expected to be drawn more often than larger ones. Because a span among large numbers is then psychologically narrower, the size principle penalises extending a range less for large-number examples than for small-number ones, so generalisation beyond the examples widens in proportion to their magnitude (Weber-like scaling).

### range_bounded_rule_concepts — pruned (experiment1 end of experiment)

**Outcome:** 7.9 nats behind sticky_guess_power_law_ranges (2.4× dse)

**Hypothesis:** People's hypothesis space contains not only whole-number-line rules ("even numbers", "ends in 7") and numerical ranges, but also rules confined to a range ("even numbers in the 20s", "multiples of 5 below 40"); because such range-bounded rules are smaller, the size principle favours them, so people generalise a rule mainly to targets near the examples. The disagreement with the current best model lives in targets far outside the examples' range that fit a rule only by a property such as their last digit or a common multiple: the best model endorses them strongly, this one predicts that endorsement falls off with distance from the examples.

### decade_restricted_rule_concepts — pruned (experiment1 end of experiment)

**Outcome:** 5.6 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 21

**Hypothesis:** People do Bayesian concept learning over familiar mathematical rules and numerical ranges (size principle, a prior weight per rule, an Erlang prior on range width), but they can also entertain a rule confined to a region of the number line bounded by decades — e.g. "numbers ending in 3 between 10 and 50" or "even numbers in the twenties" — with a fitted share of each rule's prior going to such restricted versions. Because a restricted rule is smaller, the size principle favours it whenever the examples all lie in one region, so people extend a rule like "ends in d" or "multiples of m" to far-away numbers less than a pure-rule learner does.

### fuzzy_trace_example_memory — pruned (experiment2 end of experiment)

**Outcome:** 4.7 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 20

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical ranges (with the size principle), but their memory of the examples is imprecise: each example is kept verbatim only with a fitted probability, and otherwise only its rough magnitude (within about five of its true value) is retained. Concepts are evaluated against these fuzzy memory traces, so concepts that depend on an example's exact identity (last digit, multiples, powers) lose support with every imperfectly remembered example while numerical ranges, which only need the magnitude, do not — making people generalise less along digit-based rules to far-away targets, especially from longer sets.

### digit_rule_discount_sticky_guess — pruned (experiment2 end of experiment)

**Outcome:** 3.3 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 20

**Hypothesis:** Refinement of sticky_guess_power_law_ranges: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted, with range guesses shaped by a power-law endpoint and Erlang width prior. The one change: rules defined only by how a number is written ("ends in 7", "contains the digit 3") are a priori less (or more) plausible than arithmetic rules (odd, multiples, powers, squares, primes), by a single fitted weight on the digit-rule family — so people latch onto and extend "same last digit" guesses to far-away numbers less readily than the incumbent predicts.

### sticky_guess_last_digit_rule_prior — pruned (experiment2 end of experiment)

**Outcome:** 4.3 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 20

**Hypothesis:** Refinement of sticky_guess_power_law_ranges: people hold a single working guess about the concept (a rule or a numerical range, with power-law and Erlang priors on ranges), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering only when contradicted. The one change: rules are not all equally natural guesses — the ten "ends in digit d" rules share a single fitted prior weight relative to the other rules, so people latch onto and keep last-digit rules less (or more) readily than arithmetic rules like multiples or powers, which governs how much they endorse far-away numbers merely because they share an example's final digit.

### range_restricted_rule_concepts — pruned (experiment1 end of experiment)

**Outcome:** 4.9 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 21

**Hypothesis:** Refinement of power_law_endpoints_erlang_width (same Bayesian concept learning over the 45 rules and all intervals of 1-100, size principle, 10% noisy examples, a prior weight per rule, power-law endpoint and Erlang width priors on ranges). The one change: people can also entertain a rule restricted to a range of the number line — e.g. "numbers ending in 3 between 10 and 40" rather than "all numbers ending in 3" — with a fitted share of each rule's prior given to its range-restricted version (any range equally plausible). Because such a restricted rule is smaller, the size principle favours it when the examples cluster, so people generalise a rule (especially a surface one like a shared last digit) less to numbers far outside the examples' span than the unrestricted rule predicts.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 6.7 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 21

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### target_as_evidence_sticky_guess — pruned (experiment2 end of experiment)

**Outcome:** 3.5 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 20

**Hypothesis:** Refinement of digit_rule_discount_sticky_guess (Bayesian concept learning over rules and numerical ranges, a single sticky working guess formed from the first example, digit rules discounted). The one distortion: people treat the question itself as weak evidence — they partly assume the experimenter tends to ask about numbers the program actually produces, so the asked-about target is counted, with a fitted fraction of the weight of a real example, as one more sample from the concept. This "leading question" bias inflates "yes" for targets that would belong to a small, specific concept (e.g. a power of 2 after 16 and 8) far more than for targets that only fit broad concepts.

### independent_example_generalization — pruned (experiment2 end of experiment)

**Outcome:** 702.5 nats behind coincidence_triggered_guess_rethink (14.5× dse)

**Hypothesis:** People never pool the examples into a single inferred concept. Each example on its own triggers Bayesian generalization (over mathematical rules and numerical ranges, with the size principle applied to that one example), and the target is endorsed according to the average of these separate single-example generalizations. So adding examples does not sharpen generalization onto a shared rule or fill the gap between them as a range: a target is endorsed to the extent that it is something that would be generalized to from the individual examples, one at a time.

### recency_weighted_guess_revision — pruned (experiment2 end of experiment)

**Outcome:** 6.5 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 20 of 20

**Hypothesis:** Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept, with a fitted stickiness, while each new example fits it. The one change is in how they revise when a new example contradicts the guess (or they do not stick): they rebuild the guess around the newest example, which they weigh fully, while their recall of the earlier examples is only partial — earlier examples' evidence is down-weighted by a fitted retention factor — so revisions are recency-driven and the final concept is pulled toward whatever fits the example that forced the change.

### anomaly_tolerant_sticky_guess — rejected (experiment2 round 0 candidate 3 refine incumbent digit_rule_discount_sticky_guess repair 1)

**Outcome:** predicts like existing model 'digit_rule_discount_sticky_guess' (p_yes RMSE 0.00170 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of digit_rule_discount_sticky_guess, not a new hypothesis.

**Hypothesis:** Refinement of digit_rule_discount_sticky_guess: people hold a single working guess about the concept (a rule or a numerical range, with the incumbent's power-law/Erlang range prior and discounted digit rules), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it. The one change: a contradicting example does not always overturn the guess — with a fitted tolerance people dismiss an example that does not fit their current guess as an anomaly (noise) and keep the guess anyway, rather than always reconsidering from all examples, so a guess formed early survives an odd-one-out example and the outlier itself is endorsed less.

### sticky_guess_fitted_sampling_trust — pruned (experiment2 end of experiment)

**Outcome:** 2.9 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 20

**Hypothesis:** Refinement of digit_rule_discount_sticky_guess: people hold a single working guess about the concept (a rule or a numerical range, with a discounted prior on digit-based rules and power-law/Erlang priors on ranges), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted. The one change (the component taken from inferred_sampling_reliability_concepts): how much people trust that the examples were really drawn from the program's concept, rather than being stray numbers, is fitted instead of fixed at 90% — so the strength of the size principle, and hence how readily people entertain (and answer by) guesses that do not even contain every example, is set by the data.

### fuzzy_trace_sticky_guess — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'fuzzy_trace_sticky_guess' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of fuzzy_trace_example_memory: people do Bayesian concept learning over mathematical rules and numerical ranges (size principle, Erlang prior on range width) against fuzzy memory traces of the examples — each kept verbatim only with a fitted probability, otherwise only as its rough magnitude (within about five). The one change, the component taken from first_guess_win_stay_concepts: people do not weigh all the traces at once but hold a single working guess formed from the first example, keep it with a fitted stickiness while each newly remembered example fits it, and reconsider from all the traces so far only when the remembered example contradicts it — so imprecise memory weakens digit and arithmetic rules while early guesses persist, making generalisation depend on both memory precision and example order.

### consistency_attention_similarity — pruned (experiment2 end of experiment)

**Outcome:** 443.4 nats behind coincidence_triggered_guess_rethink (13.5× dse)

**Hypothesis:** People judge a target by its similarity to the examples, but they attend to each dimension of number similarity (magnitude band, last digit, parity, divisibility, being a square, power or prime) in proportion to how surprisingly consistently the examples agree on it: a property all examples share counts more the rarer it is among 1-100 and the more examples share it. The target's psychological distance is the summed attention on the shared properties it lacks plus its number-line distance from the nearest example, and the probability of "yes" falls off along this distance — so a far-away number is endorsed when it shares a rare property the examples all have, and a near number is rejected when it breaks one.

### sampled_guess_vote_threshold — pruned (experiment2 end of experiment)

**Outcome:** 3.5 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 20

**Hypothesis:** People form guesses about the concept exactly as in the simplest sticky-guess learner (a working guess from the first example, kept while later examples fit it, revised from all examples when contradicted), but they do not answer by reporting one guess: they call to mind a handful of such candidate guesses and say "yes" only if a sufficient share of them contain the target. The effective number of guesses consulted and the share required (a yes/no criterion) are fitted, so the same belief maps onto a sharpened or softened, more liberal or more conservative yes rate; one guess with no criterion recovers plain probability matching.

### disjunctive_rule_union_concepts — rejected (experiment2 round 1 candidate 2 lens 9 repair 1)

**Outcome:** too slow to fit: the fit of 'disjunctive_rule_union_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People's space of candidate concepts includes not only single rules (odd, multiples of 4, ends in 1, ...) and numerical ranges, but also "either-or" disjunctions of two familiar rules ("a square or ends in 1", "even or ends in 5"), entertained with a fitted, complexity-penalised prior share. When the examples do not all fit one rule, people can explain them with such a two-rule union instead of writing off an example as a stray, and then endorse targets that belong to either rule — while still holding a single sticky working guess formed from the first example, as in the current best model.

### rule_vs_range_stickiness_guess — rejected (experiment2 round 1 candidate 3 refine incumbent sticky_guess_fitted_sampling_trust)

**Outcome:** predicts like existing model 'sticky_guess_fitted_sampling_trust' (p_yes RMSE 0.00166 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of sticky_guess_fitted_sampling_trust, not a new hypothesis.

**Hypothesis:** Refinement of sticky_guess_fitted_sampling_trust (a single working guess about the concept — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept while later examples fit it and reconsidered from all examples when contradicted, with digit rules discounted and a fitted trust in the examples). The one change: how firmly people hold their guess depends on its kind — a rule ("powers of 2", "ends in 3") is a categorical commitment kept with one fitted stickiness, whereas a numerical range is only a provisional summary of where the examples lie, kept with its own, separately fitted stickiness and more readily re-estimated from all examples as new ones arrive even when they fit it.

### sticky_guess_attentional_lapse — pruned (experiment2 end of experiment)

**Outcome:** 4.9 nats behind coincidence_triggered_guess_rethink (2.6× dse)

**Hypothesis:** Refinement of sticky_guess_fitted_sampling_trust: people hold a single working guess about the concept (a rule or a numerical range, digit rules discounted, power-law/Erlang priors on ranges, fitted trust in the examples), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted. The one change (the component taken from lapse_biased_guess_concepts): on a fitted fraction of trials people do not consult their guess at all and simply guess, answering "yes" with a fitted bias — which explains why they reject even the examples themselves about a quarter of the time without forcing the concept learner to distrust its examples, and puts a common floor under "yes" for numbers outside every plausible concept.

### digit_coincidence_noticing_guess — dropped (experiment3)

**Outcome:** MCMC fit failed on CV fold 1 (AssertionError: Apply node that caused the error: Subtensor{start:stop}(joined_inputs, 2, 3) Toposort index: 13 Inputs types: [TensorType(float64, shape=(7,)), ScalarType(int64), ScalarType(int64)] Backtrace when the node is created (use PyTensor flag traceback__limit=N to make it longer): File "/scr/wurgaft/auto-psych/number_game_recovery-nov26b-c1/harness_repo/src/models/pymc_inference.py", line 984, in _fit_once idata = _warm_start_sample(model, settings) File "/scr/wurgaft/auto-psych/number_game_recovery-nov26b-c1/harness_repo/src/models/pymc_inference.py", line 1027, in _warm_start_sample step = pm.NUTS(target_accept=settings["target_accept"], max_treedepth=settings["max_treedepth"]) File "/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/venv/lib/python3.12/site-packages/pymc/step_methods/hmc/nuts.py", line 198, in __init__ super().__init__(vars, **kwargs) File "/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/venv/lib/python3.12/site-packages/pymc/step_methods/hmc/base_hmc.py", line 141, in __init__ super().__init__( File "/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/venv/lib/python3.12/site-packages/pymc/step_methods/arraystep.py", line 193, in __init__ logp_dlogp_func = model.logp_dlogp_function( File "/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/venv/lib/python3.12/site-packages/pymc/model/core.py", line 570, in logp_dlogp_function return ValueGradFunction( File "/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/venv/lib/python3.12/site-packages/pymc/model/core.py", line 245, in __init__ outputs, raveled_grad_vars = join_nonshared_inputs( File "/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/venv/lib/python3.12/site-packages/pymc/pytensorf.py", line 592, in join_nonshared_inputs joined_inputs[last_idx : last_idx + arr_len].reshape(shape).astype(var.dtype) HINT: Use a linker other than the C linker to print the inputs' shapes and strides. HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.)

**Hypothesis:** Refinement of the incumbent sticky_guess_fitted_sampling_trust (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted, fitted trust in the examples). The one change: a rule defined only by how numbers are written ("ends in 6", "contains the digit 3") becomes a salient candidate only once people notice that digit as a coincidence shared by at least two of the examples seen so far; until then — always after a single example, and for a digit only one example has — its plausibility is reduced by a fitted availability factor. So a lone 46 does not readily lead people to say yes to 96, whereas 46 and 86 together do.

### fuzzy_boundary_range_sticky_guess — rejected (experiment2 round 1 candidate 3 refine incumbent sticky_guess_fitted_sampling_trust repair 1)

**Outcome:** too slow to fit: the fit of 'fuzzy_boundary_range_sticky_guess' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of sticky_guess_fitted_sampling_trust (a single working guess about the concept — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with a fitted stickiness while later examples fit it and reconsidered from all examples when contradicted, with digit rules discounted and a fitted trust in the examples). The one change: when people answer from a range guess they treat its boundaries as fuzzy ("roughly 20 to 40") rather than sharp, so a target just outside the range is still endorsed, with a probability that falls off with its distance beyond the nearer boundary at a fitted rate — rule guesses stay crisp. This predicts graded "yes" for near-misses just outside the examples' span that the incumbent's sharp ranges reject, while far-away numbers fitting only a rule are unaffected.

### category_step_exemplar_similarity — pruned (experiment2 end of experiment)

**Outcome:** 45.8 nats behind coincidence_triggered_guess_rethink (4.4× dse)

**Hypothesis:** People do not infer a concept; they retrieve familiar number categories stored in memory as ordered lists of their members (odd numbers, powers of 2, numbers ending in 6, ranges bounded by multiples of five, ...), each activated by how similar the examples are to its stored members. They then endorse a target by its exemplar similarity to the nearest example measured along each activated category — how far through the category's member list they must step from an example to reach the target — so a member only a step or two away from an example (32 after 16 in the powers of 2) is endorsed readily while one many steps away (96 after a lone 46 in "ends in 6") is endorsed much less.

### representativeness_next_output_judgment — pruned (experiment2 end of experiment)

**Outcome:** 196.6 nats behind coincidence_triggered_guess_rethink (9.2× dse)

**Hypothesis:** People do not answer by asking whether the target belongs to the concept they have inferred; they judge by representativeness — how typical the target would be as the program's next output. Using the same concept learning as the sticky single-guess learner (a working guess formed from the first example and kept while later examples fit it), they estimate how much more likely the program is to produce this particular number than an arbitrary number from 1-100, and say "yes" the more strongly that likelihood ratio exceeds a fitted criterion. So a target that belongs only to a broad concept (all even numbers, everything ending in 6) is endorsed much less than one that belongs to a small, specific concept, even when both are certainly members.

### coincidence_gated_rule_guess — pruned (experiment2 end of experiment)

**Outcome:** 3.2 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 20

**Hypothesis:** People do not suspect a rule of any kind — arithmetic ("powers of 2", "multiples of 7", "squares") or written ("ends in 6") — until it is suggested by a coincidence: a rule becomes a salient candidate only once at least two of the examples seen so far satisfy it, and before that its plausibility is reduced by a fitted availability factor, so a lone example is generalised mostly as a numerical range around it. Otherwise they hold a single sticky working guess formed from the first example by Bayesian concept learning, as in the current best model (whose availability penalty applies only to digit rules). The sharpest disagreement lives in single-example sets and in sets where only one example fits an arithmetic rule: after a lone 16 the best model still endorses far-away 64, 4 or 32 as powers of 2, whereas this model predicts these far rule-consistent targets are rejected until a second example such as 8 makes the rule a noticed coincidence.

### rule_coincidence_noticing_guess — rejected (experiment2 round 2 candidate 3 refine incumbent digit_coincidence_noticing_guess)

**Outcome:** predicts like existing model 'coincidence_gated_rule_guess' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of coincidence_gated_rule_guess, not a new hypothesis.

**Hypothesis:** Refinement of digit_coincidence_noticing_guess (a single sticky working guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, digit rules discounted, fitted trust in the examples, and rules waiting to be noticed as a coincidence). The one change: it is not only rules about how numbers are written that must be noticed — any rule (odd, even, multiples, powers, squares, primes, as well as "ends in 6") becomes a salient candidate only once at least two of the examples seen so far share it, and until then its plausibility is reduced by one fitted availability factor. So a single example is read mainly as a place on the number line, and people endorse far-away numbers that merely share a property (parity, last digit, a divisor) with one lone example less than the incumbent predicts, while two examples sharing that property make the rule readily available.

### rule_needs_two_examples_guess — rejected (experiment2 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'coincidence_gated_rule_guess' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of coincidence_gated_rule_guess, not a new hypothesis.

**Hypothesis:** Refinement of sticky_guess_fitted_sampling_trust (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted, fitted trust in the examples). The one change: any rule at all — arithmetic ("even", "powers of 2", "multiples of 4") as well as digit-based ("ends in 6") — becomes a salient candidate only once people notice it as a regularity shared by at least two of the examples seen so far; until then (always after a single example) its plausibility is reduced by one fitted availability factor, so a lone example is generalised mostly as a nearby numerical range. Unlike digit_coincidence_noticing_guess, which delays only digit rules, this predicts that a lone 46 does not readily lead people to say yes to 96 through "even" either, and a lone 16 does not readily suggest 64 through "powers of 2", while two examples sharing the rule do.

### width_then_place_range_guess — rejected (experiment2 round 2 candidate 3 refine incumbent digit_coincidence_noticing_guess repair 1)

**Outcome:** too slow to fit: the fit of 'width_then_place_range_guess' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of digit_coincidence_noticing_guess (a single sticky working guess about the concept — a rule or a numerical range — formed from the first example by Bayesian concept learning, digit rules discounted and made salient only once a digit is noticed as shared by two examples, fitted trust in the examples). The one change is in how people imagine the program picks a numerical range: they first decide how wide it is (the same preference for a typical width) and only then place a range of that width anywhere it fits on 1-100, so every width gets its share of plausibility no matter how many placements it has, rather than each individual range being weighted by its width. Because a wide range has few placements, each one becomes individually more plausible, while the width preference itself becomes much narrower; a lone example is therefore generalised along a steeper number-line gradient (and far-away numbers sharing its last digit are endorsed less), whereas widely spread examples still support a broad range.

### sticky_guess_decade_restricted_rules — rejected (experiment2 round 2 candidate 5 refine chosen repair 1)

**Outcome:** too slow to fit: the fit of 'sticky_guess_decade_restricted_rules' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of sticky_guess_fitted_sampling_trust (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning and kept with fitted stickiness while later examples fit it, reconsidered from all examples only when contradicted, digit rules discounted, fitted trust in the examples). The one change, the component taken from decade_restricted_rule_concepts: people's working guess can also be a rule confined to a decade-bounded region of the number line ("ends in 6 in the forties", "even numbers between 21 and 50"), with a fitted share of each rule's plausibility going to such restricted versions; because a restricted rule is small, the size principle favours it whenever the examples sit in one region, so people extend a rule — especially a shared last digit from a single example, like 46 to 96 — to far-away numbers much less than a whole-number-line rule predicts, while still generalising it to nearby members.

### one_back_digit_coincidence_guess — pruned (experiment2 end of experiment)

**Outcome:** 4.9 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 20

**Hypothesis:** People hold a single working guess about the concept (a rule or a numerical range), formed from the first example and kept while later examples fit it, and a rule about how numbers are written ("ends in 6", "contains the digit 3") becomes a salient guess only once they notice its digit as a coincidence. They notice such coincidences by a one-back working-memory comparison, setting each new example against the example shown just before it, so a shared digit is noticed only when two consecutive examples have it. So the order of the examples matters: "46, 86, 13" makes "ends in 6" salient and leads people to say yes to 96, whereas "46, 13, 86" does not, although both contain the same numbers.

### range_strain_triggered_rule_search — pruned (experiment2 end of experiment)

**Outcome:** 3.1 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 20

**Hypothesis:** People look for a numerical range first and search for a rule only when a range is a strained explanation: the more widely the examples seen so far are spread over the number line, the more readily they consider rules ("powers of 2", "even", "ends in 6"), whereas a single example or a tight cluster of examples is read as a place on the number line and rules stay largely unconsidered (by a fitted availability factor). Otherwise they hold a single sticky working guess formed from the first example by Bayesian concept learning, as in the current best model — so clustered examples that share a rule (20, 22, 24) are generalised as a range, while scattered ones (16, 8, 64) prompt the rule.

### graded_pairwise_digit_noticing — rejected (experiment2 round 3 candidate 4 refine incumbent digit_coincidence_noticing_guess)

**Outcome:** too slow to fit: the fit of 'graded_pairwise_digit_noticing' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of digit_coincidence_noticing_guess (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted, fitted trust in the examples, and rules about how numbers are written made salient only by a noticed shared digit). The one change: noticing a shared digit is not all-or-none at two examples but probabilistic — every pair of examples seen so far that share the digit is a separate chance (with a fitted probability) of spotting the coincidence — so a digit shared by only two examples is often missed, while one shared by three or four examples (three or six pairs) is almost always noticed, and "ends in 6" or "contains a 3" becomes a salient guess gradually as more examples carry the digit.

### coincidence_insight_guess_revision — rejected (experiment2 round 3 candidate 5 refine chosen)

**Outcome:** too slow to fit: the fit of 'coincidence_insight_guess_revision' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of coincidence_gated_all_rules_guess (a single sticky working guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, with rules made salient only once two examples share them). The one change: noticing a coincidence is itself a trigger for rethinking — when a new example makes a rule shared by two examples for the first time, people abandon their current guess and reconsider the concept from all examples (even if the guess still fits), with a fitted probability scaled by how well the newly noticed rule explains the examples. So a guess formed early, for instance from an unrepresentative first example, does not block a pattern that later examples reveal, and order effects are weaker than pure stickiness predicts.

### place_value_grid_block_concepts — pruned (experiment2 end of experiment)

**Outcome:** 540.8 nats behind coincidence_triggered_guess_rethink (12.3× dse)

**Hypothesis:** People do not think of the numbers 1-100 as points on a number line but as cells of a place-value grid (tens digit by ones digit), and besides familiar arithmetic rules (even, multiples, powers, squares, primes) the only concepts they entertain are blocks on that grid — a run of tens digits crossed with a run of ones digits, such as "the forties", "ends in 6", "ends in 5 to 7" or "20s and 30s ending in 3 to 7". They learn which block or rule the program uses by Bayesian concept learning with a single sticky working guess formed from the first example, so after 27 and 33 they endorse 23 and 37 (same block of digits) more than 30, which a number-line range would favour.

### conservative_range_stretching_guess — rejected (experiment2 round 4 candidate 1 lens 6 repair 1)

**Outcome:** too slow to fit: the fit of 'conservative_range_stretching_guess' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** When a new example falls outside the numerical range people are currently guessing, they often do not rethink the concept from scratch: with a fitted probability they make the minimal revision, stretching their range just far enough to take in the new example (so a range around 46 becomes "46 up to 86" after 86 arrives), and only otherwise reconsider from all the examples. Because the stretched range depends on which range was guessed first and on the order in which examples arrive, the same examples in different orders produce different generalisations, and contradicted range guesses grow into ranges spanning the examples instead of giving way to rules like "ends in 6" (otherwise as in the current best sticky-guess learner with coincidence-noticing of digit rules, with ranges weighted only by their width).

### cohesion_selected_metric_similarity — pruned (experiment2 end of experiment)

**Outcome:** 450.9 nats behind coincidence_triggered_guess_rethink (12.8× dse)

**Hypothesis:** People judge the target by its similarity to the examples, but on whichever single metric of number similarity the examples themselves cluster on most surprisingly: either closeness on the number line (with a graded, exponentially decaying gradient) or sharing the same remainder when divided by some k from 2 to 10 (same parity, same last digit, both multiples of 7, and so on). They attend to the metric on which the examples are more alike than random numbers would be, and say "yes" in proportion to how similar the target is to the nearest example on that metric — so scattered examples that share a residue (8, 80, 48) are generalised to far numbers with that residue, while examples that share no residue are generalised only to nearby numbers.

### coincidence_kind_graded_rethink — rejected (experiment2 round 4 candidate 4 refine incumbent coincidence_triggered_guess_rethink)

**Outcome:** predicts like existing model 'coincidence_triggered_guess_rethink' (p_yes RMSE 0.00024 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of coincidence_triggered_guess_rethink, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent coincidence_triggered_guess_rethink (a single sticky working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, digit rules discounted and salient only once a digit is shared by two examples, fitted trust in the examples, and a newly noticed shared digit triggering a rethink of the guess from all examples). The one change: how disruptive a noticed coincidence is depends on its kind — a newly shared final digit (46 then 86: "ends in 6") is a striking, structurally aligned pattern that triggers a rethink with its own fitted probability, whereas a digit that merely appears somewhere in both numbers (16 then 64, or 13 then 31) is a weaker, less noticeable coincidence that triggers a rethink with a separate fitted probability, instead of both sharing one probability.

### target_prompted_coincidence_rethink — rejected (experiment2 round 4 candidate 4 refine incumbent coincidence_triggered_guess_rethink repair 1)

**Outcome:** too slow to fit: the fit of 'target_prompted_coincidence_rethink' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of the incumbent coincidence_triggered_guess_rethink. It keeps everything: a single sticky working guess (a rule or a numerical range) formed from the first example by Bayesian concept learning, digit rules discounted and salient only once two numbers seen share the digit, fitted trust in the examples, and a newly noticed shared digit making people drop a guess that still fits and rethink from all the examples. The one change: the number being asked about can itself reveal a coincidence. When the target shares a digit with exactly one example (being asked about 96 after a lone 46, or after 46 and 13), people notice that coincidence while answering. With the same fitted probability as any other noticed coincidence, they then rethink from all the examples, and the digit rule covering both numbers ("ends in 6") is now salient. This adds no new parameter, so targets that share a digit with a single example are endorsed more than the incumbent predicts, even when no two examples share that digit.

### two_digit_ending_union_concepts — rejected (experiment3 round 0 candidate 1 lens 9 repair 1)

**Outcome:** too slow to fit: the fit of 'two_digit_ending_union_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People's space of candidate concepts includes, besides single rules and numerical ranges, "either-or" last-digit concepts — numbers ending in one of two digits, such as "ends in 2 or 7" (82, 67, 72) or "ends in 1 or 5" — which they entertain as written-form rules with their own fitted prior plausibility, shared equally among the 45 such pairs. When the examples end in just two different digits, people can therefore generalise to far-away numbers ending in either digit (62 or 17 after 82, 67, 72) instead of falling back on a range or writing an example off as a stray; otherwise they learn as the current best single-sticky-guess learner does.

### evidence_scaled_misfit_tolerance — rejected (experiment3 round 0 candidate 3 refine incumbent misfit_tolerant_coincidence_rethink repair 1)

**Outcome:** predicts like existing model 'misfit_tolerant_coincidence_rethink' (p_yes RMSE 0.00197 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of misfit_tolerant_coincidence_rethink, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent misfit_tolerant_coincidence_rethink. The incumbent holds a single working concept guess, a rule or a numerical range, formed from the first example by Bayesian concept learning. The guess is kept with fitted stickiness while later examples fit it; digit rules are discounted and become salient only once two examples share a digit; a newly noticed coincidence triggers a rethink; trust in the examples is fitted; and a misfitting example is sometimes written off as a stray. The one change: how readily a contradicting example is written off depends on how well established the guess is. Each earlier example the guess explains independently gives a fitted chance of dismissing the misfit. A guess supported by one example is then easily abandoned, one supported by two or three examples (77, 17 → "ends in 7") usually survives an outlier, and a guess supported by none is never protected. (The incumbent's weakly constrained preference for ranges at particular places on the number line is dropped as a simplification; ranges keep their preferred-width prior.)

### compressed_size_coincidence_rethink — rejected (experiment3 round 0 candidate 5 refine chosen repair 1)

**Outcome:** too slow to fit: the fit of 'compressed_size_coincidence_rethink' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of compressed_concept_size_sticky_guess (a single sticky working guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, digit rules discounted and made salient only once two examples share the digit, fitted trust in the examples, and concept sizes perceived on a compressed, numerosity-like scale). The one change, the component taken from coincidence_triggered_guess_rethink: noticing a coincidence is itself a trigger to rethink — when a new example shares a digit with an earlier one for the first time, people drop their current guess with a fitted probability and reconsider the concept from all the examples, even if the guess still fits, and that reconsideration uses the same compressed sense of concept size. So after 46 then 86 a range guess formed from 46 is often replaced by "ends in 6", and because "ends in 6" does not feel as much smaller than a broad range as it really is, the rethought concept is broader than a normative learner's.

### compressed_size_evidence_graded_tolerance — rejected (experiment3 round 1 candidate 3 refine incumbent evidence_graded_misfit_tolerance)

**Outcome:** predicts like existing model 'evidence_graded_misfit_tolerance' (p_yes RMSE 0.00157 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of evidence_graded_misfit_tolerance, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent evidence_graded_misfit_tolerance (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted and salient only once two examples share the digit, a newly noticed coincidence triggering a rethink, fitted trust, and a contradicting example written off as a stray more readily the more earlier examples the guess explains). The one change, the component taken from compressed_concept_size_sticky_guess: when people weigh how well each candidate guess explains the examples, they misjudge how many numbers the guess contains, perceiving its size on a compressed, numerosity-like scale anchored at the whole range 1-100 (compression fitted), so the size principle's preference for small, specific guesses ("powers of 2", a narrow range) over broad ones ("even numbers") is weakened or strengthened while the rest of the sequential guess-keeping process is unchanged.

### evidence_graded_guess_commitment — rejected (experiment3 round 1 candidate 4 refine incumbent evidence_graded_misfit_tolerance)

**Outcome:** predicts like existing model 'evidence_graded_misfit_tolerance' (p_yes RMSE 0.00144 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of evidence_graded_misfit_tolerance, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent evidence_graded_misfit_tolerance (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, digit rules discounted and salient only once two examples share the digit, a newly noticed coincidence triggering a rethink, fitted trust, and a contradicting example written off as a stray more readily the more earlier examples the guess explains). The one change: the same evidence-grading applies to keeping a guess that the new example fits — rather than a fixed chance of sticking, each earlier example the guess explains is an independent reason not to reconsider, so commitment to a guess grows with its confirmations (one minus the chance that none of them holds, with a fitted per-example stickiness). So a guess that has survived three examples is almost never reconsidered on the fourth, while one formed from a single example is often revised when a second one arrives, making late examples in long sets less able to reshape the concept than early ones.

### magnitude_scaled_range_width — rejected (experiment3 round 1 candidate 4 refine incumbent evidence_graded_misfit_tolerance repair 1)

**Outcome:** too slow to fit: the fit of 'magnitude_scaled_range_width' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of the incumbent evidence_graded_misfit_tolerance (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted and salient only once two examples share the digit, a newly noticed coincidence triggering a rethink, fitted trust, and a contradicting example written off as a stray more readily the more earlier examples the guess explains). The one change: the typical width people expect of a numerical range is not fixed but grows with the magnitude of the examples seen so far, as Weber's law predicts for the mental number line (scaling exponent fitted, expected near proportional). So after 82 and 92 people imagine a broader range and generalise further on the number line than after 2 and 12, and range guesses compete with rules more for large numbers than for small ones.

### partial_recall_guess_resampling — dropped (experiment3 round 2)

**Outcome:** fold 0 too slow to fit: the fit of 'partial_recall_guess_resampling' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People learn the concept as the current best sequential learner does (a single working guess — a rule or a numerical range — formed from the first example, kept while later examples fit it, digit rules salient once two examples share a digit, coincidences triggering a rethink, and misfits written off more readily the more evidence the guess has), but their memory of the earlier examples is limited: when they drop their guess and rethink, they always use the example just shown but recall each earlier example only with a fitted probability, and choose the new guess from the examples they recall. So a rethink driven by a late example often lands on a concept that fits that example and only some of the earlier ones (the classic "resample from what you remember" hypothesis-testing process), and the working guess, not a full record of the examples, is what carries the past.

### sublinear_evidence_accumulation — rejected (experiment3 round 2 candidate 2 lens 5)

**Outcome:** predicts like existing model 'sample_size_insensitive_guess' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of sample_size_insensitive_guess, not a new hypothesis.

**Hypothesis:** People treat additional examples as partly redundant rather than as independent evidence: the support that several examples lend a candidate concept grows with diminishing returns in the number of examples (as a fitted power of the count), instead of adding up example by example as the size principle assumes. So as examples accumulate, people narrow their guess toward small, specific concepts (tight ranges, "powers of 2") more slowly than a normative learner, and keep endorsing targets outside the examples' span after three or four clustered examples nearly as much as after two — while otherwise forming and keeping a single working guess as the current best learner does.

### fuzzy_edge_range_guess — rejected (experiment3 round 2 candidate 3 refine incumbent gamma_width_range_prior_guess repair 1)

**Outcome:** too slow to fit: the fit of 'fuzzy_edge_range_guess' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of the incumbent gamma_width_range_prior_guess (a single working concept guess — a rule or a numerical range with a fitted-shape gamma width prior — formed from the first example by Bayesian concept learning, kept with fitted stickiness, digit rules discounted and salient only once two examples share the digit, a newly noticed coincidence triggering a rethink, fitted trust, and misfits written off more readily the more examples the guess explains). The one change: when people's working guess is a numerical range, its edges are fuzzy rather than sharp — a target just outside the guessed range is still judged a possible output, with willingness falling off exponentially with its distance beyond the nearer edge at a fitted reach, while rules keep crisp membership. So after clustered examples such as 24 and 35, numbers a few steps past the edges (19, 40) are endorsed more than a sharp-edged range predicts, without raising endorsement of far rule members.

### arithmetic_membership_check_errors — rejected (experiment3 round 2 candidate 2 lens 5 repair 1)

**Outcome:** too slow to fit: the fit of 'arithmetic_membership_check_errors' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the program's concept as the current best learner does (a single sticky working guess — a rule or a numerical range — formed from the first example, with digit rules salient once two examples share a digit, coincidences triggering a rethink, and misfits written off more readily the more evidence the guess has), but when they answer they must check whether the target belongs to their guess, and for rules that require mental arithmetic ("multiples of 7", "primes", "squares", "powers of 2", "odd") that check is error-prone: with a fitted probability they get the membership wrong. Rules read off the written number ("ends in 6", "contains a 3") and numerical ranges are checked without error, so under an arithmetic-rule guess members are sometimes rejected and non-members sometimes endorsed, while digit-rule and range guesses yield crisp answers.

### recency_anchored_extension — rejected (experiment3 round 3 candidate 0 lens 6)

**Outcome:** too slow to fit: the fit of 'recency_anchored_extension' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of the best model, gamma_width_distance_hesitant_guess. Concept learning is unchanged: a single sticky working guess (a rule, or a range with a gamma-shaped prior on its width) formed from the first example, digit rules salient once two examples share a digit, a newly noticed coincidence triggering a rethink, and misfits written off more readily the more examples the guess explains. People still hesitate to extend their guessed concept to numbers far from the examples. The one change is that this willingness is anchored on the examples in proportion to how recently each was shown, not on whichever example is nearest. The example just shown fully licenses nearby targets, and each earlier example licenses them less, by a fitted fading factor per intervening example. So the same examples in a different order produce different gradients: after 20, 80, a target near 80 is endorsed more than one near 20, and after 80, 20 the opposite holds.

### log_ratio_similarity_hesitant_guess — rejected (experiment3 round 3 candidate 1 lens 7)

**Outcome:** too slow to fit: the fit of 'log_ratio_similarity_hesitant_guess' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of the best model, gamma_width_distance_hesitant_guess. People infer a single sticky working concept guess (a rule, or a numerical range with a gamma-shaped width prior) by Bayesian concept learning from the first example, with digit rules salient once two examples share a digit, coincidences triggering a rethink, fitted trust, and misfits written off more readily the more examples the guess explains. They then hesitate to extend that guess to a target in proportion to its similarity to the nearest example. The one change is the similarity metric: closeness is judged by ratio on a logarithmically compressed mental number line, not by raw difference. Willingness falls off smoothly (Gaussian-like) with the log-ratio distance at a fitted reach. So the same numerical gap is a small step among large numbers (90 after 80) but a long leap among small ones (12 after 2), and far rule members of small examples are discounted most.
