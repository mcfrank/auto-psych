# Tried before

65 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### digit_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 353.5 nats behind iter4_candidate5 (10.6× dse)

**Hypothesis:** People do not induce a rule at all; they judge a target by its summed similarity to the remembered examples, where numbers are represented by their two decimal digits: a target feels like it "belongs" to the extent that its tens digit is close to an example's tens digit and its ones digit matches an example's ones digit. Attention is split between the tens (magnitude) and ones (final-digit) dimensions, similarity decays exponentially with this digit-wise distance, and the probability of saying yes rises with the average similarity of the target to the examples.

### digit_overlap_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 389.6 nats behind iter4_candidate5 (11.9× dse)

**Hypothesis:** People do not induce mathematical rules or magnitude intervals at all; they judge a target by its surface resemblance to the examples, treating each number as the set of decimal digits it is written with, and say "yes" in proportion to the average digit-overlap similarity between the target and the examples (an exemplar-similarity process over written digits). This disagrees most sharply with the rule-based Bayesian model on targets that share a digit with some examples but fit no rule or interval consistent with all of them (e.g. 12 after "31,19,49,100"), and on rule-consistent targets written with entirely different digits (e.g. 2 after "16,8,64"), where it predicts low endorsement.

### serial_position_weighted_concepts — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00147 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** People infer the hidden concept by Bayesian generalization over rules and intervals, but they do not weigh the examples equally: how much an example counts as evidence depends on where it came in the order shown. One serial-position gradient, inferred from the data, can favour the first examples (primacy) or the most recent ones (recency). So a concept that fits the heavily weighted examples is endorsed even if it fits the others poorly, and the same set shown in a different order would give a different pattern of generalization.

### partial_size_principle_concepts — rejected (experiment1 round 0 candidate 3 refine incumbent bigelow_piantadosi_independent)

**Outcome:** too slow to fit: the fit of 'partial_size_principle_concepts' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learner over its 45 rules and all intervals of 1-100, with a prior weight inferred for each rule, but people apply the size principle only partially — each example's likelihood under a concept scales with the concept's size raised to an inferred exponent (1 = the incumbent's strong sampling, 0 = weak sampling that ignores concept size). The single change is this free size-principle strength, which governs how quickly a few examples make people favour small, specific concepts over broad ones.

### bayesian_rules_with_response_lapse — pruned (experiment1 end of experiment)

**Outcome:** 5.5 nats behind iter4_candidate5 (2.0× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, a prior weight inferred per rule), but on a fixed fraction of trials they do not consult that posterior at all and answer yes or no at random. The one change is this response-lapse rate, which lets the model explain "no" answers to targets that are themselves examples and stray "yes" answers to targets that no plausible concept contains.

### tenenbaum_digit_rule_inventory — pruned (experiment1 end of experiment)

**Outcome:** 6.6 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 19

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Tenenbaum's Bayesian concept learning (rules share one prior fraction equally, intervals are weighted by an Erlang density on their size, examples are sampled from the concept), but the rules they entertain are the richer inventory of the Bigelow-Piantadosi grammar — notably "contains the digit d" and "ends in 0", plus the two near-miss powers-of-2 sets — instead of Tenenbaum's 33. The one change is the hypothesis space; the claim is that what the people consider, not a separately learned weight for every rule, is what separates the two models, so equal rule weights with two free parameters should suffice.

### spread_scaled_range_generalization — pruned (experiment1 end of experiment)

**Outcome:** 421.3 nats behind iter4_candidate5 (12.7× dse)

**Hypothesis:** People do not search over mathematical rules; they treat the examples as a sample from a band of magnitudes and generalize by how far the target falls outside the span the examples cover, with a tolerance that grows with how spread out the examples are. A target inside the span of the examples is endorsed at a constant high rate, and endorsement decays smoothly with the target's distance beyond the nearest edge, more slowly for widely spaced examples than for tightly clustered ones (or a single example).

### interval_width_prior_concepts — pruned (experiment1 end of experiment)

**Outcome:** 4.9 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 19

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learner over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100, with the size principle, but people's prior over magnitude intervals depends on the interval's width — each interval's prior is proportional to its width raised to an inferred exponent (0 recovers the incumbent's uniform interval prior; negative values mean narrow ranges like "numbers in the 60s" are a priori more plausible than wide ones). The single change is this width-dependent interval prior, which governs how readily people generalise to nearby versus distant magnitudes when no rule explains the examples.

### leaky_example_memory_concepts — pruned (experiment1 end of experiment)

**Outcome:** 5.7 nats behind iter4_candidate5 (2.1× dse)

**Hypothesis:** People generalize by Bayesian concept learning over mathematical rules and magnitude intervals (the size principle, a prior weight per rule), but their working memory is leaky: each example shown is independently retained with some probability and the forgotten ones play no part in the inference. So a judgment is the rule-and-interval posterior computed from whichever examples survive, which predicts weaker rule-based tightening and broader generalization for larger example sets than full Bayesian learning, and occasional "no" answers to targets that were themselves examples.

### conservative_evidence_weighting — rejected (experiment1 round 1 candidate 1 lens 4)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00116 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, a prior weight inferred per rule), but they are conservative updaters who under-weight (or over-weight) the evidence of the examples — the examples' likelihood enters the posterior raised to a single inferred evidence-weight exponent (1 = normative; below 1 = conservatism, so a few examples move belief less than they should and generalization stays broad; above 1 = over-reaction). The one change is this mis-weighting of the evidence relative to the prior, which predicts that adding examples sharpens generalization more slowly (or faster) than Bayes prescribes.

### map_single_concept_commitment — pruned (experiment1 end of experiment)

**Outcome:** 244.6 nats behind iter4_candidate5 (9.6× dse)

**Hypothesis:** People do not average over all the concepts consistent with the examples; they commit to the single concept that best explains them — the most probable rule or magnitude interval under a size-principle Bayesian evaluation with a flat prior (rules as a group as likely as intervals as a group) — and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. This predicts all-or-none generalization: no graded falloff near the examples' range and no partial endorsement of targets that only some plausible concepts contain (e.g. 32 after "16,8,64" is rejected if the smaller "powers of 2 without 32" wins).

### inferred_example_noise_concepts — pruned (experiment1 end of experiment)

**Outcome:** 3.9 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 19

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but how much they trust the examples to come from the concept is a free parameter rather than fixed at 90%. The one change is this inferred example-noise rate: the more people believe some examples are stray draws from 1-100, the more readily they keep a tight rule or range that misses one example, rather than retreating to a broad concept that covers all of them.

### size_principle_tempered_concepts — rejected (experiment1 round 1 candidate 4 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00153 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100, but they weigh the examples' evidence with an inferred strength rather than as an ideal sampler — the likelihood of the examples under each concept (the size principle) is raised to a free exponent, so values below 1 mean people generalise more broadly than the examples warrant (weaker sampling assumptions) and values above 1 mean they commit to the smallest consistent concept faster. The single change is this evidence-weighting exponent, which governs how sharply additional examples narrow generalisation, e.g. from "even numbers" to "powers of 2" after 16, 8, 64.

### inferred_outlier_rate_rule_inventory — pruned (experiment1 end of experiment)

**Outcome:** 7.6 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 19

**Hypothesis:** Refinement of `tenenbaum_digit_rule_inventory`: people do the same Bayesian concept learning over the 45 Bigelow-Piantadosi rules (one equal, shared prior weight) and all magnitude intervals (Erlang prior on size), with examples sampled from the concept, but how strongly they believe an example may be a random stray rather than a draw from the concept is a property of the learner, inferred from the data instead of fixed at 10%. The one change is this inferred outlier rate: a high rate means people tolerate examples that break a rule and weaken the size principle (generalising more broadly), while a low rate means a single inconsistent example rules a concept out and tight concepts dominate.

### log_odds_compressed_concept_posterior — pruned (experiment1 end of experiment)

**Outcome:** 5.5 nats behind iter4_candidate5 (2.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, a prior weight inferred per rule), but they do not report the posterior probability that the target belongs faithfully — their subjective sense of that probability is distorted linearly in log-odds, so confident beliefs are pulled toward (or pushed away from) even odds, as in classic probability-weighting. The one change is this single probability-distortion slope, which predicts that near-certain "yes" and near-certain "no" cases are both answered less extremely, with the compression strongest for the most confident judgments, unlike a fixed random-response lapse that adds the same floor everywhere.

### decade_aligned_interval_prior — rejected (experiment1 round 1 candidate 4 refine incumbent bigelow_piantadosi_independent repair 1)

**Outcome:** predicts like existing model 'bigelow_piantadosi_independent' (p_yes RMSE 0.00177 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bigelow_piantadosi_independent, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they think about magnitude in decades, so ranges that line up with decade boundaries ("the 60s", "1 to 20", "the 30s and 40s") are a priori more plausible than arbitrary ranges like 57-73; the boost for decade-aligned intervals is inferred, with the intervals' total prior kept fixed. The single change is this decade-aligned interval prior, which predicts sharp drops in generalisation at decade boundaries (after 61, 64, 67, a target of 69 is endorsed but 71 much less so) rather than the smooth magnitude falloff of the incumbent.

### win_stay_lose_sample_concepts — pruned (experiment1 end of experiment)

**Outcome:** 4.7 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 19

**Hypothesis:** People learn the concept one example at a time, holding a single working guess rather than the full posterior: after the first example they pick a rule or magnitude range by Bayesian concept learning (size principle, a prior weight per rule), and as each later example arrives they keep their current guess with some probability if it still contains that example (win-stay), and otherwise re-pick a guess from the Bayesian posterior given all examples seen so far (lose-sample). The one free mechanism is this stickiness of the running guess, which makes judgments depend on the order of the examples: a broad concept that fits the first example (e.g. "even" after 16) survives later examples it happens to contain (8, 64) instead of being displaced by a tighter rule (powers of 2), so early examples shape generalization more than later ones.

### shared_property_contrast_similarity — pruned (experiment1 end of experiment)

**Outcome:** 120.7 nats behind iter4_candidate5 (6.9× dse)

**Hypothesis:** People do not infer a single hidden concept; they judge a target by its feature-based similarity to the examples as a group. Each number is represented by its mathematical properties (parity, primality, being a square, cube or power, multiples, final digit, digits it contains) and by the magnitude bands it falls in, and a target is endorsed to the extent that it possesses the properties the examples have in common, with each property's weight growing with how many of the examples share it and with how distinctive (rare among 1-100) it is. Endorsement therefore falls off gradedly as the target lacks more of the salient shared properties, with no posterior averaging over concepts and no all-or-none membership.

### response_bias_concept_learner — pruned (experiment1 end of experiment)

**Outcome:** 9.3 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 19

**Hypothesis:** People infer the hidden concept by simple Bayesian concept learning (rules from the Bigelow-Piantadosi inventory sharing one equal prior weight, magnitude intervals weighted by their size, examples sampled from the concept), but they answer with a response bias: a constant shift in the log-odds of their belief toward "yes" (liberal criterion) or toward "no" (conservative criterion), the same on every trial. The single change from the simplest Bayesian model is this criterion shift at the decision stage, which predicts that uncertain targets (near 50% belief) move most while near-certain ones barely move, and that both the endorsement of example-consistent targets and the rejection of implausible ones shift in the same direction.

### uncertain_reliability_noise_concepts — pruned (experiment2 end of experiment)

**Outcome:** 63.5 nats behind graded_exemplar_equal_rule_prior (5.9× dse)

**Hypothesis:** Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, allowing that some examples are stray draws from 1-100 — but they do not know the program's reliability, so instead of assuming one fixed stray rate they hold a belief over how reliable the program is and average over it for each example set. The one change is this uncertainty about the noise rate, which makes strays exchangeable but correlated within a set: one example that breaks a tight rule makes people suspect the program is unreliable on that set, so they discount the remaining examples too and generalise more broadly, while a set fully consistent with a tight rule is taken as strong evidence for it.

### family_pooled_rule_prior — pruned (experiment2 end of experiment)

**Outcome:** 34.4 nats behind graded_exemplar_equal_rule_prior (4.7× dse)

**Hypothesis:** Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, an inferred rate at which examples are stray draws), but their prior over rules is organised by kind of rule rather than set rule by rule: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise. The one change is this family-structured, partially pooled rule prior, which replaces the incumbent's independent weight for every rule and predicts more uniform generalisation across rules within a family.

### any_number_generator_concepts — pruned (experiment1 end of experiment)

**Outcome:** 5.1 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 19

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over its 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they also seriously entertain the possibility that the program has no structure at all and simply produces any number from 1 to 100, giving this "anything goes" hypothesis its own inferred prior weight. The one change is this unstructured-generator hypothesis: because it explains each example only as well as chance, the size principle makes it lose to any fitting rule or range as examples accumulate, so it predicts stray "yes" answers to unrelated targets that are common after one or two examples (or when nothing fits) and vanish after four consistent ones — unlike a constant response lapse.

### digit_sum_concepts — pruned (experiment2 end of experiment)

**Outcome:** 66.1 nats behind graded_exemplar_equal_rule_prior (6.1× dse)

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, uncertain belief in how often examples are stray draws), but they also entertain digit-arithmetic concepts — "numbers whose digits add up to s", one concept for each digit sum from 1 to 18 — which together hold a small a-priori share of the rule prior (about 4%, inferred around that value) split equally among them. The one change is this enlarged concept space, which predicts that sets whose examples share a digit sum (e.g. 23, 41, 50, 14) draw "yes" for other numbers with that digit sum (32, 5, 68) even though no arithmetic rule, final digit or magnitude range explains them.

### exemplar_retrieved_number_concepts — pruned (experiment1 end of experiment)

**Outcome:** 7.1 nats behind iter4_candidate5 (2.2× dse)

**Hypothesis:** People do not reason about which concept is literally consistent with the examples; they hold number concepts in memory as stored sets of member numbers (the familiar rules like even, squares, powers of 2, multiples, final and contained digits, and magnitude ranges), and judge by exemplar similarity on the number line. Each stored concept is retrieved in proportion to how similar the examples are, on average, to its stored members (a concept with few members that lie near every example is retrieved most strongly, so an example close to but not in a concept still partly supports it), and a target is endorsed by its similarity to the nearest stored member of the retrieved concepts, so endorsement falls off gradedly for targets just outside a concept rather than dropping to zero.

### pedagogical_teacher_sampling — pruned (experiment1 end of experiment)

**Outcome:** 7.2 nats behind iter4_candidate5 (2.1× dse)

**Hypothesis:** People reason about the examples as a communicative act: they assume the program's examples were picked by a helpful teacher who chooses numbers that point to the intended concept and away from other plausible concepts, rather than drawn at random from it. So an example that belongs to few other plausible rules or ranges (e.g. 64, which says "powers of 2") counts as strong evidence for the concepts containing it, while an example that belongs to many (e.g. 16 or 50) counts as weak evidence; the strength of this pedagogical assumption is inferred, and with none people reduce to ordinary size-principle Bayesian concept learning over the 45 rules and all intervals of 1-100.

### family_pooled_uncertain_reliability — pruned (experiment2 end of experiment)

**Outcome:** 21.0 nats behind graded_exemplar_equal_rule_prior (3.7× dse)

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, holding an uncertain belief about how reliable the program is (how often an example is a stray draw from 1-100) and averaging over it for each example set — but their prior over rules is organised by kind of rule rather than set rule by rule. The one change, taken from `family_pooled_rule_prior`, is this family-structured, partially pooled rule prior: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit) with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise, which predicts more uniform generalisation across rules within a family than the incumbent's independent rule weights.

### complement_stray_reliability_concepts — pruned (experiment2 end of experiment)

**Outcome:** 99.1 nats behind graded_exemplar_equal_rule_prior (7.7× dse)

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and they are just as unsure how reliable the program is on each set — but they think of a stray example as a mistake, a number that does *not* belong to the concept, rather than a draw that could land anywhere in 1-100. The one change is this stray distribution, drawn from the concept's complement: an example that a broad concept (e.g. "even", "contains a 3") misses is better explained as a stray than under a uniform stray generator, whereas a tight concept gains nothing. So people keep broad rules that miss one example more readily, and examples that a concept already contains are always taken as genuine evidence for it.

### family_pooled_win_stay_lose_sample — pruned (experiment2 end of experiment)

**Outcome:** 30.3 nats behind graded_exemplar_equal_rule_prior (4.3× dse)

**Hypothesis:** Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range picked by size-principle Bayesian concept learning) that they keep with some probability while it still contains each new example and otherwise re-pick from the posterior given all examples so far — so early examples shape generalisation more than later ones. The one change, taken from `family_pooled_rule_prior`, is the prior from which guesses are picked: rules are a priori plausible by kind (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), each rule's weight drawn around a shared family level with an inferred spread, rather than an independently learned weight for every rule, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally likely to be adopted as the working guess.

### without_replacement_example_sampling — pruned (experiment1 end of experiment)

**Outcome:** 4.5 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 19

**Hypothesis:** Refinement of the size-principle Bayesian concept learner (45 rules with their own prior weights plus all magnitude intervals of 1-100, with an uncertain belief about how often an example is a stray draw): people know the program never repeats a number within a set, so they reason as if the genuine examples were drawn from the concept without replacement, not independently. This makes each additional example much stronger evidence for small concepts that the examples nearly exhaust (cubes, high powers, very narrow ranges) while leaving large concepts unchanged, so the disagreement with the best model lies in sets that sit inside a tiny concept: there people should commit to that concept and reject neighbouring or merely broader-rule targets more sharply than independent sampling predicts.

### compressed_number_line_ranges — pruned (experiment2 end of experiment)

**Outcome:** 66.6 nats behind graded_exemplar_equal_rule_prior (6.2× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a prior weight per rule, an uncertain belief in how often an example is a stray draw), but they represent magnitude ranges on a compressed mental number line, on which small numbers take up more room than large ones, and they treat a range's examples as spread evenly along that line, not evenly over the integers. So ranges among small numbers feel mentally wide and give weak evidence, which makes generalisation broad, while equally wide ranges among large numbers feel narrow and tighten quickly. Within a range, its lower members are more typical than its upper ones, so people extend a range further upward than downward. How compressed the line is gets inferred, from logarithmic to linear.

### predictive_surprise_endorsement — pruned (experiment1 end of experiment)

**Outcome:** 51.2 nats behind iter4_candidate5 (5.3× dse)

**Hypothesis:** People answer "would the program produce this number?" by imagining the target as the program's next output and asking how expected it would be, not by checking whether it belongs to the concept they have inferred. They learn the concept by size-principle Bayesian inference over the 45 rules and all intervals of 1-100, unsure how reliable the program is. Then they say yes in proportion to how much more likely the target is as the next example than a random number, so members of tight concepts (64 after "16, 8") are endorsed more strongly than members of broad concepts that are just as certainly consistent (any even number after "16, 8"), and endorsement is graded rather than all-or-none.

### digit_slip_stray_concepts — pruned (experiment1 end of experiment)

**Outcome:** 3.8 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 19

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and they are just as unsure how reliable the program is on each set. The one change is what they think a stray example is: besides a number drawn at random from 1-100, a stray can be a slip, i.e. a member of the concept miswritten (one digit changed, added or dropped, or two digits swapped) so that it lands outside the concept, and how often strays are slips is inferred. So an example that is a slip of a tight concept's member (83 next to 38, 58, 68 under "ends in 8") is easily explained away and the tight concept survives, while an example that no slip of any member could produce counts against the concept as fully as before.

### rule_generated_stray_concepts — pruned (experiment1 end of experiment)

**Outcome:** 8.5 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 19

**Hypothesis:** Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, and are just as unsure how reliable the program is on each set — but they think of a stray example as a number that some *other* mathematical rule produced (a rule picked by how plausible they find rules, then any of its members), rather than a draw that is equally likely to be any number from 1 to 100. The one change is this rule-generated stray distribution: examples that many small rules produce (1, 2, 4, 8, 16, 64, 81, 100) are easily written off as strays, so they constrain the concept less, while "unremarkable" examples that few rules produce (e.g. 58, 87, 94) are hard to dismiss and the concept must cover them.

### iter4_candidate5 — admitted (experiment1 round 4 candidate 5 refine chosen)

**Hypothesis:** Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range) that they keep with some probability while it still contains each new example and otherwise re-pick from the Bayesian posterior given the examples so far, so early examples shape generalisation more than later ones. The one change, taken from `uncertain_reliability_noise_concepts`, is the posterior people re-pick from: they are unsure how reliable the program is and average over that uncertainty, so strays are correlated within a set. One example that breaks a tight rule makes the remaining ones suspect too and pushes re-picked guesses toward broader concepts, while a fully consistent prefix strongly favours the tight rule. This predicts order effects (from the sticky guess) that interact with how consistent the early examples are.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 316.2 nats behind iter4_candidate5 (6.6× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 4.6 nats behind iter4_candidate5; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 19

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### kind_specific_reliability_concepts — pruned (experiment2 end of experiment)

**Outcome:** 88.4 nats behind graded_exemplar_equal_rule_prior (6.8× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a family-organised prior over rules, an uncertain belief about how often an example is a stray draw from 1-100). But their belief about how reliable the program is depends on what kind of program they suppose it is: they expect a program running a mathematical rule to produce strays at a different rate from one producing numbers in a magnitude range, rather than holding one reliability belief whichever kind of concept is true. So a single off-pattern example counts differently against a rule than against a range, which shifts how readily people give up a tight rule for a broad range (or the reverse) when a set contains an outlier.

### weak_sampling_concepts — pruned (experiment2 end of experiment)

**Outcome:** 223.7 nats behind graded_exemplar_equal_rule_prior (9.8× dse)

**Hypothesis:** People treat the examples only as numbers the program is allowed to produce, not as random samples drawn from its concept (weak sampling): a candidate rule or magnitude range gains plausibility merely by containing the examples (and loses it for each example it misses, which they may write off as a stray), but not for being small, so a concept's plausibility rests on its prior — one plausibility per kind of rule (parity, primes, squares and cubes, powers, multiples, final digit, contained digit) shared by the rules of that kind, and a preference for narrower ranges — and does not tighten as more consistent examples arrive. This predicts that adding examples consistent with both a broad and a tight concept (e.g. 16, 8, 64 under "even" and "powers of 2") leaves generalisation about as broad as after the first example, unlike the size principle.

### first_example_anchored_reliability — pruned (experiment2 end of experiment)

**Outcome:** 18.5 nats behind graded_exemplar_equal_rule_prior (3.3× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a prior over rules set by kind of rule, an uncertain belief about how often an example is a stray draw from 1-100), but they treat the first example differently from the rest: the first number shown anchors the concept and is trusted to come from it at its own, inferred reliability, while only the later examples are pooled into the uncertain judgement of how reliable the program is on that set. So a stray-looking number shown first constrains the concept far more (or less) than the same number shown later, and the same set in a different order generalises differently.

### strong_weak_sampling_mixture — pruned (experiment2 end of experiment)

**Outcome:** 153.6 nats behind graded_exemplar_equal_rule_prior (8.2× dse)

**Hypothesis:** Refinement of `family_pooled_uncertain_reliability`: people do the same Bayesian concept learning over the 45 rules (plausibility organised by rule family) and all intervals of 1-100, with an uncertain belief in how often examples are stray draws. The one change is the sampling assumption: people are unsure whether the examples were sampled at random from the concept (so a small concept that fits gains strongly, the size principle) or were just numbers the program happened to produce that are consistent with it (so a concept's size carries no evidence), and they weigh both possibilities, with how much they trust random sampling inferred. This predicts broader, less size-sensitive generalisation than the incumbent, except where the examples fit a tight concept so well that the random-sampling explanation dominates.

### mixed_rule_generated_strays — rejected (experiment2 round 0 candidate 4 refine incumbent family_pooled_uncertain_reliability)

**Outcome:** too slow to fit: the fit of 'mixed_rule_generated_strays' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `family_pooled_uncertain_reliability`: people do the same Bayesian concept learning over the 45 rules (with the family-structured, partially pooled rule prior) and all intervals of 1-100, with the size principle and an uncertain belief about how reliable the program is on each set. The one change, taken from `rule_generated_stray_concepts`, is what people think a stray example is: some strays are random numbers from 1-100, but a share of them (how large is inferred) are numbers that some *other* familiar rule produced, picked in proportion to how plausible people find each rule. So examples that many small rules produce (1, 2, 4, 8, 16, 64, 81, 100) are easily written off as strays and constrain the concept less, while unremarkable examples that few rules produce (e.g. 58, 87, 94) are hard to dismiss and the concept has to cover them.

### iter0_candidate5 — rejected (experiment2 round 0 candidate 5 refine chosen repair 1)

**Outcome:** too slow to fit: the fit of 'iter0_candidate5' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `family_pooled_win_stay_lose_sample`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range, picked by size-principle Bayesian concept learning under a family-organised rule prior) that they keep with some probability while it still contains each new example and otherwise re-pick from the posterior given the examples so far, so early examples shape generalisation more than later ones. The one change, taken from the incumbent `family_pooled_uncertain_reliability`, is the posterior guesses are re-picked from: instead of trusting a fixed 90% of examples, people are unsure how reliable the program is and average over that uncertainty, so strays are correlated within a set — an example that breaks a tight rule makes the rest suspect and pushes re-picked guesses toward broader concepts, while a fully consistent prefix strongly favours the tight rule.

### familiar_number_strays — pruned (experiment2 end of experiment)

**Outcome:** 143.7 nats behind graded_exemplar_equal_rule_prior (7.8× dse)

**Hypothesis:** Refinement of `family_pooled_uncertain_reliability`: people do the same Bayesian concept learning over the 45 rules (with the family-structured, partially pooled rule prior) and all intervals of 1-100, with the size principle and an uncertain belief about how reliable the program is on each set. The one change is what people think a stray example is: some strays are random numbers from 1-100, but a share of them (how large is inferred) are "familiar" numbers, i.e. a member of some well-known rule picked at random, so how likely a number is to turn up as a stray depends on how many small, familiar rules produce it, whichever rules the learner currently favours. Thus examples that many small rules produce (1, 2, 4, 8, 16, 64, 81, 100) are easily written off as strays and constrain the concept less, while unremarkable examples that few rules produce (e.g. 58, 87, 94) are hard to dismiss and the concept has to cover them.

### consistency_attention_similarity — pruned (experiment2 end of experiment)

**Outcome:** 585.1 nats behind graded_exemplar_equal_rule_prior (13.7× dse)

**Hypothesis:** People do not infer a hidden rule; they judge a target by its similarity to the examples, comparing numbers along a few dimensions (magnitude on the number line, parity, final digit, shared written digits, and sharing a common factor), and they focus their attention on whichever dimensions the examples agree on with one another. So a target is endorsed to the extent that it matches the examples on the dimensions along which the examples are mutually similar (e.g. same final digit after 83, 38, 58, 68; close magnitude after 61, 64, 67), with similarity on magnitude falling off exponentially with numerical distance and the sharpness of attention shift inferred.

### noisy_criterion_concept_judgment — pruned (experiment2 end of experiment)

**Outcome:** 34.6 nats behind graded_exemplar_equal_rule_prior (4.8× dse)

**Hypothesis:** People form a graded belief that the target belongs to the hidden concept by simple Bayesian concept learning (rules and magnitude ranges, size principle, a fixed chance that an example is a stray), but they do not report that belief directly: they answer "yes" only when the belief clears a decision criterion, and that criterion varies from trial to trial. How strict the criterion is on average, and how much it varies, are inferred; a criterion that is equally likely to sit anywhere corresponds to probability matching, while a tight criterion produces near all-or-none answers around a fixed level of confidence.

### range_restricted_rule_concepts — pruned (experiment2 end of experiment)

**Outcome:** 18.7 nats behind graded_exemplar_equal_rule_prior (3.1× dse)

**Hypothesis:** People do the same Bayesian concept learning as `first_example_anchored_reliability`: a family-organised prior over mathematical rules, magnitude ranges of 1-100, the size principle, their own trust in the first example, and uncertain reliability for the later ones. On top of that, they also consider *range-restricted rules*, meaning a rule's members inside a magnitude range ("multiples of 4 between 52 and 88", "even numbers below 30"). A priori these restricted rules take a small, stated share of belief (about 5%). The one change is this larger concept space. It predicts that when examples fit a rule and also cluster in magnitude (e.g. 92, 56), people say "yes" to rule members near the examples but not to members far from them, and say "no" to non-members inside the range.

### anchored_rules_individual_plausibility — pruned (experiment2 end of experiment)

**Outcome:** 136.8 nats behind graded_exemplar_equal_rule_prior (7.7× dse)

**Hypothesis:** Refinement of `first_example_anchored_reliability`: people do the same Bayesian concept learning over the 45 rules and all magnitude ranges of 1-100 (size principle, the first example anchoring the concept at its own inferred trust, the later examples pooled into an uncertain belief about how often the program produces strays), but their prior over rules is not flat within a kind of rule. The one change, taken from `family_pooled_uncertain_reliability`, is that each rule has its own plausibility, drawn around its family's level with a stated spread, so that particular rules (e.g. "even" over "odd", "powers of 2" over "powers of 7", "ends in 0" over "ends in 7") can be favoured or disfavoured beyond what their kind implies, which changes which rule an anchored first example points to.

### rule_specific_anchored_reliability — pruned (experiment2 end of experiment)

**Outcome:** 54.1 nats behind graded_exemplar_equal_rule_prior (5.7× dse)

**Hypothesis:** Refinement of `first_example_anchored_reliability`. People do the same Bayesian concept learning over mathematical rules and magnitude ranges of 1-100: the size principle, a prior over rules organised by kind of rule, the first example trusted with its own reliability, and the later examples pooled into an uncertain belief about how often the program produces a stray. The one change is that each rule also has its own plausibility. It is drawn close to the shared level of its family (a small fixed spread, taken from the partially pooled prior of `family_pooled_uncertain_reliability`), so particular rules such as "powers of 5" or "ends in 5" can be adopted somewhat more or less readily than other rules of their kind, instead of all rules in a family being exactly equally plausible.

### familiar_strays_family_level_prior — pruned (experiment2 end of experiment)

**Outcome:** 15.7 nats behind graded_exemplar_equal_rule_prior (3.4× dse)

**Hypothesis:** Refinement of `familiar_number_strays`: people do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, an uncertain belief about how often an example is a stray), and they think of a stray as sometimes a "familiar" number (a member of some well-known rule) rather than always a random number, so examples that many small rules produce (1, 2, 4, 8, 16, 64, 100) are easily written off while unremarkable ones (58, 87, 94) must be covered by the concept. The one change, taken from `first_example_anchored_reliability`, is the rule prior: people judge a rule's plausibility only by its kind (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), so all rules of a kind are equally plausible, with no rule-specific preferences — predicting the same generalisation for, e.g., "contains a 3" and "contains a 7" whenever the examples fit them equally.

### graded_exemplar_concept_retrieval — pruned (experiment2 end of experiment)

**Outcome:** 14.2 nats behind graded_exemplar_equal_rule_prior (3.1× dse)

**Hypothesis:** People do not check whether a number literally belongs to a concept; they hold familiar number concepts (parity, primes, squares and cubes, powers, multiples, final digit, contained digit, and magnitude ranges of 1-100) in memory as stored lists of member numbers, and compare numbers to those stored members by closeness on the number line. A stored concept is retrieved to the extent that each example resembles its members on average (so small concepts whose members sit right at the examples win, and an example just next to a concept still partly supports it), with rules judged plausible only by their kind and an uncertain belief about how often an example is an unrelated stray; a target is then endorsed by how close it is to the nearest stored member of the retrieved concepts, so endorsement falls off gradedly just outside a concept instead of dropping to zero.

### random_looking_strays — pruned (experiment2 end of experiment)

**Outcome:** 19.6 nats behind graded_exemplar_equal_rule_prior (3.4× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a rule prior set by kind of rule, their own trust in the first example and an uncertain belief about how often the later examples are strays), but their idea of what a stray looks like follows subjective randomness (representativeness): a glitch or random draw should look random, so they expect strays to be unremarkable numbers (58, 87, 94) and not familiar, patterned ones (1, 2, 16, 64, 100), with how strong this expectation is inferred. So an unremarkable off-pattern example is easily written off and the majority rule survives, while a familiar off-pattern example is hard to dismiss as noise and counts as genuine evidence against tight rules — the opposite of the familiar-stray account.

### random_looking_strays_2 — pruned (experiment2 end of experiment)

**Outcome:** 20.0 nats behind graded_exemplar_equal_rule_prior (3.4× dse)

**Hypothesis:** People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a prior over rules set by kind of rule, an uncertain belief about how often an example is a stray), but their idea of what a stray looks like follows their intuition about randomness: a stray is a "random-looking", unremarkable number (58, 87, 94), not a special one, so examples that many familiar rules produce (1, 4, 8, 16, 64, 100) are hard to write off as strays and must be covered by the concept, while unremarkable outliers are easily excused. This disagrees most sharply with the best model (which treats familiar numbers as the easiest strays) on sets where all but one example fit a tight rule: when the lone outlier is a familiar number, people should abandon the majority rule for a concept that covers it and endorse the majority rule's other members less than when the outlier is unremarkable.

### graded_familiarity_strays — pruned (experiment2 end of experiment)

**Outcome:** 16.8 nats behind graded_exemplar_equal_rule_prior (3.7× dse)

**Hypothesis:** Refinement of `familiar_strays_family_level_prior`: people do the same Bayesian concept learning over mathematical rules (plausibility set only by kind of rule) and magnitude ranges of 1-100, with the size principle and an uncertain belief about how often an example is a stray. The one change is how they judge which numbers are likely strays. The incumbent mixes random numbers with members of familiar rules, so every number has a fixed floor chance of being a stray. Here the chance is a graded function of how familiar the number is (how readily small, well-known rules produce it), sharpened or flattened to an inferred degree that could even reverse it so that strays are expected to look random. So the ease of writing off an off-pattern example changes smoothly along the whole familiarity scale (64 > 36 > 50 > 58), not in two tiers.

### familiar_strays_first_example_anchor — pruned (experiment2 end of experiment)

**Outcome:** 18.9 nats behind graded_exemplar_equal_rule_prior (3.5× dse)

**Hypothesis:** Refinement of `familiar_strays_family_level_prior`: people do the same Bayesian concept learning over mathematical rules (plausible by kind of rule only) and magnitude ranges of 1-100, with the size principle and strays that are sometimes "familiar" numbers from well-known rules rather than random ones. The one change, taken from `first_example_anchored_reliability`, is that the first example shown anchors the concept: it is trusted to come from the concept at its own, inferred reliability, and only the later examples go into the uncertain judgement of how often the program produces strays. So whether a familiar or unremarkable outlier is written off depends on where it appears: shown first, it constrains the concept at the anchor's own trust, and the same set in another order generalises differently.

### without_replacement_anchored_concepts — pruned (experiment2 end of experiment)

**Outcome:** 26.2 nats behind graded_exemplar_equal_rule_prior (4.0× dse)

**Hypothesis:** Refinement of `first_example_anchored_reliability`: people do the same Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (rule plausibility set by kind of rule, the first example trusted at its own inferred reliability, the later examples pooled into an uncertain belief about how often the program produces a stray). The one change is the sampling assumption behind the size principle: people know the program never repeats a number within a set, so they treat the genuine examples as drawn from the concept without replacement, and each genuine example after the first must be one of the concept's remaining members. This adds no free parameter. It makes sets that nearly exhaust a tiny concept (cubes, powers of 3, 5 or 10, a range only a few numbers wide) much stronger evidence for that concept, while leaving large concepts unchanged, so people commit more sharply to tiny concepts and generalise less beyond them.

### centre_weighted_range_sampling — pruned (experiment2 end of experiment)

**Outcome:** 32.2 nats behind graded_exemplar_equal_rule_prior (3.6× dse)

**Hypothesis:** People do Bayesian concept learning over familiar mathematical rules (plausibility set by kind of rule) and magnitude ranges of 1-100 (size principle, an uncertain belief about how often an example is a stray, strays sometimes familiar numbers), but they believe a program that produces numbers from a magnitude range produces numbers near the middle of its range more often than numbers near its edges. So examples bunched together are taken as the centre of a wider range rather than as filling a narrow one, examples sitting at the extremes of a range count as weak evidence for it, and generalisation from clustered examples extends further beyond them than if people assumed every number in the range were equally likely.

### membership_verification_errors — pruned (experiment2 end of experiment)

**Outcome:** 18.1 nats behind graded_exemplar_equal_rule_prior (3.6× dse)

**Hypothesis:** People retrieve number concepts as in `graded_exemplar_concept_retrieval` (stored member lists compared by number-line closeness, rules plausible only by kind, an uncertain belief in how often an example is a stray), but when they answer they must check whether the target belongs to the retrieved rule, and that check is effortful and error-prone at a rate that depends on how hard the kind of rule is to compute. Easy checks (parity, final digit, contained digit) are nearly error-free, while hard ones (primes, multiples of 7 or 11, powers, squares and cubes) sometimes misfire: people reject true members and accept non-members far from any member, so endorsement under hard rules is diluted toward the middle independently of number-line distance.

### final_digit_binding_errors — pruned (experiment2 end of experiment)

**Outcome:** 15.8 nats behind graded_exemplar_equal_rule_prior (2.9× dse)

**Hypothesis:** People infer the hidden concept by ordinary Bayesian concept learning over mathematical rules (plausibility set by kind of rule) and magnitude ranges of 1-100, with the size principle and an uncertain belief about how often an example is a stray, but when they check a target against a final-digit rule they make feature-binding errors: they register that the target contains the rule's digit and sometimes lose track of its position. So after examples that all end in the same digit (16, 26, 96), a target that has that digit only in the tens place (67, 34 after 63, 23, 43) is sometimes accepted as if it fitted the rule, with how often this happens inferred, while all other rules and ranges are checked exactly.

### graded_exemplar_familiar_strays — pruned (experiment2 end of experiment)

**Outcome:** 12.4 nats behind graded_exemplar_equal_rule_prior (3.4× dse)

**Hypothesis:** Refinement of `graded_exemplar_concept_retrieval`: people hold familiar number concepts (rules by kind, and magnitude ranges of 1-100) as stored member lists, retrieve a concept to the extent that each example resembles its members on the number line, allow for an uncertain rate of unrelated stray examples, and endorse a target by its closeness to the nearest stored member of the retrieved concepts. The one change, taken from `familiar_strays_family_level_prior`, is what a stray looks like: a stray is sometimes a "familiar" number (a member of some well-known rule) rather than always a random number from 1-100, with how often inferred, so examples that many small rules produce (1, 2, 4, 8, 16, 64, 100) are easily written off as strays and constrain retrieval less, while unremarkable examples (58, 87, 94) must be resembled by the retrieved concept.

### graded_exemplar_first_example_anchor — pruned (experiment2 end of experiment)

**Outcome:** 16.5 nats behind graded_exemplar_equal_rule_prior (3.1× dse)

**Hypothesis:** Refinement of `graded_exemplar_concept_retrieval`: people retrieve stored number concepts (rules judged plausible by kind, magnitude ranges of 1-100) by how closely the examples resemble each concept's members on the number line, allowing for stray examples, and endorse a target by its closeness to the nearest member of the retrieved concepts. The one change, taken from `first_example_anchored_reliability`, is that the first example shown anchors retrieval: it is trusted to come from the concept at its own, inferred level of trust, while only the later examples are pooled into the uncertain judgement of how often the program produces strays. So an off-pattern number shown first constrains which concept is retrieved differently from the same number shown later, and the same set in a different order generalises differently.

### recency_anchored_familiar_strays — pruned (experiment2 end of experiment)

**Outcome:** 10.4 nats behind graded_exemplar_equal_rule_prior (2.8× dse)

**Hypothesis:** People learn the hidden concept by Bayesian inference over familiar mathematical rules (all equally plausible) and magnitude ranges, with the size principle and strays that are sometimes familiar numbers, but the most recent example anchors the concept: the last number shown is still vivid and is trusted to come from the concept at its own level of trust, while the earlier, fading examples are only pooled into an uncertain judgement of how often the program produces strays. So an off-pattern number shown last constrains the concept differently from the same number shown earlier, and the same set in a different order generalises differently.

### digit_position_exemplar_retrieval — pruned (experiment2 end of experiment)

**Outcome:** 31.6 nats behind graded_exemplar_equal_rule_prior (3.9× dse)

**Hypothesis:** People hold familiar number concepts (parity, primes, squares and cubes, powers, multiples, final digit, contained digit, and magnitude ranges of 1-100) as stored lists of member numbers, but they compare numbers to those members by their written form, not by their place on the number line: two numbers are similar to the extent that they have the same digit in the same position (tens, ones), with similarity falling off by a constant inferred factor for each digit position that differs. A concept is retrieved to the extent that each example resembles its members in this digit-wise way (allowing for an uncertain rate of unrelated stray examples, rules plausible only by kind), and a target is endorsed by its digit-wise similarity to the nearest member of the retrieved concepts, so a non-member that differs from a member by one digit (84 or 65 under "powers of 2" via 64) is partly endorsed while one that is numerically close but differs in both digits (59 next to 60) is not.

### inner_loop_model — pruned (experiment2 end of experiment)

**Outcome:** 67.5 nats behind graded_exemplar_equal_rule_prior (6.0× dse)

**Hypothesis:** Refinement of `win_stay_lose_sample_concepts`: people learn the concept one example at a time, holding a single working guess (a rule or magnitude range) that they keep with some probability while it still contains each new example and otherwise re-pick from the Bayesian posterior given the examples so far, so early examples shape generalisation more than later ones. The one change, taken from `uncertain_reliability_noise_concepts`, is the posterior people re-pick from: they are unsure how reliable the program is and average over that uncertainty, so strays are correlated within a set. One example that breaks a tight rule makes the remaining ones suspect too and pushes re-picked guesses toward broader concepts, while a fully consistent prefix strongly favours the tight rule. This predicts order effects (from the sticky guess) that interact with how consistent the early examples are.

### consensus_sampling_decision — rejected (experiment3 round 0 candidate 0 lens 8)

**Outcome:** predicts like existing model 'familiar_strays_equal_rule_prior' (p_yes RMSE 0.00050 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of familiar_strays_equal_rule_prior, not a new hypothesis.

**Hypothesis:** People infer the hidden concept by ordinary Bayesian concept learning (equally plausible mathematical rules and magnitude ranges of 1-100, the size principle, an uncertain belief about how often an example is a stray, strays sometimes being familiar numbers), but they answer through a consensus check rather than by reporting their belief: they bring a few candidate concepts to mind, sampled from what they believe, and say "yes" only if every one of them contains the target. How many concepts must agree is inferred. A demanding consensus turns uncertain targets into "no" while barely affecting near-certain ones, a lenient one ("any concept that fits will do") does the reverse, and a single sample is plain probability matching.

### two_source_shared_program_prior — rejected (experiment3 round 0 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'two_source_program_concepts' (p_yes RMSE 0.00179 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of two_source_program_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `two_source_program_concepts`: people think the hidden program may run two sources at once, a main concept (a mathematical rule or a magnitude range of 1-100, learned with the size principle and an uncertain belief about how many examples come from it) and a secondary source behind the off-pattern examples that is either random glitches or a second program they generalise from. The one change is what that second program can be: it is drawn from the same prior over programs people use for the main concept, so it can be a magnitude range as well as a rule, with no new free parameter. So off-pattern examples that cluster in magnitude (e.g. 91 and 95 among 13, 23, 43) are read as a second, range-based source, and people say "yes" to numbers near them, just as off-pattern members of a small rule license that rule's other members.
