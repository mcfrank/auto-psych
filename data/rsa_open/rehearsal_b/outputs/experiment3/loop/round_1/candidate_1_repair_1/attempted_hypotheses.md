# Tried before

66 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### grammatical_exhaustification — pruned (round 1 candidate 1 explore)

**Outcome:** ineligible: more than 4 SEs behind the best on E: 364.2 ± 77.1

**Hypothesis:** People resolve referring expressions using grammatical exhaustification over unasserted alternative descriptions rather than recursive Theory of Mind. When hearing a word, listeners rule out referents proportionally to the informativeness of any truer or more specific features of that referent that the speaker left unmentioned. On uninformative trials, visual attention defaults to contextually unique singletons.

### decoupled_salience_listener — rejected (round 1 candidate 2 explore)

**Outcome:** fitting failed: decoupled_salience_listener: the fit did not finish within 30 minutes

**Hypothesis:** Listeners rely on visual singleton salience during spontaneous visual exploration on uninformative trials, but interpret informative utterances by inverting a speaker who communicates under an unbiased, uniform target prior. In displays where a heard word is true of all objects but some are minimal duplicate targets and others are multi-feature singletons, this mechanism predicts that listeners strongly favor the minimal targets because the speaker would have chosen distinguishing features for the multi-feature singletons, sharply disagreeing with models that project visual singleton biases into communicative intent.

### costly_ambiguity_fallback_speaker — rejected (round 1 candidate 3 explore)

**Outcome:** fitting failed: costly_ambiguity_fallback_speaker: the fit did not finish within 30 minutes

**Hypothesis:** Speakers select referring expressions to maximize informativeness while penalizing utterance ambiguity, falling back to silence when a referent cannot be named. Pragmatic listeners invert this cost-sensitive speaker, attributing informative words to referents that lack more specific alternatives and attributing silence to unnameable or minimally featured referents.

### rsa_l2_unique_singleton_l0 — rejected (round 1 candidate 4 refine incumbent rsa_l2_singleton_feat_color_valence_l0)

**Outcome:** fitting failed: rsa_l2_unique_singleton_l0: the fit did not finish within 30 minutes

**Hypothesis:** RSA depth-2 pragmatic listener refining rsa_l2_singleton_feat_color_valence_l0 by restricting visual singleton salience to unique oddity displays where exactly one object in the scene lacks duplicates. Keeping the same depth-2 recursion (L2 inverting S2 which evaluates L1 over literal L0) and the identical parameter set (alpha, w_singleton, w_distinct, w_features, w_valence, w_color without additions or removals), singleton salience is modified so that multiple co-occurring distinct items do not receive spurious singleton bonuses over duplicate targets. This resolves the severe empirical under-allocation to minimal targets when a message is universal by ensuring perceptual pop-out applies only to genuine odd-one-out items.

### rsa_l2_uncompounded_prior — rejected (round 1 candidate 5 refine incumbent rsa_l2_singleton_feat_color_valence_l0)

**Outcome:** fitting failed: rsa_l2_uncompounded_prior: the fit did not finish within 30 minutes

**Hypothesis:** RSA pragmatic listener at depth 2 with uncompounded perceptual prior, refining rsa_l2_singleton_feat_color_valence_l0 by setting the simulated intermediate listener's prior to uniform rather than compounding the perceptual prior across reasoning tiers. Listeners and speakers reason at depth 2, where intermediate pragmatic reasoning (L1 and S2) tracks communicative informativeness without perceptual bias, and the contextual perceptual prior (combining singleton salience, contextual distinctiveness, feature complexity modulated by valence framing, and color contrast) is integrated once by the outer listener L2 and directly on uninformative prior trials. Differences from rsa_l2_singleton_feat_color_valence_l0: recursion depth remains depth 2 (L2), no parameters are added or removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color), L1 replaces the prior vector parameter with a uniform object prior (wpp=1), and L2 calls L1 without the prior argument.

### competitor_confusion_l2 — rejected (round 1 candidate 6 refine chosen retry 1)

**Outcome:** fitting failed: competitor_confusion_l2: the fit did not finish within 30 minutes

**Hypothesis:** Pragmatic listener reasoning at depth 2 about a competitor-confusion-averse speaker, refining competitor_confusion_speaker by extending pragmatic recursion from depth 1 to depth 2. Listeners interpret utterances by inverting a speaker who selects words to optimize informativeness for a pragmatic listener while actively penalizing expressions shared with visually confusable competitors in the context. Differences from competitor_confusion_speaker: recursion depth increases from depth 1 (L1) to depth 2 (L2), adding a depth-2 listener memo L2 that inverts speaker S2 evaluating pragmatic listener L1 with competitor confusion penalties, with no parameters added or removed.

### decoupled_salience_listener_2 — rejected (round 1 candidate 2 explore repair 1)

**Outcome:** fitting failed: decoupled_salience_listener_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners rely on visual singleton salience during spontaneous visual exploration on uninformative trials, but interpret informative utterances by inverting a speaker who communicates under an unbiased, uniform target prior. In displays where a heard word is true of all objects but some are minimal duplicate targets and others are multi-feature singletons, this mechanism predicts that listeners strongly favor the minimal targets because the speaker would have chosen distinguishing features for the multi-feature singletons, sharply disagreeing with models that project visual singleton biases into communicative intent.

### costly_ambiguity_fallback_speaker_2 — rejected (round 1 candidate 3 explore repair 1)

**Outcome:** fitting failed: costly_ambiguity_fallback_speaker_2: the fit did not finish within 30 minutes

**Hypothesis:** Speakers select referring expressions to maximize informativeness while penalizing utterance ambiguity, falling back to silence when a referent cannot be named. Pragmatic listeners invert this cost-sensitive speaker, attributing informative words to referents that lack more specific alternatives and attributing silence to unnameable or minimally featured referents.

### rsa_l2_unique_singleton_l0_2 — rejected (round 1 candidate 4 refine incumbent rsa_l2_singleton_feat_color_valence_l0 repair 1)

**Outcome:** fitting failed: rsa_l2_unique_singleton_l0_2: the fit did not finish within 30 minutes

**Hypothesis:** RSA depth-2 pragmatic listener refining rsa_l2_singleton_feat_color_valence_l0 by restricting visual singleton salience to unique oddity displays where exactly one object in the scene lacks duplicates. Keeping the same depth-2 recursion (L2 inverting S2 which evaluates L1 over literal L0) and the identical parameter set (alpha, w_singleton, w_distinct, w_features, w_valence, w_color without additions or removals), singleton salience is modified so that multiple co-occurring distinct items do not receive spurious singleton bonuses over duplicate targets. This resolves the severe empirical under-allocation to minimal targets when a message is universal by ensuring perceptual pop-out applies only to genuine odd-one-out items.

### rsa_l2_uncompounded_prior_2 — rejected (round 1 candidate 5 refine incumbent rsa_l2_singleton_feat_color_valence_l0 repair 1)

**Outcome:** fitting failed: rsa_l2_uncompounded_prior_2: the fit did not finish within 30 minutes

**Hypothesis:** RSA pragmatic listener at depth 2 with uncompounded perceptual prior, refining rsa_l2_singleton_feat_color_valence_l0 by setting the simulated intermediate listener's prior to uniform rather than compounding the perceptual prior across reasoning tiers. Listeners and speakers reason at depth 2, where intermediate pragmatic reasoning (L1 and S2) tracks communicative informativeness without perceptual bias, and the contextual perceptual prior (combining singleton salience, contextual distinctiveness, feature complexity modulated by valence framing, and color contrast) is integrated once by the outer listener L2 and directly on uninformative prior trials. Differences from rsa_l2_singleton_feat_color_valence_l0: recursion depth remains depth 2 (L2), no parameters are added or removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color), L1 replaces the prior vector parameter with a uniform object prior (wpp=1), and L2 calls L1 without the prior argument.

### competitor_confusion_l2_2 — rejected (round 1 candidate 6 refine chosen retry 1 repair 1)

**Outcome:** fitting failed: competitor_confusion_l2_2: the fit did not finish within 30 minutes

**Hypothesis:** Pragmatic listener reasoning at depth 2 about a competitor-confusion-averse speaker, refining competitor_confusion_speaker by extending pragmatic recursion from depth 1 to depth 2. Listeners interpret utterances by inverting a speaker who selects words to optimize informativeness for a pragmatic listener while actively penalizing expressions shared with visually confusable competitors in the context. Differences from competitor_confusion_speaker: recursion depth increases from depth 1 (L1) to depth 2 (L2), adding a depth-2 listener memo L2 that inverts speaker S2 evaluating pragmatic listener L1 with competitor confusion penalties, with no parameters added or removed.

### contextual_oddity_listener — pruned (round 2 candidate 1 explore)

**Outcome:** elpd_diff 662.9 > 2.0 x clustered dse 148.5 vs bounded_depth_listener on exp1 (grouped CV)

**Hypothesis:** People interpret referring expressions by integrating communicative informativeness with contextual oddity salience. Before hearing an utterance, visual attention is captured by a solitary odd-one-out that breaks the homogeneous background of duplicate distractors, establishing a default referential prior over objects. Pragmatic listeners invert an informative speaker using this oddity-grounded referent prior, defaulting to uniform expectations when multiple distinct items or no oddities exist.

### graded_truth_listener — rejected (round 2 candidate 3 explore)

**Outcome:** fitting failed: graded_truth_listener: the fit did not finish within 30 minutes

**Hypothesis:** People interpret referring expressions using graded semantic truth values that scale inversely with a referent's feature complexity. When hearing a word, listeners evaluate an object's applicability not as a rigid binary match, but as graded truth that diminishes as the object possesses more competing visual features. On uninformative trials without an informative word, visual attention defaults to contextually unique singletons.

### test_model — rejected (round 2 candidate 4 refine incumbent rsa_l2_singleton_feat_color_valence_l0 retry 1)

**Outcome:** fitting failed: test_model: the fit did not finish within 30 minutes

**Hypothesis:** Test hypothesis.

### grammatical_exhaustification_distinct — pruned (round 2 candidate 6 refine chosen)

**Outcome:** elpd_diff 151.2 > 2.0 x clustered dse 41.8 vs bounded_depth_listener on exp1 (grouped CV)

**Hypothesis:** Refining grammatical_exhaustification by incorporating continuous contextual distinctiveness into the visual attention prior: listeners interpret referring expressions through grammatical exhaustification over unasserted alternative features, ruling out referents proportionally to the informativeness of any more specific descriptions the speaker omitted, while spontaneous visual attention weights referents by combining discrete singleton salience with continuous feature contrast (contextual distinctiveness) relative to display competitors. This directly improves the representation of displays lacking identical duplicates as well as empty singletons in full-duplicate contexts, addressing the empirical discrepancy where humans allocate substantial prior belief to empty singleton objects among identical full-feature duplicates. Differences from grammatical_exhaustification: recursion depth remains unchanged (depth 0, L_exh); parameter w_distinct is added with prior dist.Normal(0.0, 1.0); no parameters are removed; and the visual prior is extended by adding w_distinct times the total contextual feature contrast (summed Hamming distance across all context objects).

### graded_truth_listener_2 — rejected (round 2 candidate 3 explore repair 1)

**Outcome:** fitting failed: graded_truth_listener_2: the fit did not finish within 30 minutes

**Hypothesis:** People interpret referring expressions using graded semantic truth values that scale inversely with a referent's feature complexity. When hearing a word, listeners evaluate an object's applicability not as a rigid binary match, but as graded truth that diminishes as the object possesses more competing visual features. On uninformative trials without an informative word, visual attention defaults to contextually unique singletons.

### feature_parsimony_listener — pruned (round 3 candidate 1 explore)

**Outcome:** ineligible: more than 4 SEs behind the best on E: 478.0 ± 99.8

**Hypothesis:** Listeners interpret referring expressions using a feature parsimony heuristic: among candidate referents satisfying a heard word, listeners choose the object possessing the fewest total visual features, assuming speakers provide minimal descriptions for simple targets rather than leaving distinguishing features unmentioned. Speakers communicate by selecting words that guide this parsimony-biased listener to the intended referent. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### structured_lapse_listener — pruned (round 3 candidate 2 explore)

**Outcome:** ineligible: more than 4 SEs behind the best on E: 440.7 ± 96.9

**Hypothesis:** People separate communicative inference from response generation: listeners infer communicative intent using an unbiased, uniform target prior, and resolve their response through a salience-guided structured lapse that captures attention toward contextually unique visual singletons. On uninformative displays lacking a communicative word, choices default entirely to visual singleton salience; on informative trials, communicative beliefs are mixed at the decision level with the salience lapse rather than multiplied inside Bayes' rule.

### aspect_qud_listener — rejected (round 3 candidate 3 explore)

**Outcome:** fitting failed: aspect_qud_listener: the fit did not finish within 30 minutes

**Hypothesis:** In reference games, listeners interpret utterances by reasoning about a speaker who communicates an informative aspect of the target rather than its complete identity. When selecting an aspect to address as the question under discussion, the speaker chooses among the target's true properties with a preference for distinctive features, and selects an utterance to communicate that aspect to a literal listener. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### bounded_depth_unbiased_prior — pruned (round 3 candidate 4 refine incumbent bounded_depth_listener)

**Outcome:** elpd_diff 574.2 > 2.0 x clustered dse 144.2 vs bounded_depth_listener on exp1 (grouped CV)

**Hypothesis:** Refining bounded_depth_listener by decoupling visual singleton salience from communicative inference: listeners interpret informative referring expressions with bounded reasoning depth under an unbiased, uniform referent prior, confining visual singleton salience strictly to spontaneous visual attention on uninformative displays without an informative word. This prevents spurious singleton bonuses from overriding scalar implicatures on informative trials, eliminating severe over-allocation to multi-feature singleton competitors when words are shared with duplicate minimal targets. Differences from bounded_depth_listener: recursion depth remains unchanged (depth 1, listener calls L1); no parameters are added or removed (retaining alpha and w_singleton); the prior vector parameter is removed from L1 so that the listener chooses referents proportionally to speaker choice probabilities alone; and listener calls L1 without the prior argument, restricting the singleton prior exclusively to prior trials.

### bounded_depth_distinct_listener — rejected (round 3 candidate 5 refine incumbent bounded_depth_listener)

**Outcome:** not novel: its listener and speaker predictions are within RMSE 0.0013 of rsa_l1_singleton_feat_color_valence_l0's (threshold 0.002) over the novelty pool of 1597 scenes. A re-parameterisation of an existing mechanism is not a new hypothesis; change what the listener or speaker computes.

**Hypothesis:** Listeners interpret referring expressions with a strictly bounded reasoning depth (depth 1), inverting an informative speaker who evaluates literal semantics, while visual attention weights candidate referents by combining discrete singleton salience with continuous contextual distinctiveness. This refinement of bounded_depth_listener incorporates contextual feature contrast relative to all display competitors, resolving empirical under-allocation to empty singletons in full-duplicate displays and providing graded perceptual salience across distinct objects. Differences from bounded_depth_listener: recursion depth remains depth 1 (listener calls L1); parameter w_distinct is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); and the object prior is extended by adding w_distinct times the total contextual feature contrast (summed Hamming distance across display objects).

### aspect_qud_listener_2 — rejected (round 3 candidate 3 explore repair 1)

**Outcome:** fitting failed: aspect_qud_listener_2: the fit did not finish within 30 minutes

**Hypothesis:** In reference games, listeners interpret utterances by reasoning about a speaker who communicates an informative aspect of the target rather than its complete identity. When selecting an aspect to address as the question under discussion, the speaker chooses among the target's true properties with a preference for distinctive features, and selects an utterance to communicate that aspect to a literal listener. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### bounded_depth_costly_speaker — rejected (round 3 candidate 5 refine incumbent bounded_depth_listener repair 1)

**Outcome:** fitting failed: bounded_depth_costly_speaker: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions with bounded reasoning depth, inverting an informative speaker who balances literal informativeness against utterance extension costs. Refining bounded_depth_listener by incorporating utterance extension costs (from isolated_graded_costly_l3) into the communicative speaker: speakers penalize words with broad contextual extensions, favoring more specific descriptions when available. This addresses severe empirical over-allocation to multi-feature singleton competitors and under-allocation to minimal targets when words are shared across objects. Differences from bounded_depth_listener: recursion depth remains unchanged (depth 1, listener calls L1); parameter cost_weight is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); and the speaker choice utility subtracts cost_weight times the utterance extension cost (fraction of display objects satisfying each real utterance).

### literal_pragmatic_mixture — rejected (round 4 candidate 1 explore)

**Outcome:** fitting failed: literal_pragmatic_mixture: the fit did not finish within 30 minutes

**Hypothesis:** Listeners vary in their communicative reasoning depth: the population is a mixture of literal listeners who interpret referring expressions according to literal semantic truth, and pragmatic listeners who invert an informative speaker. Communicative speakers choose words by expecting this heterogeneous population of listeners. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### causal_diagnostic_listener — rejected (round 4 candidate 2 explore)

**Outcome:** fitting failed: causal_diagnostic_listener: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions through causal diagnostic inference rather than recursive mental simulation, attributing a heard word to the referent that maximizes causal power relative to the counterfactual production baseline across display alternatives. Speakers communicate by selecting words that reliably distinguish the intended target from contextual competitors. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons and feature contrast.

### communicative_regret_speaker — rejected (round 4 candidate 3 explore)

**Outcome:** fitting failed: communicative_regret_speaker: the fit did not finish within 30 minutes

**Hypothesis:** Speakers choose referring expressions by balancing communicative recovery against the regret of omitting a more informative description available for the target. Listeners invert this regret-averse speaker, attributing underspecified descriptions to minimal referents that lack distinguishing alternatives rather than multi-feature singletons. On displays where a heard word is shared between duplicate minimal targets and multi-feature singletons, this mechanism predicts listeners will strongly favor the minimal targets because the speaker would regret omitting the singletons' distinguishing features, disagreeing sharply with models whose visual singleton biases inflate belief in the complex competitor.

### bounded_depth_parsimony_listener — rejected (round 4 candidate 5 refine incumbent bounded_depth_listener)

**Outcome:** not novel: its listener and speaker predictions are within RMSE 0.0017 of bounded_depth_listener's (threshold 0.002) over the novelty pool of 1597 scenes. A re-parameterisation of an existing mechanism is not a new hypothesis; change what the listener or speaker computes.

**Hypothesis:** Refining bounded_depth_listener by incorporating visual feature parsimony into the referent prior: listeners interpret referring expressions with strictly bounded reasoning depth (depth 1), inverting an informative speaker who evaluates literal semantics, while visual attention weights candidate referents by combining discrete singleton salience with a parsimony penalty on total feature count. This directly penalizes multi-feature competitors in scalar implicature displays and favors simpler or empty referents on prior displays, preventing spurious singleton bonuses from overriding communicative informativeness when words match duplicate minimal targets. Differences from bounded_depth_listener: recursion depth remains unchanged (depth 1, listener calls L1); parameter w_features is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); and the referent prior is extended by adding w_features times the object's feature count (ctx.feature_count).

### rsa_l1_feature_parsimony — rejected (round 4 candidate 6 refine chosen)

**Outcome:** fitting failed: rsa_l1_feature_parsimony: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions by combining depth-1 pragmatic communicative inference with a feature parsimony penalty, favoring candidate referents with the fewest total visual features because speakers provide minimal descriptions for simple targets rather than leaving distinguishing features unmentioned. Speakers communicate by selecting expressions that guide this parsimony-biased listener to the target referent, while spontaneous visual attention on uninformative displays without a word follows contextual visual salience. Refining rsa_l1_singleton_feat_color_valence_l0 by incorporating the feature parsimony mechanism from feature_parsimony_listener: when interpreting a heard word, listeners discount referents proportionally to their total visual feature count, directly counteracting spurious singleton bonuses and resolving the severe empirical over-allocation to multi-feature singleton competitors and under-allocation to minimal duplicate targets in universal scenes. Differences from rsa_l1_singleton_feat_color_valence_l0: recursion depth remains depth 1 (listener calls L1); parameter beta is added with prior dist.LogNormal(0.0, 1.0); no parameters are removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color); in L1 the listener choice weight multiplies speaker probability by exp(-beta * vec(feature_count, r)); and listener passes ctx.feature_count into L1.

### literal_pragmatic_mixture_2 — rejected (round 4 candidate 1 explore repair 1)

**Outcome:** fitting failed: literal_pragmatic_mixture_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners vary in their communicative reasoning depth: the population is a mixture of literal listeners who interpret referring expressions according to literal semantic truth, and pragmatic listeners who invert an informative speaker. Communicative speakers choose words by expecting this heterogeneous population of listeners. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### causal_diagnostic_listener_2 — rejected (round 4 candidate 2 explore repair 1)

**Outcome:** fitting failed: causal_diagnostic_listener_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions through causal diagnostic inference rather than recursive mental simulation, attributing a heard word to the referent that maximizes causal power relative to the counterfactual production baseline across display alternatives. Speakers communicate by selecting words that reliably distinguish the intended target from contextual competitors. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons and feature contrast.

### communicative_regret_speaker_2 — rejected (round 4 candidate 3 explore repair 1)

**Outcome:** fitting failed: communicative_regret_speaker_2: the fit did not finish within 30 minutes

**Hypothesis:** Speakers choose referring expressions by balancing communicative recovery against the regret of omitting a more informative description available for the target. Listeners invert this regret-averse speaker, attributing underspecified descriptions to minimal referents that lack distinguishing alternatives rather than multi-feature singletons. On displays where a heard word is shared between duplicate minimal targets and multi-feature singletons, this mechanism predicts listeners will strongly favor the minimal targets because the speaker would regret omitting the singletons' distinguishing features, disagreeing sharply with models whose visual singleton biases inflate belief in the complex competitor.

### bounded_depth_parsimony_listener_2 — rejected (round 4 candidate 5 refine incumbent bounded_depth_listener repair 1)

**Outcome:** not novel: its listener and speaker predictions are within RMSE 0.0017 of bounded_depth_listener's (threshold 0.002) over the novelty pool of 1597 scenes. A re-parameterisation of an existing mechanism is not a new hypothesis; change what the listener or speaker computes.

**Hypothesis:** Refining bounded_depth_listener by incorporating visual feature parsimony into the referent prior: listeners interpret referring expressions with strictly bounded reasoning depth (depth 1), inverting an informative speaker who evaluates literal semantics, while visual attention weights candidate referents by combining discrete singleton salience with a parsimony penalty on total feature count. This directly penalizes multi-feature competitors in scalar implicature displays and favors simpler or empty referents on prior displays, preventing spurious singleton bonuses from overriding communicative informativeness when words match duplicate minimal targets. Differences from bounded_depth_listener: recursion depth remains unchanged (depth 1, listener calls L1); parameter w_features is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); and the referent prior is extended by adding w_features times the object's feature count (ctx.feature_count).

### rsa_l1_feature_parsimony_2 — rejected (round 4 candidate 6 refine chosen repair 1)

**Outcome:** fitting failed: rsa_l1_feature_parsimony_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions by combining depth-1 pragmatic communicative inference with a feature parsimony penalty, favoring candidate referents with the fewest total visual features because speakers provide minimal descriptions for simple targets rather than leaving distinguishing features unmentioned. Speakers communicate by selecting expressions that guide this parsimony-biased listener to the target referent, while spontaneous visual attention on uninformative displays without a word follows contextual visual salience. Refining rsa_l1_singleton_feat_color_valence_l0 by incorporating the feature parsimony mechanism from feature_parsimony_listener: when interpreting a heard word, listeners discount referents proportionally to their total visual feature count, directly counteracting spurious singleton bonuses and resolving the severe empirical over-allocation to multi-feature singleton competitors and under-allocation to minimal duplicate targets in universal scenes. Differences from rsa_l1_singleton_feat_color_valence_l0: recursion depth remains depth 1 (listener calls L1); parameter beta is added with prior dist.LogNormal(0.0, 1.0); no parameters are removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color); in L1 the listener choice weight multiplies speaker probability by exp(-beta * vec(feature_count, r)); and listener passes ctx.feature_count into L1.

### isolated_graded_costly_l3 — pruned (seed)

**Outcome:** ineligible: more than 4 SEs behind the best on E: 419.6 ± 100.3

**Hypothesis:** Pragmatic listener at depth 3 with perceptual isolation, graded semantics, and utterance extension costs. Refines isolated_graded_costly_l2 by increasing the recursion depth from depth 2 to depth 3. Listeners reason at depth 3, inverting a speaker who simulates a depth-2 pragmatic listener. Visual pop-out is determined by an object's perceptual isolation (minimum Hamming distance to any other context object), semantic applicability diminishes with additional competing features (graded truth), and speakers evaluate communicative utility trade-offs with utterance extension costs proportional to the fraction of display objects sharing each feature. On uninformative prior trials, choice follows the perceptual isolation prior. A lapse parameter mixes in uniform guessing.

### oddity_heuristic_listener — pruned (seed)

**Outcome:** ineligible: more than 4 SEs behind the best on E: 1773.0 ± 255.1

**Hypothesis:** Perceptual oddity heuristic listener. Listeners interpret referring expressions using a perceptual oddity heuristic rather than recursive Theory of Mind. When hearing a word that applies to multiple referents, listeners select the contextually unique singleton that lacks identical duplicates in the visual display. On uninformative trials without an informative word, listeners spontaneously choose the unique singleton, defaulting to uniform choice when all items are distinct or no oddity contrast exists. A lapse parameter captures random clicking.

### rsa_l1 — pruned (seed)

**Outcome:** ineligible: more than 4 SEs behind the best on E: 620.5 ± 152.0

**Hypothesis:** Vanilla RSA pragmatic listener (depth 1), uniform prior over objects. The listener reasons about a speaker who chooses a true word with probability proportional to exp(alpha * log L0(object | word)), where L0 is the literal listener; the listener inverts that speaker with a uniform prior over objects (Frank & Goodman, 2012). With no informative word the choice is uniform. A lapse parameter mixes in uniform guessing.

### competitor_confusion_speaker — pruned (seed)

**Outcome:** elpd_diff 637.0 > 2.0 x clustered dse 131.6 vs bounded_depth_listener on exp1 (grouped CV)

**Hypothesis:** Pragmatic listener reasoning about a competitor-confusion-averse speaker. Speakers in reference games actively avoid referring expressions that are shared with visually similar or confusable distractors in the context. Candidate words are penalized proportionally to their feature overlap with alternative referents that also satisfy the word. Pragmatic listeners invert this confusion-averse speaker, expecting words to target referents whose competitors are visually distinct rather than near-identical clones.

### rsa_l2 — pruned (seed)

**Outcome:** elpd_diff 1034.8 > 2.0 x clustered dse 144.4 vs bounded_depth_listener on exp1 (grouped CV)

**Hypothesis:** RSA pragmatic listener at depth 2, uniform prior over objects. One more level of recursion than the vanilla model: a speaker S2 chooses words by how well the depth-1 pragmatic listener L1 would recover the object (exp(alpha * log L1)), and the listener L2 inverts S2. Depth-2 reasoning is what the pragmods "complex" game needs to single out the object described by the twice-implicated word. One rationality alpha at every level; uniform choice with no informative word; a lapse parameter mixes in uniform guessing.

### sortal_partition_listener — rejected (round 1 candidate 1 explore)

**Outcome:** fitting failed: sortal_partition_listener: the fit did not finish within 30 minutes

**Hypothesis:** People interpret and produce referring expressions over sortal equivalence classes of objects rather than individual visual tokens. Communicative speakers choose words expecting listeners to identify referent types, while pragmatic listeners partition hypothesis weight across duplicate tokens within each sortal class, preventing identical distractors from spuriously diluting unique minimal referents. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### type_level_reference — rejected (round 1 candidate 2 explore)

**Outcome:** fitting failed: type_level_reference: the fit did not finish within 30 minutes

**Hypothesis:** People interpret and produce referring expressions by reasoning over perceptual equivalence classes (referential types) rather than raw object tokens: identical visual objects in a scene are perceptually grouped into a single type, and communicative speakers choose utterances to distinguish the target's type from alternative types. Pragmatic listeners invert this type-discriminating speaker while maintaining spontaneous visual attention on contextually unique singletons. In displays where a heard word is universally true of all objects and distractors are identical duplicates, this mechanism predicts listeners will strongly attribute the word to the visual singleton, disagreeing sharply with standard token-based models that impose spurious M-implicature penalties on singletons.

### contrast_deficit_speaker — rejected (round 1 candidate 3 explore)

**Outcome:** fitting failed: contrast_deficit_speaker: the fit did not finish within 30 minutes

**Hypothesis:** Speakers choose referring expressions to maximize contextual contrast, penalizing descriptions that leave avoidable ambiguity when a more distinguishing alternative is available for the target. When an object has no distinguishing features, speakers fall back to general descriptions without penalty. Pragmatic listeners invert this contrast-sensitive speaker, attributing underspecified words to referents that lack more specific alternatives.

### rsa_l1_singleton_feat_color_valence_l0_2 — rejected (round 1 candidate 4 refine incumbent rsa_l1_singleton_feat_color_valence_l0)

**Outcome:** hypothesis.md is word for word rsa_l1_singleton_feat_color_valence_l0's hypothesis, or part of it. State this candidate's own mechanism, including what it changes from the theory it refines.

**Hypothesis:** RSA pragmatic listener at depth 1 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0.

### rsa_l1_shared_perceptual_prior — rejected (round 1 candidate 5 refine incumbent rsa_l1_singleton_feat_color_valence_l0)

**Outcome:** fitting failed: rsa_l1_shared_perceptual_prior: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions by inverting an informative speaker who models a foundational listener sharing common ground of perceptual salience, so that visual salience directly informs communicative expectations rather than being applied solely as a downstream selection bias. When hearing a word that applies universally across a scene, the speaker's expectation that the listener will naturally attend to visually unique singletons prevents a spurious M-implicature penalty against the singleton, correctly aligning predicted ratings with human responses on duplicate displays. Refining rsa_l1_singleton_feat_color_valence_l0 by making the contextual perceptual prior common knowledge in the foundational simulated listener L0: recursion depth remains depth 1 (listener calls L1); no parameters are added or removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color); L0 adds the prior vector to its signature and weights referents by the product of the prior vector and semantic truth; and in L1 the speaker evaluates L0 with the prior argument.

### bounded_depth_graded_truth — rejected (round 1 candidate 6 refine chosen)

**Outcome:** fitting failed: bounded_depth_graded_truth: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions with strictly bounded reasoning depth (depth 1) by inverting an informative communicative speaker, but evaluate semantic applicability as graded truth that diminishes with each additional competing visual feature. Refining bounded_depth_listener by incorporating graded semantic applicability from isolated_graded_costly_l3 into the literal semantics: candidate referents with additional unmentioned visual features possess lower baseline applicability, enabling listeners to strongly favor minimal referents when words are universally shared while maintaining visual singleton salience on uninformative displays. Differences from bounded_depth_listener: recursion depth remains depth 1 (listener calls L1); parameter gamma is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); literal lexicon applicability is scaled by exp(-gamma * (feature_count - 1)) for real utterances; and the listener passes this graded lexicon into L1 on informative trials.

### sortal_partition_listener_2 — rejected (round 1 candidate 1 explore repair 1)

**Outcome:** fitting failed: sortal_partition_listener_2: the fit did not finish within 30 minutes

**Hypothesis:** People interpret and produce referring expressions over sortal equivalence classes of objects rather than individual visual tokens. Communicative speakers choose words expecting listeners to identify referent types, while pragmatic listeners partition hypothesis weight across duplicate tokens within each sortal class, preventing identical distractors from spuriously diluting unique minimal referents. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### type_level_reference_2 — rejected (round 1 candidate 2 explore repair 1)

**Outcome:** fitting failed: type_level_reference_2: the fit did not finish within 30 minutes

**Hypothesis:** People interpret and produce referring expressions by reasoning over perceptual equivalence classes (referential types) rather than raw object tokens: identical visual objects in a scene are perceptually grouped into a single type, and communicative speakers choose utterances to distinguish the target's type from alternative types. Pragmatic listeners invert this type-discriminating speaker while maintaining spontaneous visual attention on contextually unique singletons. In displays where a heard word is universally true of all objects and distractors are identical duplicates, this mechanism predicts listeners will strongly attribute the word to the visual singleton, disagreeing sharply with standard token-based models that impose spurious M-implicature penalties on singletons.

### contrast_deficit_speaker_2 — rejected (round 1 candidate 3 explore repair 1)

**Outcome:** fitting failed: contrast_deficit_speaker_2: the fit did not finish within 30 minutes

**Hypothesis:** Speakers choose referring expressions to maximize contextual contrast, penalizing descriptions that leave avoidable ambiguity when a more distinguishing alternative is available for the target. When an object has no distinguishing features, speakers fall back to general descriptions without penalty. Pragmatic listeners invert this contrast-sensitive speaker, attributing underspecified words to referents that lack more specific alternatives.

### rsa_l1_singleton_feat_color_valence_l0_3 — rejected (round 1 candidate 4 refine incumbent rsa_l1_singleton_feat_color_valence_l0 repair 1)

**Outcome:** hypothesis.md is word for word rsa_l1_singleton_feat_color_valence_l0's hypothesis, or part of it. State this candidate's own mechanism, including what it changes from the theory it refines.

**Hypothesis:** RSA pragmatic listener at depth 1 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0.

### rsa_l1_shared_perceptual_prior_2 — rejected (round 1 candidate 5 refine incumbent rsa_l1_singleton_feat_color_valence_l0 repair 1)

**Outcome:** fitting failed: rsa_l1_shared_perceptual_prior_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions by inverting an informative speaker who models a foundational listener sharing common ground of perceptual salience, so that visual salience directly informs communicative expectations rather than being applied solely as a downstream selection bias. When hearing a word that applies universally across a scene, the speaker's expectation that the listener will naturally attend to visually unique singletons prevents a spurious M-implicature penalty against the singleton, correctly aligning predicted ratings with human responses on duplicate displays. Refining rsa_l1_singleton_feat_color_valence_l0 by making the contextual perceptual prior common knowledge in the foundational simulated listener L0: recursion depth remains depth 1 (listener calls L1); no parameters are added or removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color); L0 adds the prior vector to its signature and weights referents by the product of the prior vector and semantic truth; and in L1 the speaker evaluates L0 with the prior argument.

### bounded_depth_graded_truth_2 — rejected (round 1 candidate 6 refine chosen repair 1)

**Outcome:** fitting failed: bounded_depth_graded_truth_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions with strictly bounded reasoning depth (depth 1) by inverting an informative communicative speaker, but evaluate semantic applicability as graded truth that diminishes with each additional competing visual feature. Refining bounded_depth_listener by incorporating graded semantic applicability from isolated_graded_costly_l3 into the literal semantics: candidate referents with additional unmentioned visual features possess lower baseline applicability, enabling listeners to strongly favor minimal referents when words are universally shared while maintaining visual singleton salience on uninformative displays. Differences from bounded_depth_listener: recursion depth remains depth 1 (listener calls L1); parameter gamma is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); literal lexicon applicability is scaled by exp(-gamma * (feature_count - 1)) for real utterances; and the listener passes this graded lexicon into L1 on informative trials.

### topic_discriminability_listener — pruned (round 2 candidate 1 explore retry 1)

**Outcome:** ineligible: more than 4 SEs behind the best on exp1: 1006.2 ± 146.0

**Hypothesis:** Listeners interpret referring expressions by modeling what speakers are likely to talk about before an utterance is heard: a referential prior over objects grounded in communicative discriminability. Before hearing a word, listeners expect speakers to choose topics that can be successfully distinguished in context, assigning prior probability to each candidate referent according to the maximal literal communicative recovery of its features. On uninformative displays or universal utterances, this topic prior directs belief toward uniquely discriminable objects without relying on spurious visual singleton heuristics that distort scalar implicatures.

### bounded_sample_listener — rejected (round 2 candidate 2 explore)

**Outcome:** fitting failed: bounded_sample_listener: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions through sample-based pragmatic inference rather than exact Bayesian normalization, where a bounded mental sample budget sharpens belief onto the most communicatively informative candidate. Communicative speakers choose expressions by anticipating this sample-bounded listener. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### feature_uncertainty_listener — rejected (round 2 candidate 3 explore)

**Outcome:** fitting failed: feature_uncertainty_listener: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions under referential feature uncertainty, being unsure which visual feature of a candidate referent the speaker intended to pick out. When evaluating literal semantics, listeners weight an object's compatibility by the probability that the speaker chose the heard feature from that object's feature bundle, naturally favoring minimal referents whose features are exhaustively described over cluttered multi-feature competitors. Communicative speakers select expressions expecting this feature-attribution inference, while spontaneous visual attention on uninformative displays defaults to contextually unique singletons.

### rsa_l1_communicative_parsimony — rejected (round 2 candidate 4 refine incumbent rsa_l1_singleton_feat_color_valence_l0 retry 1)

**Outcome:** fitting failed: rsa_l1_communicative_parsimony: the fit did not finish within 30 minutes

**Hypothesis:** RSA pragmatic listener at depth 1 with communicative feature parsimony, singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, and visual color contrast. Refining rsa_l1_singleton_feat_color_valence_l0 by incorporating the communicative feature parsimony component from bounded_depth_feature_parsimony: on informative displays, listeners interpret referring expressions by expecting cooperative speakers to provide minimal descriptions for simple referents, penalizing candidate referents that possess redundant unmentioned visual features while maintaining contextual visual salience on uninformative displays. This parsimony penalty directly curbs spurious singleton prior preservation for multi-feature scalar competitors and widens pragmatic contrast in large displays, addressing empirical under-prediction on universal word displays and over-allocation to redundant competitors without over-polarizing scalar implicatures. Differences from rsa_l1_singleton_feat_color_valence_l0: recursion depth remains depth 1 (listener calls L1); parameter beta is added with prior dist.LogNormal(0.0, 1.0); no parameters are removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color); L1 adds beta and feature_count to its signature and scales referent choice weights by exp(-beta * vec(feature_count, r)); and listener passes beta and ctx.feature_count into L1 on informative trials.

### test_benchmark — rejected (round 2 candidate 5 refine incumbent rsa_l1_singleton_feat_color_valence_l0)

**Outcome:** fitting failed: test_benchmark: the fit did not finish within 30 minutes

**Hypothesis:** Temporary benchmark hypothesis.

### bounded_depth_shared_salience — rejected (round 2 candidate 6 refine chosen)

**Outcome:** fitting failed: bounded_depth_shared_salience: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions by inverting an informative speaker who models a foundational listener with shared visual singleton salience, so that visual distinctiveness directly informs communicative expectations rather than acting solely as an unmodeled downstream selection bias. When a heard word is shared universally across a scene with duplicate distractors, the speaker anticipates that visual pop-out guides the listener toward the unique singleton, eliminating the spurious M-implicature penalty that causes standard models to under-rate singletons. Refining bounded_depth_listener: recursion depth remains depth 1 (listener calls L1); no parameters are added or removed (retaining alpha and w_singleton); L0 adds prior to its signature and weights candidate referents by the product of semantic truth and the singleton salience prior; and in L1 the speaker evaluates L0 with the prior argument.

### bounded_sample_listener_2 — rejected (round 2 candidate 2 explore repair 1)

**Outcome:** fitting failed: bounded_sample_listener_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions through sample-based pragmatic inference rather than exact Bayesian normalization, where a bounded mental sample budget sharpens belief onto the most communicatively informative candidate. Communicative speakers choose expressions by anticipating this sample-bounded listener. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### rsa_l1_communicative_parsimony_2 — rejected (round 2 candidate 4 refine incumbent rsa_l1_singleton_feat_color_valence_l0 retry 1 repair 1)

**Outcome:** fitting failed: rsa_l1_communicative_parsimony_2: the fit did not finish within 30 minutes

**Hypothesis:** RSA pragmatic listener at depth 1 with communicative feature parsimony, singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, and visual color contrast. Refining rsa_l1_singleton_feat_color_valence_l0 by incorporating the communicative feature parsimony component from bounded_depth_feature_parsimony: on informative displays, listeners interpret referring expressions by expecting cooperative speakers to provide minimal descriptions for simple referents, penalizing candidate referents that possess redundant unmentioned visual features while maintaining contextual visual salience on uninformative displays. This parsimony penalty directly curbs spurious singleton prior preservation for multi-feature scalar competitors and widens pragmatic contrast in large displays, addressing empirical under-prediction on universal word displays and over-allocation to redundant competitors without over-polarizing scalar implicatures. Differences from rsa_l1_singleton_feat_color_valence_l0: recursion depth remains depth 1 (listener calls L1); parameter beta is added with prior dist.LogNormal(0.0, 1.0); no parameters are removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color); L1 adds beta and feature_count to its signature and scales referent choice weights by exp(-beta * vec(feature_count, r)); and listener passes beta and ctx.feature_count into L1 on informative trials.

### rsa_l1_gestalt_grouping — rejected (round 2 candidate 5 refine incumbent rsa_l1_singleton_feat_color_valence_l0 repair 1)

**Outcome:** fitting failed: rsa_l1_gestalt_grouping: the fit did not finish within 30 minutes

**Hypothesis:** RSA pragmatic listener at depth 1 with Gestalt perceptual grouping over identical objects, singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, and visual color contrast. Refining rsa_l1_singleton_feat_color_valence_l0 by incorporating Gestalt perceptual grouping into the prior over referents: human observers group identical display objects into a single perceptual type, so that duplicate tokens partition the prior probability mass of their visual equivalence class inversely proportional to their duplicate count, preventing identical distractors from artificially diluting unique visual singletons. This directly increases predicted ratings for probed visual singletons under universal words and widens pragmatic contrast in large asymmetric scenes. Differences from rsa_l1_singleton_feat_color_valence_l0: recursion depth remains depth 1 (listener calls L1); no parameters are added or removed (retaining alpha, w_singleton, w_distinct, w_features, w_valence, and w_color); and in listener the prior logits decrement each object's utility by the logarithm of its duplicate count (subtracting log(duplicate_count) inside softmax_prior).

### bounded_depth_shared_salience_2 — rejected (round 2 candidate 6 refine chosen repair 1)

**Outcome:** fitting failed: bounded_depth_shared_salience_2: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions by inverting an informative speaker who models a foundational listener with shared visual singleton salience, so that visual distinctiveness directly informs communicative expectations rather than acting solely as an unmodeled downstream selection bias. When a heard word is shared universally across a scene with duplicate distractors, the speaker anticipates that visual pop-out guides the listener toward the unique singleton, eliminating the spurious M-implicature penalty that causes standard models to under-rate singletons. Refining bounded_depth_listener: recursion depth remains depth 1 (listener calls L1); no parameters are added or removed (retaining alpha and w_singleton); L0 adds prior to its signature and weights candidate referents by the product of semantic truth and the singleton salience prior; and in L1 the speaker evaluates L0 with the prior argument.

### semantic_coverage_listener — rejected (round 1 candidate 1 explore)

**Outcome:** fitting failed: semantic_coverage_listener: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions through semantic information coverage, weighting candidate referents by the proportion of the object's total contextual feature information explained by the heard word. When a speaker uses a shared or underspecified description, referents possessing unmentioned distinguishing features are sharply discounted because cooperative speakers do not leave highly informative properties unsaid. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

### description_exhaustivity_listener — rejected (round 1 candidate 2 explore)

**Outcome:** fitting failed: description_exhaustivity_listener: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions through description exhaustivity, assuming that cooperative speakers use single words to exhaustively describe minimal referents and applying a discrete non-exhaustivity penalty to any candidate referent possessing unmentioned features regardless of how many additional features it has. Communicative speakers choose expressions by anticipating this exhaustivity-sensitive listener, while spontaneous visual attention on uninformative displays defaults to contextually unique singletons. On multi-level scalar hierarchy displays containing minimal, intermediate, and maximal candidates, this mechanism predicts that listeners reject intermediate competitors almost as completely as maximal competitors because both violate description exhaustivity, disagreeing sharply with feature-uncertainty models that scale smoothly with feature count and over-predict intermediate choices.

### contrastive_alternatives_speaker — rejected (round 1 candidate 3 explore)

**Outcome:** fitting failed: contrastive_alternatives_speaker: the fit did not finish within 30 minutes

**Hypothesis:** Communicative speakers select referring expressions by penalizing avoidable ambiguity: candidate words are discounted proportionally to the excess competitors they fail to eliminate compared to the most specific alternative available for that referent. When an object cannot be uniquely named, the speaker communicates without penalty using the most informative description available. Pragmatic listeners invert this ambiguity-sensitive speaker, attributing underspecified words to referents that lack more specific alternatives, while spontaneous visual attention on uninformative displays defaults to contextually unique singletons.

### feature_uncertainty_distinct — rejected (round 1 candidate 4 refine incumbent feature_uncertainty_listener_2)

**Outcome:** fitting failed: feature_uncertainty_distinct: the fit did not finish within 30 minutes

**Hypothesis:** Refining feature_uncertainty_listener_2 by incorporating continuous contextual distinctiveness from rsa_l1_singleton_feat_color_valence_l0 into the visual attention prior: listeners interpret referring expressions under referential feature uncertainty over literal semantics, but visual attention weights candidate referents by combining discrete singleton salience with continuous feature contrast relative to display competitors. This directly accounts for the contextual uniqueness of unmentioned features in scalar competitor displays and scales singleton salience in large competitor sets, addressing empirical discrepancies in competitor selection and singleton salience. Communicative speakers select expressions anticipating this feature-attribution inference, while spontaneous visual attention on uninformative displays follows the combined perceptual prior. Differences from feature_uncertainty_listener_2: recursion depth remains depth 1 (listener calls L1); parameter w_distinct is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); the object prior is extended by adding w_distinct times contextual distinctiveness (summed pairwise Hamming distance across context objects for real features); memo definitions (L0 and L1) remain unchanged; and listener passes the extended prior into L1.

### feature_uncertainty_distinct_listener — rejected (round 1 candidate 5 refine incumbent feature_uncertainty_listener_2)

**Outcome:** fitting failed: feature_uncertainty_distinct_listener: the fit did not finish within 30 minutes

**Hypothesis:** Listeners interpret referring expressions under referential feature uncertainty, weighting literal semantic compatibility inversely by a referent's total visual feature count, while communicative speakers select expressions expecting this feature-attribution inference. Spontaneous visual attention weights candidate referents by combining discrete singleton salience with continuous contextual distinctiveness (feature contrast relative to all display competitors), addressing empirical failures to account for contextually unique unmentioned features and under-prediction of singleton salience in redundant or large competitor sets. Differences from feature_uncertainty_listener_2: recursion depth remains unchanged (depth 1, listener calls L1); parameter w_distinct is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); and the visual prior is extended by adding w_distinct times the continuous contextual distinctiveness (summed Hamming feature distance across display objects).

### distinct_feature_parsimony — rejected (round 1 candidate 6 refine chosen)

**Outcome:** fitting failed: distinct_feature_parsimony: the fit did not finish within 30 minutes

**Hypothesis:** Refining bounded_depth_feature_parsimony by incorporating continuous contextual distinctiveness into the visual salience prior: candidate referents are weighted by combining discrete visual singleton salience with continuous feature contrast (contextual distinctiveness) relative to all context competitors. On informative trials, listeners interpret referring expressions with bounded depth (depth 1) by inverting an informative speaker over literal semantics, penalizing candidate referents possessing redundant visual features through communicative feature parsimony while contextual distinctiveness captures the salience of unmentioned contextually unique features. On uninformative displays, visual attention defaults jointly to discrete singleton salience and continuous contextual distinctiveness. Differences from bounded_depth_feature_parsimony: recursion depth remains depth 1 (listener calls L1); parameter w_distinct is added with prior dist.Normal(0.0, 1.0); no parameters are removed (retaining alpha, w_singleton, and beta); the referent prior is extended by adding w_distinct times the total contextual feature contrast (summed Hamming distance across display objects); and L1 memo signature and listener choice weights remain unchanged.
