# Tried before

22 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

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

### graded_truth_listener — rejected (round 2 candidate 3 explore)

**Outcome:** fitting failed: graded_truth_listener: the fit did not finish within 30 minutes

**Hypothesis:** People interpret referring expressions using graded semantic truth values that scale inversely with a referent's feature complexity. When hearing a word, listeners evaluate an object's applicability not as a rigid binary match, but as graded truth that diminishes as the object possesses more competing visual features. On uninformative trials without an informative word, visual attention defaults to contextually unique singletons.

### test_model — rejected (round 2 candidate 4 refine incumbent rsa_l2_singleton_feat_color_valence_l0 retry 1)

**Outcome:** fitting failed: test_model: the fit did not finish within 30 minutes

**Hypothesis:** Test hypothesis.

### graded_truth_listener_2 — rejected (round 2 candidate 3 explore repair 1)

**Outcome:** fitting failed: graded_truth_listener_2: the fit did not finish within 30 minutes

**Hypothesis:** People interpret referring expressions using graded semantic truth values that scale inversely with a referent's feature complexity. When hearing a word, listeners evaluate an object's applicability not as a rigid binary match, but as graded truth that diminishes as the object possesses more competing visual features. On uninformative trials without an informative word, visual attention defaults to contextually unique singletons.

### aspect_qud_listener — rejected (round 3 candidate 3 explore)

**Outcome:** fitting failed: aspect_qud_listener: the fit did not finish within 30 minutes

**Hypothesis:** In reference games, listeners interpret utterances by reasoning about a speaker who communicates an informative aspect of the target rather than its complete identity. When selecting an aspect to address as the question under discussion, the speaker chooses among the target's true properties with a preference for distinctive features, and selects an utterance to communicate that aspect to a literal listener. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

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
