# Tried before

13 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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
