We refine empty_baseline_salience_l2 by incorporating contextual feature prevalence from feature_prevalence_speaker_l2 into the speaker's utterance choices. When choosing how to describe an object, speakers exhibit a contextual accessibility bias that favors words describing more objects in the visual scene over rare or specific distinguishing features. Pragmatic listeners invert this prevalence-biased speaker at depth 2, expecting speakers to use prevalent labels even for objects that possess unique alternatives, which softens preemption penalties against competitor referents while featureless baseline salience maintains default expectations on unprompted trials.

Differences from empty_baseline_salience_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_prevalence (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Speaker utterance utility / probability: In L1 and L2, the speaker's candidate utterances are weighted by contextual feature prevalence accessibility (vec(accessibility, u)), where accessibility scales with feature occurrence frequency across context objects (jnp.exp(w_prevalence * jnp.log(freq))).
- All other terms: Discrete singleton salience, contextual distinctiveness, visual feature count, featureless baseline salience (w_empty), visual color contrast, unbiased literal L0 semantics, depth-2 RSA recursion, and uniform lapse mixing remain identical to empty_baseline_salience_l2.
