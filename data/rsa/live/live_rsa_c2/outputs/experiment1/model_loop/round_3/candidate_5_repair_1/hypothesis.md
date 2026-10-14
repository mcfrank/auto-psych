Refining bounded_capacity_listener_l2, we incorporate contextual feature prevalence from feature_prevalence_speaker_l2 into the speaker's utterance choices. When communicating about an object, speakers exhibit a contextual accessibility bias that favors words describing more objects in the visual scene over rare or specific distinguishing features. Capacity-limited pragmatic listeners invert this prevalence-biased speaker under finite information-processing capacity, naturally expecting speakers to use prevalent labels even for objects that possess unique alternatives and appropriately softening preemption penalties against competitor referents in scalar reference games.

Differences from bounded_capacity_listener_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_prevalence (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Speaker utterance utility / probability: In S1 and S2, candidate utterances are weighted by contextual feature prevalence accessibility (vec(accessibility, u)), where accessibility scales with feature occurrence frequency across context objects (jnp.exp(w_prevalence * jnp.log(freq))).
- All other terms: Capacity-limited likelihood updating via inferential precision exponentiation (kappa) at L1 and L2, shared visual referent prior (w_singleton, w_distinct, w_features, w_empty, w_color), truth-conditional literal semantics at L0, depth-2 RSA recursion, and uniform lapse mixing remain identical to bounded_capacity_listener_l2.
