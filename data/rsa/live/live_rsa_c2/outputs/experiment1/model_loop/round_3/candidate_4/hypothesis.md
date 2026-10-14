Refining bounded_capacity_listener_l2, we incorporate contextual feature redundancy discounting on speaker alternatives from feature_redundancy_discounted_l2 into the communicative likelihood. When interpreting referring expressions, capacity-limited listeners reason about speakers who discount the availability of redundant alternative utterances based on their contextual feature overlap across the scene, preventing synonymous or co-occurring attributes from repeatedly compounding the preemption penalty against shared labels. This directly resolves the source model's severe underprediction of competitor choices when competitor referents possess redundant distinguishing features.

Differences from bounded_capacity_listener_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_redundancy (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Speaker utterance utility / probability: In S1 and S2, the speaker's candidate utterances are weighted by contextual feature redundancy discounting (vec(alt_weight, u)), where alt_weight scales with contextual pairwise Jaccard feature overlap (jnp.exp(-w_redundancy * jnp.log(jnp.maximum(redundancy, 1.0)))).
- All other terms: Bounded capacity likelihood exponentiation (kappa) in L1 and L2, referent prior (singleton salience w_singleton, contextual distinctiveness w_distinct, feature count w_features, featureless baseline salience w_empty, and color contrast w_color), literal semantics at L0, depth-2 RSA recursion, and uniform lapse mixing remain identical to bounded_capacity_listener_l2.
