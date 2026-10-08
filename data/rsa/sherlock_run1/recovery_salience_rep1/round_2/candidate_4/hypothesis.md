We refine feature_salience_listener by integrating empirical base rates from familiarization exposure into the listener's referent prior alongside visual feature complexity. Rather than relying solely on visual simplicity to form baseline expectations, listeners combine experiential frequency and visual feature salience to establish prior probabilities over candidate referents. This allows the model to capture how participants systematically favor familiar objects when base rates are present while retaining the preference for simpler objects when base rates are absent.

Differences from feature_salience_listener:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from feature_salience_listener).
- Parameters added: baserate ~ LogNormal(0.0, 1.0), scaling the log-probability of empirical familiarization base rates.
- Parameters removed: None (alpha, salience, and lapse retained unchanged).
- Functional terms changed: In choice_probs, the referent prior vector incorporates empirical familiarization frequencies via softmax_prior(params["salience"] * ctx.feature_count + params["baserate"] * jnp.where(ctx.has_familiarization > 0, jnp.log(ctx.familiarization + EPS), 0.0)), replacing the feature-only prior.
