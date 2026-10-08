Listeners operate under visual attentional constraints that attenuate non-matching distractors during speaker simulation, while evaluating candidate referents through an inductive prior that integrates visual feature complexity (cognitive parsimony favoring objects with fewer defining features) alongside empirical familiarization base rates. By combining distractor attenuation with a joint base-rate and feature-complexity prior, listeners account for prior exposure frequencies and simplicity biases without sacrificing attentional discounting of non-matching visual competitors on communicative displays.

### Refinement of distractor_attenuation_base_rate
We refine distractor_attenuation_base_rate with a single stated change: extending the inductive referent prior to incorporate the feature-complexity parsimony mechanism from feature_salience_listener, so candidate referents are evaluated by both empirical familiarization frequencies and visual feature count.

### Differences from distractor_attenuation_base_rate
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from distractor_attenuation_base_rate).
- Parameter added: salience (prior dist.Normal(0.0, 1.0)), which scales the feature-complexity prior over candidate referents (taken from feature_salience_listener); no parameters removed (alpha, distractor_weight, base_rate_weight, and lapse retained with identical priors).
- Memo recursion: unchanged memo signatures and agent definitions; L0 and L1 continue to take lex: ..., atten: ..., and prior: ..., forwarding prior to the speaker (wpp=vec(prior, r)).
- Choice probabilities: choice_probs computes prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization + params["salience"] * ctx.feature_count), passing this joint inductive prior to L1 and returning it on prior trials (jnp.where(ctx.is_prior > 0, prior, heard)).
