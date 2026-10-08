Depth-2 pragmatic listeners evaluate candidate referents through an inductive prior that integrates visual feature complexity (cognitive parsimony favoring objects with fewer defining features) alongside empirical familiarization base rates, while expecting speakers to incur a cognitive cost for ambiguous utterances. On displays without prior familiarization, this simplicity preference guides baseline expectations on uninformative trials and shapes recursive speaker simulation, capturing human biases toward simpler referents on complex displays. On familiarization displays, empirical exposure frequencies combine with feature complexity to guide reference resolution without degrading base-rate sensitivity.

### Refinement of base_rate_ambiguity_l2
We refine base_rate_ambiguity_l2 with a single stated change: extending the inductive referent prior to incorporate the feature-complexity parsimony mechanism from feature_salience_l2, so candidate referents are evaluated by both empirical familiarization frequencies and visual feature count.

### Differences from base_rate_ambiguity_l2
- Recursion depth: depth 2 (choice_probs calls L2, unchanged from base_rate_ambiguity_l2).
- Parameter added: salience (prior dist.Normal(0.0, 1.0)), which scales the feature-complexity prior over candidate referents (taken from feature_salience_l2); no parameters removed (alpha, cost, base_rate_weight, and lapse retained with identical priors).
- Memo recursion: unchanged memo signatures and agent definitions; L1 and L2 continue to take costs: ..., lex: ..., prior: ..., and forward prior to the speaker (wpp=vec(prior, r)).
- Choice probabilities: choice_probs computes prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization + params["salience"] * ctx.feature_count), passing this joint inductive prior to L2 and returning it on prior trials (jnp.where(ctx.is_prior > 0, prior, heard)).
