We refine feature_prior_l2 by incorporating speaker ambiguity costs from costly_feature_speaker, positing that communicators combine an inductive preference for simpler referents with an explicit penalty against uttering descriptors that apply to multiple referents in the scene. Listeners reason recursively at depth 2 about an ambiguity-averse speaker who considers both referent simplicity and descriptor ambiguity, expecting a speaker who used a shared descriptor to have lacked any less ambiguous alternative for their intended referent.

Differences from feature_prior_l2:
- Recursion depth: unchanged (choice_probs calls L2, depth 2).
- Parameters added: cost (LogNormal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes utterance ambiguity costs via (1.0 - ctx.is_sink) * jnp.maximum(0.0, jnp.sum(ctx.lex, axis=1) - 1.0) and passes them to L1 and L2; in both L1 and L2, the speaker subtracts cost * vec(costs, u) from communicative utility when choosing an utterance.
