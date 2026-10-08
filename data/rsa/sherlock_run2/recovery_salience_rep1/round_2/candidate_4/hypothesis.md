We refine feature_simplicity_listener by extending it with the speaker ambiguity cost component from costly_feature_speaker, positing that communicators combine an inductive preference for simpler referents with an explicit communicative penalty against uttering descriptors that apply to multiple referents in the scene. Listeners reason about this ambiguity-averse speaker, expecting a speaker who used a shared word to have lacked any less ambiguous alternative for their intended referent, which improves referential disambiguation in contexts where competitors have equal feature counts.

Differences from feature_simplicity_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: cost (LogNormal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes utterance ambiguity costs via (1.0 - ctx.is_sink) * jnp.maximum(0.0, jnp.sum(ctx.lex, axis=1) - 1.0) and passes them to L1, where the speaker subtracts cost * vec(costs, u) in their utility.
