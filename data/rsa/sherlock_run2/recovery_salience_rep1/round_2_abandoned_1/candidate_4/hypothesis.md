We refine feature_simplicity_listener by extending it with the speaker ambiguity cost component from costly_feature_speaker, positing that communicators combine an inductive preference for simpler referents with an explicit penalty against uttering descriptors that apply to multiple referents in the scene. In choice_probs, recursion depth remains unchanged at depth 1 (calling L1), one parameter is added (cost with a LogNormal(0, 1) prior), and no parameters are removed. In L1, the speaker discounts ambiguous utterances proportionally to their scene extension via an ambiguity cost vector, allowing the listener to resolve referential ambiguity even when objects share the same number of features.

Differences from feature_simplicity_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: cost (LogNormal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes utterance ambiguity costs via (1.0 - ctx.is_sink) * jnp.maximum(0.0, jnp.sum(ctx.lex, axis=1) - 1.0) and passes them to L1, where the speaker subtracts cost * vec(costs, u) in their utility.
