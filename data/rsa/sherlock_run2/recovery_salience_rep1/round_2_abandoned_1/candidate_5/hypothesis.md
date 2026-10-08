We refine feature_simplicity_listener by incorporating an utterance ambiguity cost from costly_feature_speaker, positing that speakers not only favor referents with fewer distinguishing features a priori, but also experience an explicit communicative penalty when choosing ambiguous words that describe multiple referents in the scene. Listeners invert this ambiguity-averse and simplicity-sensitive speaker to resolve referential ambiguity, expecting a speaker who used a shared descriptor to have lacked any less ambiguous alternative for their intended referent. This improves fit because participants in reference games with redundant or overlapping features penalize ambiguous descriptors even beyond what literal informativeness dictates, complementing the referent-simplicity preference that governs display-level expectations.

Differences from feature_simplicity_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: cost (LogNormal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes an utterance ambiguity vector measuring extension beyond a single referent via (1.0 - ctx.is_sink) * jnp.maximum(0.0, jnp.sum(ctx.lex, axis=1) - 1.0) and passes it to L1; in L1, the speaker subtracts cost times this ambiguity from communicative utility.
