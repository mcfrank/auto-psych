Listeners interpret referential and evaluative descriptions by combining depth-2 recursive pragmatic reasoning with graded semantic truth values, refining costly_valence_l2_listener by incorporating lexical dilution from graded_semantics_listener into the lexicon shared across reasoning levels. Rather than treating descriptive terms as binary, listeners perceive a word as a prototypical match for an object with only that feature, but as a diluted, less complete description of an object cluttered with extraneous features, sharpening the preference for minimal exemplars while continuing to invert a speaker who balances informativeness against utterance extension costs under evaluative framing.

Refined model: costly_valence_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: beta_graded (Normal(0.0, 1.0)); no parameters removed (retains alpha, w_features, w_familiar, w_valence, w_extension, lapse with identical priors).
- Terms changed: The lexicon passed to L2 is graded_lex instead of ctx.lex, where graded_lex scales non-sink entries of ctx.lex by dilution = exp(-beta_graded * max(feature_count - 1.0, 0.0)).
