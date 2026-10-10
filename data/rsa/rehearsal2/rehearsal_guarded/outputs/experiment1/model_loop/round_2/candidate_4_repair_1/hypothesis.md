Listeners interpret referring expressions by inverting a speaker who avoids competitor confusion while evaluating words under graded semantics. When an object possesses multiple visual attributes, the semantic applicability and prominence of any single feature is diluted compared to an isolated feature, leading listeners to prefer simpler or more prototypical referents when descriptions are ambiguous. This graded semantic precision operates alongside visual salience and competitor confusion avoidance at depth-2 pragmatic recursion.

Refining rsa_l2_salience_confusion_l0 by incorporating graded semantics from isolated_graded_costly_l3:
Speakers and listeners evaluate candidate descriptions using graded truth values where a feature's semantic precision diminishes exponentially with the number of competing features on the referent. This captures graded typicality and feature interference, improving fit on displays where candidate referents vary in visual complexity.

Differences from rsa_l2_salience_confusion_l0:
- Recursion depth: unchanged at depth 2 (choice_probs calls L2).
- Parameters added: gamma (Normal(0.0, 1.0)), governing the rate of semantic precision decay with additional competing features.
- Parameters removed: none.
- Other terms: added compute_graded_lex to scale word applicability by exp(-gamma * (feature_count - 1)) for non-sink utterances; passed graded_lex instead of ctx.lex to literal listener L0 and simulated speakers S1 and S2.
