We refine fewest_features_listener by proposing that the listener's parsimony heuristic is modulated by the speaker's affective framing: when interpreting an utterance, listeners penalize feature complexity under neutral or positive framing ("favorite"), but invert this preference when the speaker describes their "least favorite" object, favoring referents with more features. When no informative word is heard, listeners choose uniformly among candidate objects, modulated only by familiarization base rates.

Differences from source (fewest_features_listener):
- Recursion depth: Unchanged (choice_probs calls L0 at depth 0).
- Parameters added: w_valence (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs, the feature parsimony exponent `-params["beta"] * ctx.feature_count` is replaced by `-(params["beta"] + params["w_valence"] * ctx.valence) * ctx.feature_count`, modulating the parsimony penalty according to speaker valence.
