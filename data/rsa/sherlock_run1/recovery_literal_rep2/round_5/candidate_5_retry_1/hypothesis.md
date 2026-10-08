This model refines valence_feature_economy_listener by incorporating visual color salience from color_salience_listener into the feature economy heuristic: when candidate referents literally match the uttered word, listeners penalize objects possessing extraneous unmentioned features (modulated by the speaker's evaluative valence) and discount desaturated grayscale objects relative to full-color alternatives. Full-color objects capture visual attention and enhance perceived referential suitability, whereas desaturated grayscale objects are dispreferred.

Differences from valence_feature_economy_listener:
- Refined model: valence_feature_economy_listener
- Recursion depth: depth 0 (choice_probs calls L_heuristic, unchanged).
- Parameters added: w_grayscale ~ Normal(0.0, 1.0) governing the visual color salience penalty for desaturated grayscale referents.
- Parameters removed: none.
- Heuristic listener L_heuristic: candidate referents are weighted by at(lex, u, r) * exp(-beta_eff * vec(feature_count, r) + w_grayscale * vec(grayscale, r)), incorporating the visual color salience component from color_salience_listener.
- Arguments to L_heuristic: takes w_grayscale from params["w_grayscale"] and grayscale: ... from ctx.grayscale.
- All other memo agents, distributions, and prior terms are unchanged.
