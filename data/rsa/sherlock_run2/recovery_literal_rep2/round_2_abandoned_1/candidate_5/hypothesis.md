Listeners interpret referring expressions through direct feature contrast rather than recursive mental simulation, but evaluate referent salience by conditioning on the speaker's affective framing (taken from valence_salience_listener). Under positive or neutral framing, listeners penalize matching referents that possess additional unmentioned features while maintaining a mild prior preference for feature-rich objects; under negative framing ('least favorite'), this prior preference inverts so that simpler objects are favored. This allows direct feature contrast to account for context-dependent shifts in reference resolution when speakers express negative attitudes.

Differences from feature_contrast_listener:
- Recursion depth: Depth 0 (choice_probs calls L_contrast, unchanged).
- Parameters added: w_valence (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs, the feature-count coefficient in the object salience prior is modulated by affective framing: feature_weight = params["w_features"] + params["w_valence"] * ctx.valence (taken from valence_salience_listener).
