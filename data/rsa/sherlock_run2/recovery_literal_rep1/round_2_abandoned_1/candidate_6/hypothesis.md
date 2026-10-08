Listeners interpret referential expressions through depth-one pragmatic reasoning with bounded decision rationality, sharing an affective valence-modulated salience prior with the speaker. When a speaker describes their favorite object or speaks neutrally, listeners expect them to favor feature-rich items, whereas negative framing ('least favorite') attenuates or inverts this preference toward plainer items. Listeners coordinate on this affective prior while making softmax-rational decisions over the speaker posterior, smoothing pragmatic inferences across referents.

Differences from source (listener_rationality_shared_prior):
- Model refined: listener_rationality_shared_prior
- Recursion depth: Unchanged (choice_probs calls L1 at depth 1).
- Parameters added: w_valence (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs, the object prior logits replace params["w_features"] * ctx.feature_count with (params["w_features"] + params["w_valence"] * ctx.valence) * ctx.feature_count, incorporating the affective valence modulation mechanism from valence_salience_listener.
