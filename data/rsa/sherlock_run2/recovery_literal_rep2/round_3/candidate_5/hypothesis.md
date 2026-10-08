Listeners interpret referring expressions through feature contrast weighted by contextual distinctiveness (taken from contextual_distinctiveness_listener). Rather than penalizing all unmentioned features equally, listeners discount common or ubiquitous features that are shared across objects in the display, penalizing candidate referents primarily when they possess rare or unique distinguishing features that the speaker conspicuously omitted. This sensitive contrast evaluation captures how listeners attend to contextual communicative alternatives without requiring recursive speaker simulation.

Differences from feature_contrast_neutral_prior:
- Refined model: feature_contrast_neutral_prior
- Recursion depth: Depth 0 (calls L_contrast, unchanged).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In choice_probs, unmentioned features are weighted by contextual distinctiveness (inverse frequency across objects in the display, taken from contextual_distinctiveness_listener) rather than receiving a uniform penalty of 1 per feature.
