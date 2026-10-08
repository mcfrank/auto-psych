This model refines softmax_listener_shared_prior by replacing the raw feature-count salience prior with contextual feature rarity: an object is salient to the extent that its features are distinctive and rare across competitors in the visual scene, rather than merely numerous. Integrating contextual rarity into the shared prior combines common-knowledge distinctive salience with independent softmax decision rationality for the pragmatic listener, predicting that listeners prefer referents with unique identifying features even when competitor objects have equal total feature counts.

Differences from softmax_listener_shared_prior:
- Refined model: softmax_listener_shared_prior
- Recursion depth: depth 1 (choice_probs calls L1, unchanged).
- Parameters added: w_rarity ~ Normal(0.0, 1.0) governing the weight of contextual feature rarity in the shared prior.
- Parameters removed: w_features.
- Prior calculation in choice_probs: replaces raw feature count with contextual feature rarity (features weighted inversely by their frequency across objects in the display, scaled by params["w_rarity"]).
- All other memo agents, distributions, and prior terms are unchanged.
