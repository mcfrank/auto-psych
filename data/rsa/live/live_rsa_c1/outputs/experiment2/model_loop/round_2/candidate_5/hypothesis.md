Pre-attentive pop-out from scene uniqueness requires the presence of distinguishing visual features, so an unadorned, empty object does not capture visual attention merely because it lacks identical copies in the display. Pragmatic listeners invert a similarity-contrast speaker under common knowledge of visual endpoint extremity and stochastic transmission noise, but evaluate candidate referents under a prior where visual singleton salience applies only to unique objects possessing actual visual features. This feature-gated singleton salience directly corrects the incumbent model's over-selection of featureless singleton objects in scenes containing duplicated competitors on uninformative mumble trials, while preserving similarity-contrastive reasoning and endpoint salience.

Differences from source (contrastive_extremity_noisy_speaker):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: none.
- Parameters removed: none.
- Other terms: In the referent prior, discrete visual singleton salience is gated by the presence of visual features (`featured_singleton = is_singleton * jnp.where(ctx.feature_count > 0, 1.0, 0.0)`), so that empty objects lacking visual features receive no singleton pop-out boost.
