Perceptual pop-out from uniqueness in a visual scene requires distinguishing features; an unadorned, empty object does not capture attention merely because it lacks identical copies. Pragmatic listeners invert a first-order noisy-channel speaker with a referent prior where visual singleton salience applies only to unique objects possessing at least one visual feature. This prevents listeners from overselecting featureless singleton objects in displays containing duplicated competitors on uninformative trials, while transmission noise tempers pragmatic likelihood ratios when hearing descriptive words.

Refining noisy_channel_speaker_l1 by incorporating the feature-gated singleton salience component from feature_singleton_salience_l1:
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: none.
- Parameters removed: none.
- Other terms: In the referent prior, visual singleton salience is gated by the presence of visual features (`featured_singleton = is_singleton * (ctx.feature_count > 0)`), so that empty objects with no duplicates receive zero singleton pop-out boost.
