Visual pop-out from scene uniqueness requires the presence of distinguishing visual features, so an unadorned empty object does not capture pre-attentive attention merely because it lacks duplicates. Pragmatic listeners invert a communicative speaker who alternates between uniquely identifying a referent and simply describing an aspect of it, under a referent prior where visual singleton salience applies only to unique objects possessing actual visual features. This prevents listeners from overselecting empty singletons when competing against duplicated full objects on uninformative trials, while aspect-goal speaker uncertainty cushions pragmatic likelihood ratios when hearing descriptive words.

Differences from source (speaker_aspect_goal_listener):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: none.
- Parameters removed: none.
- Other terms: In the referent prior, discrete visual singleton salience is gated by the presence of visual features (`featured_singleton = is_singleton * (ctx.feature_count > 0)`), ensuring that empty objects lacking visual features receive no singleton pop-out boost.
