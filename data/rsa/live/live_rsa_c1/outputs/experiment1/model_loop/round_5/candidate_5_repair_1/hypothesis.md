Perceptual pop-out from visual uniqueness requires the presence of distinguishing features, so an unadorned, empty object does not capture pre-attentive attention merely because it lacks identical copies in the scene. Pragmatic listeners invert a communicative speaker under a referent prior where visual singleton salience applies only to unique objects possessing actual visual features, translating their subjective posterior beliefs into choices via a bounded-precision softmax decision rule. Refining softmax_belief_listener by incorporating the feature-gated singleton salience component from feature_singleton_salience_l1: this prevents listeners from over-selecting empty singletons when competing against duplicated full objects on uninformative mumble trials, while bounded softmax sensitivity maps subjective beliefs onto behavioral clicks.

Differences from source (softmax_belief_listener):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: none.
- Parameters removed: none.
- Other terms: In the referent prior, discrete visual singleton salience is gated by visual feature presence (`featured_singleton = is_singleton * (ctx.feature_count > 0)`), ensuring that empty objects lacking visual features receive no singleton pop-out boost.
