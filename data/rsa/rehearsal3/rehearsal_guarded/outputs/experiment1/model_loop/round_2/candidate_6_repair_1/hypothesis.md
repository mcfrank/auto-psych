When identifying a speaker's intended referent in complex visual scenes, listeners reason at depth 3 over graded semantics and utterance extension costs, integrating visual feature complexity alongside discrete singleton pop-out and continuous perceptual isolation. Rather than assuming that prior salience is determined solely by isolation and duplicate status, listeners expect speakers to favor visually richer, feature-complex objects a priori and carry this prior through higher-order recursive mentalizing when interpreting ambiguous expressions.

Model refined: `isolated_graded_costly_singleton_l3`.
Differences from source:
- Recursion depth: Unchanged (depth 3, calling L3 in choice_probs).
- Parameters added: `w_features` (prior Normal(0.0, 1.0)) weighting visual feature complexity in the referent prior.
- Parameters removed: None.
- Referent prior: Extended from combining discrete singleton salience and continuous isolation (`softmax_prior(w_singleton * is_singleton + w_distinct * dist_vec)`) to also include visual feature complexity (`softmax_prior(w_singleton * is_singleton + w_distinct * dist_vec + w_features * ctx.feature_count)`).
- All other components (graded semantics with parameter `gamma`, perceptual isolation via minimum Hamming distance, utterance extension costs with parameter `cost_weight`, and background lapses) unchanged.
