When identifying a speaker's intended referent in visual contexts containing duplicate objects, listeners combine continuous perceptual isolation and graded semantic applicability with a discrete perceptual singleton bias that prioritizes unique singleton objects. Rather than evaluating distinctiveness purely through continuous feature distances, listeners spontaneously privilege solitary referents that lack identical copies in the scene when interpreting ambiguous referring expressions or uninformative prior displays.

Model refined: `isolated_graded_costly_l3`.
Differences from source:
- Recursion depth: Unchanged (depth 3, calling L3 in choice_probs).
- Parameters added: `w_singleton` (prior Normal(0.0, 1.0)) weighting discrete singleton salience in the referent prior.
- Parameters removed: None.
- Referent prior: Extended from purely continuous perceptual isolation (`softmax_prior(w_distinct * dist_vec)`) to combine discrete singleton pop-out with continuous isolation (`softmax_prior(w_singleton * is_singleton + w_distinct * dist_vec)`).
