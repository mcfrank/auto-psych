Listeners interpret referring expressions through proportional feature contrast, evaluating candidate referents by penalizing unmentioned features relative to each object's total feature complexity rather than applying a fixed additive penalty per omitted feature. In accordance with proportional perceptual comparison, an unmentioned feature is penalized more heavily when it constitutes a larger fraction of the object's total features, reflecting greater descriptive incompleteness for simple referents than for already complex ones. This proportional penalty allows listeners to resolve scalar implicatures sensitively across visual scenes of varying feature density without requiring recursive speaker simulation.

Differences from feature_contrast_neutral_prior:
- Refined model: feature_contrast_neutral_prior
- Recursion depth: Depth 0 (calls L_contrast, unchanged).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In choice_probs, the penalty for unmentioned features is normalized by each object's total feature count (unmentioned_count / max(1, feature_count)) rather than scaling linearly with raw unmentioned feature count.
