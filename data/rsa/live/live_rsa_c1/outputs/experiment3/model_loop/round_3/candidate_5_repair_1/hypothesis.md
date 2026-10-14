In visual reference games, human referential attention exhibits strong feature-positive asymmetry, favoring the maximally elaborated focal exemplar over unadorned baseline objects rather than treating complexity endpoints symmetrically. Refining contrastive_graded_noisy_speaker by incorporating focal exemplar salience from contrastive_focal_extremity_speaker: communicative speakers reason over a graded lexicon with perceptual confusability decaying with feature distance, but anchor candidate expectations preferentially on the visual scene's most feature-rich object. A pragmatic listener inverts this graded similarity-contrast speaker with stochastic execution noise, directly boosting choices of maximal-feature referents and suppressing intermediate competitors on both uninformative mumble trials and ambiguous utterance trials.

Differences from source (contrastive_graded_noisy_speaker):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: w_max_feat (prior: dist.Normal(0.0, 1.0)), representing the visual salience of the maximal-feature focal exemplar.
- Parameters removed: none.
- Parameter priors changed: none.
- Other terms: In choice_probs, the presence of maximal visual elaboration in the display is computed as `is_max_feat = jnp.where((ctx.feature_count == max_count) & (ctx.feature_count > 0.0), 1.0, 0.0)` and included in the referent prior linear predictor with weight w_max_feat.
