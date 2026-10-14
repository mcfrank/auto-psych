Visual scene expectations and communicative attention are anchored asymmetrically by the maximally elaborated focal exemplar, rather than distributed symmetrically across complexity extremes. Refining contrastive_extremity_noisy_speaker by incorporating focal exemplar salience: while intermediate objects suffer perceptual compression relative to complexity boundaries, human visual search and communicative topic focus exhibit strong feature-positive asymmetry, favoring the most feature-rich exemplar over bare baseline contrasts. This focal exemplar boost directly resolves the incumbent model's underprediction of maximal-feature objects and overprediction of intermediate objects on uninformative mumble trials, while curbing overprediction of intermediate competitors on ambiguous utterance trials.

Differences from source (contrastive_extremity_noisy_speaker):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: w_max_feat (prior: dist.Normal(0.0, 1.0)), representing the visual salience of the maximal-feature focal exemplar.
- Parameters removed: none.
- Other terms: In choice_probs, the presence of maximal visual elaboration is computed as `is_max_feat = jnp.where((ctx.feature_count == max_count) & (ctx.feature_count > 0.0), 1.0, 0.0)` and included in the referent prior linear predictor with weight w_max_feat.
