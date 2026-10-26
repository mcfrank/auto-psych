Speakers choose referring expressions to maximize contextual contrast against visually similar competitors in the immediate display, while listeners interpret utterances by inverting this contrastive speaker. On uninformative displays without a referring expression and when forming referent expectations, visual salience integrates discrete singleton pop-out with visual feature complexity, expecting speakers to refer to richer, more complex objects with more visual features.

Refining contextual_contrast_listener by incorporating visual feature complexity from rsa_l2_singleton_feat_color_valence_l0 directly resolves the discrepancy on distinct uninformative displays (prior_dense_vs_sparse_distinct_unit42), where discrete singletons are otherwise treated as uniform, while elevating belief in feature-rich singleton competitors (multiclone_competitor_margin_unit20).

## Differences from contextual_contrast_listener

1. **Recursion depth**: Unchanged (depth 1, listener calls `L_contrast`).
2. **Parameters added**: `w_features` with prior `dist.Normal(0.0, 1.0)`. No parameters removed.
3. **Object prior**: In `listener(params, ctx)`, `prior` now adds `params["w_features"] * ctx.feature_count` to the logit alongside `params["w_singleton"] * is_singleton`. All speaker contrast and utility calculations are unchanged.
