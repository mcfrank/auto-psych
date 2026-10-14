We refine feature_redundancy_discounted_l2 by incorporating featureless baseline salience from empty_baseline_salience_l2 into the shared referent prior. In visual reference displays, an entirely featureless object serves as a distinct perceptual ground and baseline anchor that attracts default attention, rather than being monotonically penalized by visual feature complexity. This baseline attraction elevates the prior probability of featureless objects on uninformative trials and reference games alike, directly resolving the source model's severe overprediction of intermediate feature-count objects in multi-level displays.

Differences from feature_redundancy_discounted_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_empty (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Object prior: Adds featureless baseline salience weighted by w_empty (is_empty = jnp.where(ctx.feature_count == 0, 1.0, 0.0)) to the sum of singleton salience, contextual distinctiveness, visual feature count, and color contrast.
- All other terms: Contextual feature redundancy discounting via pairwise Jaccard overlap (w_redundancy), unbiased literal semantics at L0, depth-2 RSA recursion (L1, L2), and uniform lapse mixing remain identical to the source model.
