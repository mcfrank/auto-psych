Refining feature_redundancy_discounted_l2, we incorporate a featureless baseline salience component from empty_baseline_salience_l2 into the shared referent prior. In visual displays containing featured objects, an entirely featureless object acts as a distinct perceptual ground and visual anchor that attracts baseline attention, rather than having its prior probability monotonically penalized by visual feature complexity. This baseline salience directly rectifies the empirical under-selection of empty objects and over-selection of intermediate-feature objects on unprompted mumble trials.

Differences from feature_redundancy_discounted_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_empty (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Object prior: Adds featureless baseline salience (w_empty * is_empty, where is_empty indicates zero features) to the sum of singleton salience, contextual distinctiveness, feature count, and color contrast.
- All other terms: Contextual feature redundancy discounting on speaker alternatives, pairwise Jaccard overlap calculation, depth-2 RSA recursion, and uniform lapse mixing remain identical to the source model.
