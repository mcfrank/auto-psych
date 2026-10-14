Refining bounded_capacity_redundancy_l2, we incorporate contextual end-point salience from endpoint_salience_l2 into the shared referent prior. In visual reference games, listeners and speakers anchor their default expectations at the boundaries of the scene's feature complexity distribution—favoring the minimal baseline object and the maximal exemplar—while intermediate objects falling strictly between these poles are disfavored as unremarkable non-targets. This bipolar extremity prior directly eliminates the systematic overprediction of intermediate-feature objects on unprompted mumble trials while modulating competitor choices during reference resolution.

Differences from bounded_capacity_redundancy_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_endpoint (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Object prior: Adds contextual end-point salience (w_endpoint * is_endpoint) to the sum of singleton salience, contextual distinctiveness, feature count, featureless baseline salience, and color contrast, where is_endpoint is 1.0 for objects with the minimum or maximum feature count in the scene and 0.0 for strictly intermediate objects.
- All other terms: Capacity-limited inferential updating (kappa) at L1 and L2, contextual feature redundancy discounting (w_redundancy via pairwise Jaccard overlap) in S1 and S2, literal semantics at L0, depth-2 RSA recursion, and uniform lapse mixing remain identical to bounded_capacity_redundancy_l2.
