We refine familiar_isolated_graded_costly_l2 by incorporating visual feature complexity from rsa_l2_singleton_distinct_feat_color_l0 into the shared referent prior. People intuitively direct visual attention toward objects possessing a greater number of distinct features, making feature-rich items more salient candidate referents during both unprompted expectation and reference resolution. This visual complexity bias elevates the prior probability of feature-dense competitor objects, explaining why listeners choose overspecified competitors with redundant features far more frequently than predicted by standard communicative efficiency alone.

Differences from familiar_isolated_graded_costly_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_features (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Object prior: Adds feature count weighted by w_features to the sum of perceptual isolation and familiarization.
- All other terms: Graded semantic applicability, perceptual isolation, utterance extension costs, depth-2 RSA recursion, and uniform lapse remain identical to the source model.
