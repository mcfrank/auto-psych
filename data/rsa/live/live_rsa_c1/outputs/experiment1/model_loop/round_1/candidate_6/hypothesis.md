Pragmatic listener at depth 1 with singleton salience, contextual distinctiveness, visual feature complexity, visual color contrast, and familiarization base rates. Refining rsa_l1 by incorporating the multi-attribute perceptual and experiential referent prior from rsa_l2_singleton_feat_color_familiar_l0: listeners reason at first-order depth, directly inverting a cooperative speaker who balances literal informativeness against visual and exposure biases. Because listeners perform only single-step pragmatic inversion, they do not compound likelihood ratios across recursive tiers, curbing the excessive target preferences observed at depth 2 when competitor objects possess multiple additional features or duplicate copies. On uninformative mumble trials, choices directly reflect this shared perceptual and experiential prior rather than uniform guessing.

Differences from rsa_l1:
- Recursion depth: depth 1 (calls L1, unchanged from rsa_l1).
- Parameters added: w_singleton, w_distinct, w_features, w_color, w_familiar.
- Parameters removed: none.
- Other terms: The simulated speaker in L1 samples referents according to the multi-attribute perceptual prior vec(prior, r) instead of a uniform distribution, and on uninformative prior trials choices follow this prior directly rather than uniform guessing.
