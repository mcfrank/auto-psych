This model refines bounded_capacity_listener by incorporating the feature economy heuristic from fewest_features_listener into the literal baseline: when processing a referring expression, the default literal interpretation penalizes candidate referents possessing extraneous unmentioned features in proportion to their total feature count. Listeners maintain bounded cognitive capacity for pragmatic reasoning, anchoring on this feature-economical literal interpretation and incorporating recursive speaker likelihood only to the extent permitted by their processing resources. This improves model fit by decoupling a priori visual salience from the communicative preference for minimally specified referents.

Differences from bounded_capacity_listener:
- Refined model: bounded_capacity_listener
- Recursion depth: depth 1 (choice_probs calls L_bounded, unchanged).
- Parameters added: beta_features ~ LogNormal(0.0, 1.0) governing the feature economy penalty against extraneous unmentioned features.
- Parameters removed: none.
- Literal listener L0: listener chooses r in OBJ with probability proportional to vec(prior, r) * at(lex, u, r) * exp(-beta_features * vec(feature_count, r)), incorporating the feature economy heuristic from fewest_features_listener so that the default literal baseline penalizes objects possessing extraneous unmentioned features.
- Speaker choice in S1: speaker simulates this feature-economical L0, passing beta_features and feature_count: ... into L0.
- Pragmatic listener L_bounded: listener anchors on the feature-economical literal baseline L0 and incorporates speaker likelihood S1 to the extent permitted by processing capacity, passing beta_features and feature_count: ... into L0 and S1.
- All other memo agents, distributions, and prior terms are unchanged.
