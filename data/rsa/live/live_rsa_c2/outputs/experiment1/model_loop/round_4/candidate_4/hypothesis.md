We refine bounded_capacity_redundancy_l2 by incorporating salience-captured attentional lapses from salience_capture_lapse_l2 into listener response choice. When interpreting referring expressions under cognitive load, inattentive or impulsive responses are not distributed uniformly across candidate objects, but are captured by visual salience toward prominent exemplars according to the perceptual prior. This structured lapse mechanism directly resolves the severe underprediction of literal semantic violations (false-feature choices) while directing inattentive choices toward feature-rich competitors.

Differences from bounded_capacity_redundancy_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: None.
- Parameters removed: None.
- Response choice lapse mechanism: Replaces uniform lapse mixing (with_lapse) on utterance trials with salience-captured lapse mixing ((1.0 - lapse) * heard + lapse * prior) from salience_capture_lapse_l2.
- All other terms: Capacity-limited likelihood updating via inferential precision exponentiation (kappa) at L1 and L2, speaker contextual feature redundancy discounting (w_redundancy via pairwise Jaccard overlap), shared perceptual referent prior (w_singleton, w_distinct, w_features, w_empty, w_color), truth-conditional literal semantics at L0, depth-2 RSA recursion, and unprompted prior choice remain identical to bounded_capacity_redundancy_l2.
