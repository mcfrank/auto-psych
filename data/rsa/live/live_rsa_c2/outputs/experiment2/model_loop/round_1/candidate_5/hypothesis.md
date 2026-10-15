We refine bounded_endpoint_prev_redundancy_l2 by incorporating salience-captured attentional lapses from salience_capture_lapse_l2 into listener response choice. When interpreting referring expressions under cognitive load, inattentive or impulsive responses are not distributed uniformly across candidate objects, but are captured by visual salience toward prominent exemplars according to the perceptual prior. This structured lapse mechanism directly resolves the severe underprediction of literal semantic violations (false-feature choices) on word trials while directing inattentive choices toward feature-rich competitors.

Differences from bounded_endpoint_prev_redundancy_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: None.
- Parameters removed: None.
- Response choice lapse mechanism: Replaces uniform lapse mixing (with_lapse) on utterance trials with salience-captured attentional lapse mixing (p_choice = (1.0 - lapse) * heard + lapse * prior, returning prior on uninformative prior trials) from salience_capture_lapse_l2.
- All other terms: Capacity-limited inferential updating (kappa) at L1 and L2, contextual feature prevalence accessibility (w_prevalence) and referential equivalence class size redundancy discounting (w_redundancy via compute_equivalence_class) on speaker alternatives in S1 and S2, literal truth-conditional semantics at L0, depth-2 RSA recursion, and shared perceptual referent prior (w_singleton, w_distinct, w_features, w_empty, w_color) remain identical to bounded_endpoint_prev_redundancy_l2.
