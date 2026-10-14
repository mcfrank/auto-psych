We refine bounded_prevalence_redundancy_l2 by incorporating salience-captured attentional lapse from salience_capture_lapse_l2 into listener choice. When listeners experience an attentional lapse or execute an impulsive response during reference resolution, choices are not uniform random noise across the visual scene, but are instead captured by bottom-up visual salience according to the perceptual referent prior. This structured lapse directly resolves the systematic underprediction of literal semantic violations (false-feature choices) and elevates choices for feature-rich competitors while preserving bounded-capacity pragmatic inference, contextual feature prevalence, and redundancy discounting.

Differences from bounded_prevalence_redundancy_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: None.
- Parameters removed: None.
- Response selection / lapse mechanism: Replaces uniform random lapse mixing (with_lapse) with salience-captured attentional lapse from salience_capture_lapse_l2 (p_choice = (1.0 - lapse) * heard + lapse * prior, returning prior on uninformative prior trials).
- All other terms: Capacity-limited inferential updating (kappa) at L1 and L2, contextual feature prevalence accessibility (w_prevalence) and contextual feature redundancy discounting (w_redundancy) on speaker alternatives in S1 and S2, literal semantics at L0, and shared referent prior (w_singleton, w_distinct, w_features, w_empty, w_color) remain identical to bounded_prevalence_redundancy_l2.
