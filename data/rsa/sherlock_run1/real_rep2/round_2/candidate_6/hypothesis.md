Listeners engage in depth-2 recursive pragmatic reasoning by inverting an informative speaker who simulates a pragmatic listener, while incorporating both shared visual salience and speaker-side utterance ambiguity costs. When candidate expressions apply to multiple competitor objects, speakers incur production costs proportional to this referential ambiguity, modulating communicative choices across the recursive hierarchy while maintaining mutual sensitivity to perceptual color distinctiveness, feature count, and familiarization base rates.

Refinement of rsa_l2_color_salience:
- Base model refined: rsa_l2_color_salience
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from rsa_l2_color_salience).
- Parameters added: cost_ambiguity (prior: Normal(0.0, 1.0)).
- Parameters removed: None (retains alpha, w_features, w_familiar, w_color, and lapse).
- Other terms: choice_probs computes an utterance ambiguity cost as cost_ambiguity times the excess number of referents each non-sink word applies to beyond one, and passes it to L1 and L2; simulated speaker utilities in L1 and L2 subtract this cost vector (component taken from costly_speaker_shared_prior).
