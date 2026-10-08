Listeners interpret a speaker by combining common-knowledge object salience with an expectation that speakers evaluate referring expressions based on both communicative specificity and utterance ambiguity cost. When multiple candidate words are true, speakers incur a cost proportional to the extension of ambiguous features across competitor objects, and pragmatic listeners invert this cost-sensitive speaker while preserving shared perceptual salience. This fits better because pragmatic listeners penalize broad, ambiguous expressions more realistically than literal informativeness alone would dictate, while retaining base-rate and salience sensitivity.

Refinement of rsa_l1_shared_prior:
- Base model refined: rsa_l1_shared_prior
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from rsa_l1_shared_prior).
- Parameters added: cost_ambiguity (prior: Normal(0.0, 1.0)).
- Parameters removed: None (retains alpha, w_features, w_familiar, and lapse).
- Other terms: choice_probs computes an utterance ambiguity cost as cost_ambiguity times the excess number of referents each non-sink word applies to beyond one, and passes it to L1; the simulated speaker utility in L1 subtracts this cost vector (component taken from costly_feature_speaker).
