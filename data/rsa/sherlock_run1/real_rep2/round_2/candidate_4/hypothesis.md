Listeners engage in depth-2 recursive pragmatic reasoning by inverting an informative speaker who simulates a depth-1 pragmatic listener, while maintaining both common-knowledge object salience and an expectation that speakers evaluate referring expressions based on communicative specificity minus utterance ambiguity cost across all levels of recursion. This should fit better because depth-2 reasoning resolves higher-order scalar implicatures in complex multi-feature contexts where depth-1 listeners under-differentiate referents, while preserving the realistic utterance ambiguity penalization and perceptual base-rate sensitivity of the incumbent.

Refinement of costly_speaker_shared_prior:
- Base model refined: costly_speaker_shared_prior
- Recursion depth: Depth 2 (choice_probs calls L2 instead of L1).
- Parameters added: None (retains alpha, w_features, w_familiar, cost_ambiguity, and lapse).
- Parameters removed: None.
- Other terms: choice_probs calls L2 instead of L1, adding the L2 listener memo that inverts an informative speaker S2 who simulates L1; speaker S2 evaluates candidate utterances using informativeness under L1 minus the shared utterance ambiguity cost vector.
