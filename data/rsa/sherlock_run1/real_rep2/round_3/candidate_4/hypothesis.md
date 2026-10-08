Listeners engage in depth-2 recursive pragmatic reasoning by inverting an informative speaker who simulates a depth-1 pragmatic listener, while maintaining both a common-knowledge object salience prior grounded in diagnostic feature contrast and an expectation that speakers evaluate referring expressions based on communicative specificity minus utterance ambiguity cost across all levels of recursion. In complex multi-feature displays, depth-2 listeners resolve higher-order scalar implicatures where depth-1 listeners under-differentiate referents, while preserving the mutual sensitivity to distinctive features and speaker production costs. This fits better because pragmatic speakers anticipate counterfactual pragmatic inferences rather than merely literal ones, sharpening referent identification across multi-feature contexts while retaining diagnostic contrast salience.

Refinement of costly_speaker_contrast_prior:
- Base model refined: costly_speaker_contrast_prior
- Recursion depth: Depth 2 (choice_probs calls L2 instead of L1).
- Parameters added: None (retains alpha, w_contrast, w_familiar, cost_ambiguity, and lapse).
- Parameters removed: None.
- Other terms: choice_probs calls L2 instead of L1, adding the L2 listener memo that inverts an informative speaker S2 who simulates L1; speaker S2 evaluates candidate utterances using informativeness under L1 minus the shared utterance ambiguity cost vector.
