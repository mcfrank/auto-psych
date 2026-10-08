Listeners engage in depth-2 recursive pragmatic reasoning by inverting an informative speaker who simulates a depth-1 pragmatic listener, while sharing both common-knowledge object salience and an expectation that speakers incur costs for ambiguous expressions. When reasoning about referring expressions, interlocutors at each recursive step anticipate that candidate words carrying higher ambiguity across competitors penalize speaker utility, sharpening inferences about intended referents in multi-feature contexts. This should fit better by correctly resolving higher-order scalar implicatures across complex displays while preserving the ambiguity-cost penalties and shared salience captured by the incumbent model.

Refinement of costly_speaker_shared_prior:
- Base model refined: costly_speaker_shared_prior
- Recursion depth: Depth 2 (choice_probs calls L2 instead of L1).
- Parameters added: None.
- Parameters removed: None (retains alpha, w_features, w_familiar, cost_ambiguity, and lapse).
- Other terms: Adds the L2 listener memo inverting a simulated speaker S2 who evaluates utterances against L1 pragmatic informativeness subject to the shared utterance ambiguity cost; choice_probs evaluates L2 instead of L1.
