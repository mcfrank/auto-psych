Listeners interpret a speaker by combining an expectation that speakers incur utterance ambiguity costs with common-knowledge object salience grounded in diagnostic feature contrast rather than raw feature count. In visual reference displays, features unique to an object heighten its prior communicative salience while shared competitor features dilute it, and speakers penalize candidate words that apply ambiguously across multiple objects. This fits better because diagnostic feature contrast captures the mutual informativeness and distinctiveness of competitor referents before word interpretation, complementing the speaker's utterance-level ambiguity penalty.

Refinement of costly_speaker_shared_prior:
- Base model refined: costly_speaker_shared_prior
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from costly_speaker_shared_prior).
- Parameters added: w_contrast (prior: Normal(0.0, 1.0)).
- Parameters removed: w_features (retains alpha, w_familiar, cost_ambiguity, and lapse).
- Other terms: choice_probs computes the shared object salience prior from diagnostic feature contrast (unique non-sink features minus shared competitor features) rather than raw feature counts before the softmax (component taken from diagnostic_contrast_listener); the cost-sensitive speaker utility is unchanged.
