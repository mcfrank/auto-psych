People choose referents by evaluating pragmatic posterior beliefs through a softmax decision rule, while grounding their common-knowledge prior over referents in diagnostic feature contrast rather than raw feature count. In visual reference displays, distinctive features unique to an object heighten its prior communicative salience whereas features shared with competitor objects dilute it, and listeners softly maximize over pragmatic beliefs that incorporate this diagnostic contrast. This fits better because diagnostic feature contrast accurately captures the perceptual distinctiveness of candidate referents before communication, which informs both counterfactual speaker simulation and boundedly rational belief maximization.

Refinement of softmax_belief_listener:
- Base model refined: softmax_belief_listener
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from softmax_belief_listener).
- Parameters added: w_contrast (prior: Normal(0.0, 1.0)).
- Parameters removed: w_features (retains alpha, gamma, w_familiar, and lapse).
- Other terms: choice_probs computes the shared object salience prior from diagnostic feature contrast (unique non-sink features minus shared competitor features) rather than raw feature counts before the softmax (component taken from diagnostic_contrast_listener); the L1 pragmatic listener memo, softmax decision rule with precision gamma, and prior-trial choice rule are unchanged.
