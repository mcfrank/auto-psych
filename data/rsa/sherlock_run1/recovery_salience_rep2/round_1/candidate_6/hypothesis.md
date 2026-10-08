Listeners invert a pragmatic speaker under an object prior that penalizes feature complexity, preferring referents with fewer extraneous features (a simplicity bias). When interpreting an utterance or predicting choices without informative words, participants favor simpler objects over more heavily modified alternatives.

Refinement of rsa_l1:
- Base model: rsa_l1 (depth 1).
- Recursion depth: L1 (depth 1), unchanged.
- Parameters added: feature_cost (prior: Normal(0.0, 0.5)), representing the penalty per feature on an object's log prior probability.
- Parameters removed: none (retains alpha and lapse).
- Mathematical / structural changes: In choice_probs and L1, the uniform object prior is replaced by a complexity-penalized prior over objects (softmax_prior(-feature_cost * feature_count)), passed into L1 as prior and used on uninformative prior trials instead of uniform guessing.
