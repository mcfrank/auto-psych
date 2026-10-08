Listeners perform depth-2 recursive social reasoning to invert a pragmatic speaker, while treating object salience—shaped by perceptual feature counts and prior familiarization—as shared common knowledge across all interlocutors. In the absence of an informative description, people choose objects according to this common salience prior.

Refinement of rsa_l2:
- Base model refined: rsa_l2
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from rsa_l2).
- Parameters added: w_features (weight on feature count) and w_familiar (weight on familiarization base rate).
- Parameters removed: None (retains alpha and lapse from rsa_l2).
- Other terms: The shared salience prior is passed to L0, L1, and L2, replacing uniform object weights in L0 choices and S1/S2 speaker priors; uninformative prior trials predict this salience prior rather than uniform choice.
