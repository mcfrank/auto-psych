We refine rsa_l1_shared_prior by increasing listener reasoning to depth 2: listeners invert a speaker who simulates a depth-1 pragmatic listener rather than a literal listener, while maintaining the shared salience prior across all levels of recursive reasoning. This higher-order recursive inference allows listeners to resolve complex scalar implicatures that depth-1 reasoning fails to disambiguate, while preserving the grounding of referential expectations in mutual perceptual and familiarization salience.

Differences from rsa_l1_shared_prior:
- Recursion depth: choice_probs calls L2 instead of L1.
- Agent hierarchy: introduces a pragmatic speaker S2 who anticipates pragmatic listener L1, and an L2 listener who inverts S2, with the shared salience prior used at all levels (L0, S1, L1, S2, L2).
- Parameters: no parameters added or removed (retains alpha, w_features, w_familiar, and lapse).
