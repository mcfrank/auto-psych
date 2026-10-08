People interpret referential expressions through depth-2 pragmatic reasoning with a shared salience prior: the listener inverts a speaker who reasons about a pragmatic listener, where visual salience (feature counts and familiarization history) is common ground that biases expectations across all levels of recursion. On prior trials without an informative word, listeners fall back directly on this shared salience distribution.

This model refines `rsa_l1_shared_prior` with a single change: increasing the recursion depth from depth 1 to depth 2.
Differences from `rsa_l1_shared_prior`:
- Recursion depth: `choice_probs` calls `L2` instead of `L1`.
- Parameters added or removed: None (retains `alpha`, `w_features`, `w_familiar`, and `lapse`).
- Other terms: Adds the level-2 pragmatic listener `L2[u, r]` inverting a level-2 speaker `S2` who evaluates utility against `L1[u, r]` with the shared salience prior, while `L0`, `L1`, the prior calculation, and the lapse mixture remain identical.
