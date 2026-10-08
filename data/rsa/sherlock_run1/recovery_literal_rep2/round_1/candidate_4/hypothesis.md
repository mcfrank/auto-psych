Depth-1 RSA with a shared salience prior whose feature weighting is modulated by speaker valence framing: when a speaker describes a disliked or least-favorite referent, people expect the speaker to favor objects with fewer features rather than more features. This refines rsa_l1_shared_prior to capture valence-dependent shifts in prior referent expectations while preserving mutual knowledge of salience between speaker and literal listener.

Differences from rsa_l1_shared_prior:
- Recursion depth: unchanged (choice_probs calls L1).
- Parameters: added w_valence ~ Normal(0, 1).
- Prior calculation: feature count weight is (w_features + w_valence * ctx.valence) instead of w_features alone.
- All other memo agents, distributions, and terms are unchanged.
