Listeners reason at depth 2 (inverting a speaker who simulates the depth-1 pragmatic listener), but rather than assuming all referents are equally likely a priori, agents maintain a feature-salience prior over objects that scales with each object's feature count. When no informative word is heard, listeners choose referents according to this feature-based prior, and when an utterance is heard, pragmatic reasoning updates this prior. This captures the cognitive tendency for simpler, less cluttered objects to serve as default prototypes or baselines in reference games.

Differences from rsa_l2:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from rsa_l2).
- Parameters added: w_feature ~ Normal(0.0, 1.0) weighting object feature counts.
- Parameters removed: None (alpha and lapse retained unchanged).
- Functional terms changed:
  - Computed prior vector in choice_probs as softmax_prior(params["w_feature"] * ctx.feature_count) and passed it to L2.
  - In L1 and L2, added array parameter prior: ... and changed speaker referent weight from wpp=1 to wpp=vec(prior, r).
  - In choice_probs, replaced uniform guessing on prior trials with prior: jnp.where(ctx.is_prior > 0, prior, heard).
