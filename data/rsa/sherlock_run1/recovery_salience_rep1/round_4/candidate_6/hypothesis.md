We refine prior_literal_baserate_listener by integrating focal visual attention from focal_attention_listener: listeners restrict their communicative reasoning primarily to candidate referents matching the heard utterance, downweighting irrelevant distractor objects in the display. When evaluating the communicative informativeness of alternative utterances in their mental model of the speaker, the speaker anticipates a Bayesian literal listener who evaluates candidate referents against this attention-weighted display while integrating experiential base rates and visual feature simplicity priors. This combined mechanism dampens exaggerated pragmatic inferences on complex visual displays while preserving prior-driven expectations for frequent or simple referents.

Differences from prior_literal_baserate_listener:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from prior_literal_baserate_listener).
- Parameters added: distractor_weight ~ Beta(1.0, 1.0), representing the visual attentional weight allocated to non-candidate display objects.
- Parameters removed: None (alpha, salience, baserate, and lapse retained unchanged).
- Functional terms changed:
  - In L0, added array parameter atten: ... and weighted literal listener referent selection by at(lex, u, r) * (vec(prior, r) + {EPS}) * (vec(atten, r) + {EPS}), replacing the uniform-attention Bayesian choice rule.
  - In L1, passed atten: ... through to L0.
  - In choice_probs, computed object attention weights as atten = jnp.where(is_cand > 0, 1.0, params["distractor_weight"]) where is_cand = jnp.where(ctx.is_prior > 0, 1.0, ctx.lex[ctx.utterance]), and passed atten into L1.
