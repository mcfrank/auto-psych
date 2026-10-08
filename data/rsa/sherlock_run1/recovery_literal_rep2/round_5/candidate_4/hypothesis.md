This model refines valence_feature_economy_listener by incorporating Weber-Fechner diminishing marginal sensitivity into the feature economy heuristic: listeners penalize candidate referents possessing extraneous unmentioned features, but the psychological cost of complexity scales logarithmically rather than linearly with feature count. The first extraneous feature incurs the sharpest penalty for departing from minimality, whereas each additional extraneous feature adds diminishing marginal cost. Evaluative valence framing continues to modulate this diminishing feature penalty, shifting or inverting the penalty under negative framing.

Differences from valence_feature_economy_listener:
- Refined model: valence_feature_economy_listener
- Recursion depth: depth 0 (choice_probs calls L_diminishing, unchanged).
- Parameters added: none.
- Parameters removed: none.
- Functional form in L_diminishing: listeners weight candidate referents proportional to at(lex, u, r) * exp(-beta_eff * log(1.0 + vec(extraneous, r))) rather than at(lex, u, r) * exp(-beta_eff * vec(feature_count, r)), replacing linear feature penalization with a logarithmic Weber-Fechner functional form.
- Extraneous features in choice_probs: computed as jnp.maximum(0.0, ctx.feature_count - 1.0) and passed into L_diminishing.
- All other memo agents, distributions, and prior terms are unchanged.
