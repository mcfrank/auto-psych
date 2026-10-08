This model refines fewest_features_listener by incorporating the speaker's evaluative stance into the feature economy heuristic: listeners penalize candidate referents that possess extraneous unmentioned features, but modulate this penalty according to the speaker's evaluative framing. Under neutral framing, listeners apply the default penalty favoring minimally specified referents; when the speaker describes a 'favorite' object, extraneous features are penalized more heavily or expected to align with prominence, whereas under negative framing ('least favorite'), the evaluative polarity shifts, softening or inverting the penalty on extraneous features.

Differences from fewest_features_listener:
- Refined model: fewest_features_listener
- Recursion depth: depth 0 (choice_probs calls L_heuristic, unchanged).
- Parameters added: w_valence ~ Normal(0.0, 1.0) governing the modulation of feature economy by speaker valence framing.
- Parameters removed: none.
- Feature economy penalty in choice_probs: the effective feature penalty is beta_eff = params["beta"] + params["w_valence"] * ctx.valence.
- Heuristic listener L_heuristic: takes beta_eff instead of params["beta"].
- All other memo agents, distributions, and prior terms are unchanged.
