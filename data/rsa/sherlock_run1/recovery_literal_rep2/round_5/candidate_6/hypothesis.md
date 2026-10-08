This model refines literal_pragmatic_mixture_listener by incorporating the speaker's evaluative stance into the literal listener's feature economy heuristic. Listeners in reference games comprise a discrete mixture of literal and pragmatic reasoning types, where literal listeners penalize extraneous unmentioned features with an economy penalty that is dynamically modulated by the speaker's evaluative framing (amplified under positive framing and softened or inverted under negative framing), while pragmatic listeners infer communicative intent through recursive mental simulation of the speaker.

Differences from literal_pragmatic_mixture_listener:
- Refined model: literal_pragmatic_mixture_listener
- Recursion depth: depth 1 (choice_probs calls L_literal and L_pragmatic, unchanged).
- Parameters added: w_valence ~ Normal(0.0, 1.0) governing the modulation of the literal listener's feature economy penalty by speaker valence framing.
- Parameters removed: none.
- Feature economy penalty in choice_probs: the effective feature penalty for the literal listener is beta_eff = params["beta"] + params["w_valence"] * ctx.valence.
- Literal listener L_literal: takes beta_eff instead of params["beta"].
- All other memo agents, distributions, and prior terms are unchanged.
