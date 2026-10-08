Depth-2 pragmatic listeners evaluate candidate referents through an inductive prior integrating visual feature complexity and empirical familiarization base rates, while expecting speakers to incur a cognitive cost for ambiguous utterances, but additionally operate under visual attentional constraints that attenuate non-matching distractors during speaker simulation. When simulating alternative utterances the speaker could produce, listeners attend primarily to referents matching the heard word, discounting competitor objects that do not share the focal feature. This attentional bottleneck prevents listeners from overestimating competitor ambiguity on complex visual displays while preserving empirical base-rate sensitivity and cognitive parsimony.

### Refinement of salience_base_rate_ambiguity_l2
We refine salience_base_rate_ambiguity_l2 with a single stated change: extending depth-2 recursive reasoning with the distractor attenuation mechanism from distractor_attenuation_listener, discounting non-matching objects during recursive speaker simulation.

### Differences from salience_base_rate_ambiguity_l2
- Recursion depth: depth 2 (choice_probs calls L2, unchanged from salience_base_rate_ambiguity_l2).
- Parameter added: distractor_weight (prior dist.Beta(2.0, 2.0)), which scales the attentional weight assigned to non-matching distractors during speaker simulation (taken from distractor_attenuation_listener); no parameters removed (alpha, cost, base_rate_weight, salience, and lapse retained with identical priors).
- Distractor attenuation in memo recursion: added array parameter atten: ... to L0, L1, and L2; in L0, listener choice weights are modulated by vec(atten, r) (wpp = at(lex, u, r) * vec(atten, r)), and atten is forwarded through L2 to L1 and L0.
- Choice probabilities: choice_probs computes matches = ctx.lex[ctx.utterance] and atten = jnp.where(ctx.is_prior > 0, 1.0, jnp.where(matches > 0, 1.0, params["distractor_weight"])), and passes atten to L2.
