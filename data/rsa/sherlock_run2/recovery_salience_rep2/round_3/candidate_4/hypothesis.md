Depth-2 pragmatic listeners incorporate visual attentional constraints into recursive speaker reasoning by attenuating non-matching distractors in the display, in addition to combining an inductive base-rate prior with speaker ambiguity costs. When simulating what the speaker would say to describe candidate referents, this attentional bottleneck discounts competitor objects that do not share the heard feature, reducing the perceived ambiguity of alternative expressions that overlap with unattended distractors. This mechanism explains why listeners show bounded pragmatic implicatures on complex displays with shared competitor features (such as target-complex displays in Mayn & Demberg) while maintaining empirical base-rate sensitivity and ambiguity-driven disambiguation.

### Refinement of base_rate_ambiguity_l2
We refine base_rate_ambiguity_l2 with a single stated change: extending depth-2 recursive reasoning with the distractor attenuation mechanism from distractor_attenuation_listener, discounting non-matching objects during speaker simulation.

### Differences from base_rate_ambiguity_l2
- Recursion depth: depth 2 (choice_probs calls L2, unchanged from base_rate_ambiguity_l2).
- Parameter added: distractor_weight (prior dist.Beta(2.0, 2.0)), which scales the attentional weight assigned to non-matching distractors during speaker simulation (taken from distractor_attenuation_listener); no parameters removed (alpha, cost, base_rate_weight, and lapse retained with identical priors).
- Distractor attenuation in memo recursion: added array parameter atten: ... to L0, L1, and L2; in L0, listener choice weights are modulated by vec(atten, r) (wpp = at(lex, u, r) * vec(atten, r)), and atten is forwarded through L2 to L1 and L0.
- Choice probabilities: choice_probs computes matches = ctx.lex[ctx.utterance] and atten = jnp.where(ctx.is_prior > 0, 1.0, jnp.where(matches > 0, 1.0, params["distractor_weight"])), and passes atten to L2.
