Listeners operate under visual attentional constraints that attenuate non-matching distractors during speaker simulation, while additionally incorporating empirical familiarization base rates into referent expectations both when simulating the speaker and as their default choice on uninformative baseline displays. By combining distractor attenuation with inductive base-rate expectations, listeners account for prior object exposure frequencies without sacrificing attentional discounting of non-matching visual competitors on communicative displays.

### Refinement of distractor_attenuation_listener
We refine distractor_attenuation_listener with a single stated change: extending the model with an empirical base-rate prior over candidate referents (from base_rate_prior_l2) rather than assuming objects are equally likely a priori.

### Differences from distractor_attenuation_listener
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from distractor_attenuation_listener).
- Parameter added: base_rate_weight (prior dist.Normal(0.0, 2.0)), which scales sensitivity to familiarization base rates; no parameters removed (alpha, distractor_weight, and lapse retained with identical priors).
- Prior in memo recursion: added array parameter prior: ... to L1, replacing speaker: given(r in OBJ, wpp=1) with speaker: given(r in OBJ, wpp=vec(prior, r)).
- Choice probabilities: choice_probs computes prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization), passes prior to L1, and uses prior instead of uniform on prior trials (jnp.where(ctx.is_prior > 0, prior, heard)).
