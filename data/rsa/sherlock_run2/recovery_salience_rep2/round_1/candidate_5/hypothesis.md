Depth-2 pragmatic listeners incorporate observed object base rates into their prior expectations over referents rather than assuming all objects are equally likely a priori. When familiarization frequencies are available, the listener assumes that speakers choose referents proportionally to these base rates and uses these base rates directly on trials without an informative word. On informative trials, listeners combine recursive pragmatic informativeness at depth 2 with this learned base-rate prior to disambiguate the speaker's intended referent.

### Refinement of rsa_l2
The incumbent model (rsa_l2) assumes a uniform prior over objects at all levels and guesses uniformly on uninformative prior trials, ignoring familiarization manipulations entirely. We refine rsa_l2 with a single stated change: replacing the uniform referent prior with an empirical base-rate prior scaled by a sensitivity parameter. This fits human behavior better because participants are strongly biased toward frequently seen objects on prior trials and integrate these base rates into their pragmatic inferences.

### Differences from rsa_l2
- Recursion depth: Unchanged at depth 2 (`choice_probs` calls `L2`).
- Parameters added: `base_rate_weight` with prior `dist.HalfNormal(1.0)`, governing sensitivity to familiarization base rates.
- Parameters removed: None (`alpha` and `lapse` retained with identical priors).
- Other terms: A prior vector over objects (`prior`) is computed from `ctx.familiarization` when `ctx.has_familiarization > 0` and defaults to uniform otherwise; `prior` is passed into `L1` and `L2` as the speaker's referent prior (`wpp=vec(prior, r)`), and returned directly on prior trials (`jnp.where(ctx.is_prior > 0, prior, heard)`).
