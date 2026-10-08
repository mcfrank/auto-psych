We refine focal_prior_literal_baserate_listener by replacing stimulus-independent uniform lapse with a prior-anchored lapse fallback: when participants experience inattention or task disengagement, their choices fall back on their intuitive baseline prior expectations (favoring visual simplicity and familiarization base rates) rather than choosing uniformly at random across all display objects. On uninformative trials, choices reflect prior expectations directly without artificial flattening from uniform noise, while on communicative trials, lapse errors are naturally biased toward simpler or familiar candidates rather than cluttered distractors.

Differences from focal_prior_literal_baserate_listener:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from focal_prior_literal_baserate_listener).
- Parameters added: None (alpha, salience, baserate, distractor_weight, and lapse retained unchanged).
- Parameters removed: None.
- Functional terms changed:
  - In choice_probs, replaced uniform lapse mixing with prior-anchored lapse fallback, computing choice probabilities as (1.0 - params["lapse"]) * jnp.where(ctx.is_prior > 0, prior, heard) + params["lapse"] * prior instead of with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]).
