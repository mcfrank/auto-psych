We refine fewest_features_listener by proposing that listeners balance descriptive parsimony against visual perceptual prominence: when interpreting an utterance, listeners penalize feature complexity among matching referents while favoring visually prominent referents shown in full color over desaturated grayscale objects. When no informative word is heard, listeners have no descriptive cues to evaluate and choose uniformly among candidate objects, modulated only by familiarization base rates.

Differences from source (fewest_features_listener):
- Recursion depth: Unchanged (choice_probs calls L0 at depth 0).
- Parameters added: w_color (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs, the referent choice weight incorporates a perceptual prominence boost for colored objects: `weight = jnp.exp(-params["beta"] * ctx.feature_count + params["w_color"] * (1.0 - ctx.grayscale))`.
