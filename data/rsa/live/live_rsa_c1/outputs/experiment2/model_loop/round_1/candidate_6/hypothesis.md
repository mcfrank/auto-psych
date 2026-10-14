Pragmatic listeners reason at depth 1 about a noisy-channel speaker under common knowledge of a referent prior anchored by visual endpoint extremity. Refining noisy_channel_speaker_l1 by incorporating visual endpoint extremity into the referent prior: visual attention and communicative expectations are concentrated at the extreme ends of the scene's complexity spectrum—the bare baseline contrast and the maximally elaborated focal exemplar—while intermediate-complexity objects suffer perceptual compression. This endpoint salience directly resolves the model's severe overprediction of intermediate objects on mumble trials and cushions inferences on ambiguous words, while transmission noise accounts for stochastic feature articulation errors.

Differences from source (noisy_channel_speaker_l1):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: w_extremity (prior: dist.Normal(0.0, 1.0)), weighting the visual endpoint extremity of referents along the scene's complexity continuum.
- Parameters removed: none.
- Other terms: In the referent prior, an endpoint extremity term extremity = jnp.abs(ctx.feature_count - 0.5 * (jnp.min(ctx.feature_count) + jnp.max(ctx.feature_count))) is added to the log-prior weights, favoring extreme endpoints over intermediate items.
