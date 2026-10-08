This model refines rsa_l1_salience by incorporating visual color contrast into the listener's salience prior over objects. Listeners infer referents under depth-1 pragmatic reasoning with an object prior that favors visually prominent colored objects over grayscale ones, in addition to feature count and familiarization.

Differences from rsa_l1_salience:
- Refined model: rsa_l1_salience
- Recursion depth: Depth 1 (L1, unchanged)
- Parameters added: w_color (prior: Normal(0.0, 1.0))
- Parameters removed: None
- Terms changed: In choice_probs, added params["w_color"] * (1.0 - ctx.grayscale) to the object salience logits.
