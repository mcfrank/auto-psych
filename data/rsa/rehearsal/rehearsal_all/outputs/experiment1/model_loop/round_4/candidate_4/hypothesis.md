Listeners convert their pragmatic referent beliefs into physical choices via a decisive softmax decision rule rather than passive probability matching. When resolving referential ambiguity, people act decisively by amplifying their highest-confidence referent hypothesis over weaker alternatives in proportion to an internal decision rationality. Pragmatic listeners invert a context-normalized confusion-averse speaker at depth two while combining shared perceptual salience, empirical base rates, and decisive referent selection.

Refining rsa_l2_mean_confusion_fam_l0 by incorporating the softmax decision rule from softmax_belief_listener into the final referent selection stage. This should fit better by allowing listeners to act decisively rather than strictly probability-matching their posterior beliefs, better capturing the concentrated distribution of human referent clicks on high-probability choices across displays while preserving depth-two recursive reasoning, context-normalized competitor-confusion aversion, perceptual salience, and empirical base-rate sensitivity.

Differences from source (rsa_l2_mean_confusion_fam_l0):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: beta ~ LogNormal(0.0, 1.0) governing listener decision rationality.
- Parameters removed: None.
- Terms changed: Exactly one term changed at the decision stage: decision = jax.nn.softmax(params["beta"] * jnp.log(belief + EPS)) where belief = jnp.where(ctx.is_prior > 0, prior, heard). The prior computation, competitor confusion matrix calculation, literal L0 semantics, S1 and S2 speaker choice utilities, depth-2 recursive reasoning, and lapse process remain identical to the source.
