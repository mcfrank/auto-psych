Listeners combine pragmatic reference resolution under an integrated Bayesian salience and base-rate prior and selective distractor suppression with a soft-maximizing decision rule over posterior beliefs. Rather than matching choice probabilities directly to posterior beliefs, an internal decision determinism sharpens subjective beliefs into action, concentrating choices on the most probable referent more decisively than normative probability matching predicts. On uninformative trials without speech, this decision determinism similarly sharpens choices under the integrated Bayesian prior.

Refinement of distractor_bayesian_salience_listener:
- Base model: distractor_bayesian_salience_listener (best).
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from distractor_bayesian_salience_listener).
- Parameters added: decision_determinism (LogNormal(0.0, 1.0), response determinism / power parameter sharpening posterior beliefs into choice probabilities, taken from belief_softmax_listener).
- Parameters modified: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, distractor_attention, and lapse).
- Mathematical / structural changes: In choice_probs, after forming posterior beliefs over candidate referents (jnp.where(ctx.is_prior > 0, prior, heard)), listeners apply a soft-maximizing response rule (jax.nn.softmax(params["decision_determinism"] * jnp.log(belief + EPS))) before lapses, rather than matching choice probabilities linearly to posterior beliefs.
