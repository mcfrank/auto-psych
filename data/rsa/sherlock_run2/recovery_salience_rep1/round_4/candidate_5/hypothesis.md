We refine base_rate_simplicity_listener by extending it with the softmax decision rule component from softmax_decision_listener, positing that listeners evaluate referential statements by integrating pragmatic speaker reasoning with familiarization base rates and feature simplicity, but convert their resulting posterior beliefs into behavioral choices via a softmax decision rule rather than pure probability matching. Governed by a decision precision parameter, this response rule allows listeners to overmatch their posterior beliefs by sharpening preference toward the most probable referent under high confidence, or undermatch toward indifference under uncertainty. On uninformative prior trials, choices are similarly guided by this softmax decision rule applied to the integrated base-rate and simplicity prior.

Differences from base_rate_simplicity_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: decision_precision (LogNormal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs converts posterior beliefs into choice probabilities via jax.nn.softmax(params["decision_precision"] * jnp.log(beliefs + EPS)) before applying the lapse rate, sharpening or flattening choices relative to posterior beliefs.
