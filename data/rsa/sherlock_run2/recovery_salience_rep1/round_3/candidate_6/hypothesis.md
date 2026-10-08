We refine selective_attention_listener by extending it with the feature-simplicity prior component from feature_simplicity_listener, positing that communicators combine selective visual attention toward utterance-compatible referents with an inductive preference for simpler referents. When interpreting an utterance, the speaker in L1 evaluates candidate referents weighted by their feature simplicity rather than uniformly, while the literal listener L0 discounts unattended distractors; in the absence of an informative utterance, the listener chooses according to this feature-simplicity prior rather than guessing uniformly.

Differences from selective_attention_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: simplicity (Normal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes a referent simplicity prior via softmax_prior(params["simplicity"] * ctx.feature_count), passes it to L1 where the speaker samples referents with wpp=vec(prior, r) + {EPS} (instead of wpp=1), and returns this prior on uninformative prior trials (ctx.is_prior > 0) instead of a uniform distribution.
