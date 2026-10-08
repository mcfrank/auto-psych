Listeners maintain an inductive prior over objects based on their feature count rather than assuming all referents are equally likely, favoring simpler objects with fewer features as default referents. This improves fit because participants exhibit a systematic preference for less modified objects when choosing referents and when encountering uninformative utterances, which uniform-prior depth-2 reasoning fails to capture.

Differences from rsa_l2:
- Recursion depth: unchanged (choice_probs calls L2, depth 2).
- Parameters added: feature_weight (Normal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes an object prior from feature counts via softmax_prior(feature_weight * ctx.feature_count); L1 and L2 pass this prior to the speaker (wpp=vec(prior, r) instead of wpp=1); and choice_probs uses this prior rather than a uniform distribution on uninformative prior trials (ctx.is_prior > 0).
