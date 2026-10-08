Emotional valence linearly modulates the common-knowledge feature salience prior: while people expect speakers in neutral or positive settings to refer to feature-rich salient objects, a speaker describing their least favorite object is expected to refer to simpler, less feature-rich objects. In Depth-1 RSA, this valence-dampened salience prior is shared between the literal listener and the pragmatic listener, flattening or reversing the prior expectation over objects when valence is negative.

Differences from source (rsa_l1_shared_prior):
- Recursion depth: Unchanged (choice_probs calls L1 at depth 1).
- Parameters added: w_valence (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs, the prior feature logit term params["w_features"] * ctx.feature_count is replaced by (params["w_features"] + params["w_valence"] * ctx.valence) * ctx.feature_count.
