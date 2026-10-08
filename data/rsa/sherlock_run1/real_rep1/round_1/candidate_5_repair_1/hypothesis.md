People interpret referential descriptions by inverting a speaker who balances communicative informativeness against utterance production costs, while maintaining a shared common-ground salience prior over referents. When a speaker uses an ambiguous or shared descriptor, listeners recognize that this preference reflects lower production costs for broad, accessible terms rather than an uninformative speaker. This refines rsa_l1_shared_prior by integrating utterance extension costs from costly_feature_speaker into the speaker's utility, improving predictions on displays where referents have both unique and shared features.

Differences from rsa_l1_shared_prior:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: w_extension (Normal(0.0, 1.0)); no parameters removed (retains alpha, w_features, w_familiar, lapse with identical priors).
- Speaker utility: In memo L1, speaker chooses utterances with utility augmented by vec(weights, u), where weights are w_extension * (1.0 - ctx.is_sink) * (jnp.sum(ctx.lex, axis=1) - 1.0).
