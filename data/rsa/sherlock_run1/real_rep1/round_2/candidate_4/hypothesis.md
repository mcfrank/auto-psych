Listeners interpret referential and evaluative descriptions by reasoning at depth 2 while inverting a speaker who balances communicative informativeness against utterance production costs, refining the valence_salience_l2_listener model by incorporating utterance extension costs into the speaker's utility. When a speaker uses a broader or shared descriptor, listeners recognize that this choice reflects lower production costs for accessible terms rather than an uninformative speaker, resolving competition between specific and shared features across displays while maintaining the evaluative salience prior across recursive depths.

Refined model: valence_salience_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: w_extension (Normal(0.0, 1.0)); no parameters removed (retains alpha, w_features, w_familiar, w_valence, lapse with identical priors).
- Terms changed: In memo L1 and L2, speakers choose utterances with utility augmented by vec(weights, u), where weights are w_extension * (1.0 - ctx.is_sink) * (jnp.sum(ctx.lex, axis=1) - 1.0).
