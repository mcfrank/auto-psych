Listeners interpret referential and evaluative descriptions by combining depth-2 recursive pragmatic reasoning with utterance production costs in the speaker and softmax decision precision in the listener, refining softmax_valence_l2_listener by incorporating utterance extension costs into the speaker's utility. When a speaker chooses between accessible, shared terms and narrow descriptors, listeners recognize that broader terms reflect lower production costs rather than an uninformative speaker, resolving competition between specific and shared features across displays while sharpening their final referential choices toward the most probable referent. This unifies cost-sensitive speaker production with decision-rational listener comprehension across all recursive depths of the evaluative reference game.

Refined model: softmax_valence_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: w_extension (Normal(0.0, 1.0)); no parameters removed (retains alpha, beta, w_features, w_familiar, w_valence, lapse with identical priors).
- Terms changed: In memo L1 and L2, speakers choose utterances with utility augmented by vec(weights, u), where weights are w_extension * (1.0 - ctx.is_sink) * (jnp.sum(ctx.lex, axis=1) - 1.0).
