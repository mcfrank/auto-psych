Listeners interpret evaluative descriptions by combining depth-2 recursive pragmatic reasoning with a softmax choice rule over inferred referents, refining the valence_salience_l2_listener model by allowing listeners to maximize choices rather than strictly probability-match. While standard depth-2 listeners invert a pragmatic speaker to compute posterior probabilities over objects, human participants exhibit decision precision that sharpens their selections toward the most probable referent. Incorporating a decision rationality parameter for the listener into the depth-2 evaluative architecture better captures the high determinism of participant choices on clear communicative displays.

Refined model: valence_salience_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: beta (LogNormal(0.0, 1.0)); no parameters removed (retains alpha, w_features, w_familiar, w_valence, lapse with identical priors).
- Terms changed: In memo L2, the pragmatic listener makes referential choices via a softmax decision rule with rationality parameter beta, choosing with weight wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})) instead of probability matching with wpp=Pr[speaker.r == r].
