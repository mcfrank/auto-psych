Listeners interpret referential and evaluative descriptions by combining depth-2 recursive pragmatic reasoning with utterance extension costs and a softmax choice rule over inferred referents, refining costly_valence_l2_listener by allowing listeners to maximize choices rather than strictly probability-match. While listeners invert a cost-sensitive pragmatic speaker to infer the posterior probability of each referent, human participants exhibit decision precision that sharpens their final selections toward the most likely target object. Integrating a listener decision rationality parameter into the depth-2 cost-sensitive architecture better captures the heightened determinism of participant choices across unambiguous communicative displays.

Refined model: costly_valence_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: beta (LogNormal(0.0, 1.0)); no parameters removed (retains alpha, w_features, w_familiar, w_valence, w_extension, lapse with identical priors).
- Terms changed: In memo L2, the pragmatic listener makes referential choices via a softmax decision rule with rationality parameter beta, choosing with weight wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})) instead of probability matching with wpp=Pr[speaker.r == r].
