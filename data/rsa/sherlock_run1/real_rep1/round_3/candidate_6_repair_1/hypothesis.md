Listeners interpret referential and evaluative descriptions by combining depth-2 recursive pragmatic reasoning, utterance production costs, and graded semantic truth values with perceptual salience, refining graded_costly_valence_l2_listener by incorporating visual pop-out from multimodal_salience_listener into the common-knowledge salience prior. Before hearing an informative utterance, listeners assign higher prior prominence to chromatic objects contrasting with grayscale competitors alongside evaluative and feature-based priors, which communicative speakers share across reasoning depths while balancing informativeness against production costs over graded descriptions.

Refined model: graded_costly_valence_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: w_color (Normal(0.0, 2.0)); no parameters removed (retains alpha, w_features, w_familiar, w_valence, w_extension, beta_graded, lapse with identical priors).
- Terms changed: In choice_probs, the prior over objects incorporates perceptual color salience + params["w_color"] * (1.0 - ctx.grayscale).
