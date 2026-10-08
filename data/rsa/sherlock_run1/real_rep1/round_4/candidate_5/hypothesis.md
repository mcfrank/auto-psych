Listeners interpret referential and evaluative descriptions by combining depth-2 recursive pragmatic reasoning, utterance production costs, graded semantics, and perceptual salience with visual distinctiveness, refining multimodal_graded_costly_l2_listener by incorporating feature distinctiveness from distinctiveness_salience_listener into the common-knowledge salience prior. Before hearing an informative utterance, listeners assign higher prior prominence to distinctive objects possessing unique or narrowly shared features that contrast with the visual context, resolving competition between singleton referents and identical competitors while maintaining evaluative, color, and complexity priors across reasoning depths.

Refined model: multimodal_graded_costly_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: w_distinct (Normal(0.0, 1.0)); no parameters removed (retains alpha, w_features, w_familiar, w_valence, w_color, w_extension, beta_graded, lapse with identical priors).
- Terms changed: In choice_probs, the prior over objects incorporates feature distinctiveness + params["w_distinct"] * distinctiveness, where distinctiveness is the sum across non-sink features true of each object weighted by specificity (inverse feature frequency).
