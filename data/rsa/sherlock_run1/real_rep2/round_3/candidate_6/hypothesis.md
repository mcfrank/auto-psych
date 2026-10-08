Listeners engage in depth-2 recursive pragmatic reasoning by inverting an informative speaker who simulates a pragmatic listener, while maintaining speaker-side utterance ambiguity costs and a shared visual salience prior grounded in diagnostic feature contrast rather than raw feature count. In visual reference displays, features unique to an object heighten its prior communicative salience while shared competitor features dilute it, which interlocutors treat alongside color distinctiveness and familiarization as common knowledge when resolving higher-order scalar implicatures. This fits better because diagnostic feature contrast captures the distinctiveness of competitor referents prior to communication, complementing both recursive pragmatic reasoning and speaker utterance costs.

Refinement of rsa_l2_costly_color_salience:
- Base model refined: rsa_l2_costly_color_salience
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from rsa_l2_costly_color_salience).
- Parameters added: w_contrast (prior: Normal(0.0, 1.0)).
- Parameters removed: w_features (retains alpha, w_familiar, w_color, cost_ambiguity, and lapse).
- Other terms: choice_probs computes the shared object salience prior from diagnostic feature contrast (unique non-sink features minus shared competitor features) rather than raw feature counts before the softmax (component taken from diagnostic_contrast_listener), while retaining sensitivity to perceptual color distinctiveness and familiarization base rates; the cost-sensitive speaker utilities at depth 1 and depth 2 are unchanged.
