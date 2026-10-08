Listeners interpret referential descriptions with bounded cognitive resources whose reasoning precision decays as visual display clutter increases, refining distinctive_multimodal_l2_listener by incorporating capacity limitation from capacity_limited_listener into the depth-2 pragmatic reasoning architecture. While listeners maintain visual distinctiveness, evaluative framing, color pop-out, and graded semantics, their pragmatic rationality follows a power law decay with display set size, enabling sharp, decisive pragmatic inferences on compact two-object displays but producing more diffuse, bounded choices in larger, cluttered visual scenes.

Refined model: distinctive_multimodal_l2_listener
Differences from source:
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from source).
- Parameters added: alpha_base (LogNormal(0.0, 1.0)), gamma (Beta(1.0, 1.0)).
- Parameters removed: alpha. (Retains w_features, w_familiar, w_valence, w_color, w_distinct, w_extension, beta_graded, lapse with identical priors).
- Terms changed: In choice_probs, the rationality parameter alpha passed to L2 is modulated by display set size via alpha = params["alpha_base"] * (2.0 / ctx.lex.shape[1]) ** params["gamma"] instead of a constant scalar alpha.
