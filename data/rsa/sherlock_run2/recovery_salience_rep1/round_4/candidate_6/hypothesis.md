We refine costly_base_rate_simplicity by replacing linear base-rate integration with sublinear base-rate conservatism, positing that communicators regress empirical familiarization frequencies toward uniformity rather than matching extreme exposure odds linearly. Governed by a base-rate sensitivity exponent, listeners discount extreme familiarization disparities while preserving feature simplicity preferences and speaker ambiguity avoidance, properly balancing pragmatic informativeness against prior exposure. In the absence of an informative utterance, listeners choose directly according to this conservatively weighted base-rate and simplicity prior.

Differences from costly_base_rate_simplicity:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: base_rate_weight (LogNormal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs raises empirical familiarization frequencies to the power base_rate_weight before normalization when familiarization is present (raw_base_rate = (ctx.familiarization + EPS) ** params["base_rate_weight"], normalized to sum to 1), replacing linear base-rate weighting with sublinear conservatism.
