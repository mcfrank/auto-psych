We refine base_rate_simplicity_listener by introducing conservative, sublinear weighting of empirical familiarization base rates, positing that communicators exhibit base-rate conservatism when resolving competition between exposure frequency and referent simplicity. Rather than weighting empirical base rates linearly according to normative Bayesian multiplication, communicators scale exposure frequencies with a sublinear exponent parameter that regresses extreme familiarization disparities toward uniformity while preserving relative frequency ordering. On displays without familiarization, the base rate remains uniform, preserving the baseline feature-simplicity prior.

Differences from base_rate_simplicity_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: base_rate_weight (LogNormal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs scales the familiarization base-rate vector by raising it to the power of params["base_rate_weight"] and renormalizing it (when ctx.has_familiarization > 0) before point-wise multiplication with the feature-simplicity prior.
