Listeners interpret referring expressions at depth 2 by integrating empirical familiarization base rates with an intrinsic perceptual prior over referents based on visual feature complexity. Rather than assuming referents are equiprobable in the absence of exposure, listeners favor visually simpler objects with fewer features as their baseline expectation, updating from this perceptual prior toward observed exposure rates when familiarization occurs.

Refinement of base_rate_l2:
- Base model: base_rate_l2 (best).
- Recursion depth: depth 2 (calls L2 in choice_probs, unchanged from base_rate_l2).
- Parameters added: salience_weight (Normal(0.0, 1.0), sensitivity of the referent prior to visual feature count, taken from feature_salience_l2).
- Parameters removed: none (retains alpha, base_rate_weight, and lapse).
- Mathematical / structural changes: In choice_probs, the baseline prior over candidate objects is governed by visual feature complexity (softmax_prior(salience_weight * feature_count)) rather than an equiprobable uniform distribution. When familiarization base rates are available, listeners blend empirical frequencies with this feature-salience baseline rather than with a uniform baseline; when familiarization is absent, listeners rely on the feature-salience baseline directly.
