Listeners interpret referring expressions through direct feature contrast, selecting matching referents by penalizing additional unmentioned features against a communicative feature-salience prior, but fall back to unbiased uniform guessing when no communicative word is provided (taken from rsa_l1_referential_salience). In unprompted prior-elicitation settings, participants exhibit no intrinsic preference for feature-rich objects, so communicative salience only shapes choice when resolving an informative signal.

Differences from feature_contrast_listener:
- Recursion depth: Depth 0 (choice_probs calls L_contrast, unchanged).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In choice_probs, responses on prior-elicitation trials (ctx.is_prior > 0) follow a uniform distribution across available referents (taken from rsa_l1_referential_salience) rather than the feature-count salience prior.
