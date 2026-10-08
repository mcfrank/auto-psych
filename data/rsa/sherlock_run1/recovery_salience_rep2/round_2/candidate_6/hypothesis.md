Listeners combine pragmatic reference resolution with an integrated object prior that incorporates both visual feature complexity and empirical familiarization base rates. When objects have prior exposure frequencies, listeners blend those base rates with their perceptual feature-salience expectations, using the resulting prior to interpret referring expressions and guide choices when speech is absent.

Refinement of feature_salience_listener:
- Base model: feature_salience_listener
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from feature_salience_listener).
- Parameters added: base_rate_weight (Beta(2.0, 2.0), prior weight blending familiarization base rates with the feature-salience baseline).
- Parameters removed: none (retains alpha, salience_weight, and lapse).
- Mathematical / structural changes: In choice_probs, when familiarization exposure is present (has_familiarization > 0), the referent prior blends empirical base rates with the visual feature-salience prior ((1.0 - base_rate_weight) * salience_prior + base_rate_weight * familiarization) rather than relying on feature salience alone across all displays; on uninformative prior trials, listeners choose directly according to this integrated prior.
