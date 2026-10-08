Listeners interpret referring expressions through proportional feature contrast, evaluating candidate referents by penalizing unmentioned features relative to each object's total feature complexity, but modulate this contrast evaluation according to communicative valence framing (taken from evaluative_prominence_listener). Under neutral or positive framing, candidate referents are penalized for omitted features because speakers are expected to provide distinguishing descriptions; under evaluative framing ('least favorite'), unmentioned features act as compounding negative attributes that reverse this penalty to favor more complex referents.

Differences from proportional_feature_contrast:
- Refined model: proportional_feature_contrast
- Recursion depth: Depth 0 (calls L_contrast, unchanged).
- Parameters added: w_valence (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs, the coefficient on proportional unmentioned features is modulated by communicative framing (coeff = -params["theta"] - params["w_valence"] * ctx.valence, taken from evaluative_prominence_listener) rather than using a constant negative penalty -params["theta"].
