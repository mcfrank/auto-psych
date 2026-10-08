Listeners interpret referring expressions using a perceptual distinctiveness heuristic whose direction is modulated by the speaker's communicative valence framing (taken from evaluative_prominence_listener). Under neutral or positive framing, listeners restrict attention to matching referents and select the object with lower contextual distinctiveness, penalizing extraneous unmentioned features. Under evaluative framing ('least favorite'), this heuristic inverts: listeners expect the speaker's least favorite object to possess conspicuous, distinguishing quirks, favoring matching referents that exhibit higher contextual distinctiveness.

Differences from distinctiveness_heuristic_listener:
- Refined model: distinctiveness_heuristic_listener
- Recursion depth: Depth 0 (calls L_heuristic, unchanged).
- Parameters added: w_valence (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs, the coefficient passed to L_heuristic is modulated by communicative framing (coeff = params["beta"] - params["w_valence"] * ctx.valence) rather than using a constant beta.
