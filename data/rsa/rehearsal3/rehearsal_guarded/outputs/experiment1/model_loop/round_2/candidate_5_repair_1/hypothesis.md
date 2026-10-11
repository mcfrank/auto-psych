When interpreting referring expressions in visual reference games, listeners reason about speakers who actively penalize utterances with broad contextual extension while avoiding confusion with visually similar competitors. Pragmatic listeners at depth 2 invert this cost-sensitive, confusion-averse speaker, expecting that a speaker intending a referent with unique distinguishing features would avoid using an overly broad expression shared by many objects in the scene. Uninformative prior trials remain governed by visual singleton pop-out, contextual distinctiveness, and feature complexity.

Model refined: `r1_c4`.
Differences from source:
- Recursion depth: Unchanged (depth 2, calling L2 in choice_probs).
- Parameters added: `cost_weight` (prior Normal(0.0, 1.0)) weighting utterance extension cost in simulated speaker utilities.
- Parameters removed: None.
- Speaker utility: Extended at both simulated speaker tiers S1 and S2 to penalize words with wider contextual extension by subtracting utterance extension cost (`alpha * (log(L) - vec(cost, u)) - w_confusion * at(confusion, u, r)`), where cost is proportional to the fraction of display objects satisfying each feature (`cost_weight * (jnp.sum(real_lex, axis=-1) / real_lex.shape[-1])`).
- Referent prior: Unchanged (combining discrete singleton salience, continuous contextual distinctiveness, visual feature count modulated by framing valence, and color contrast).
