Speakers in reference games evaluate communicative utility trade-offs with utterance extension costs proportional to the contextual extension of each feature, actively penalizing broad, non-discriminating expressions that apply to many objects in the visual scene. Pragmatic listeners at depth 2 invert this cost-sensitive speaker, reasoning that an intended referent with unique features would have elicited a specific rather than an overly broad description, which curtails unwarranted singleton choices when a heard word applies indiscriminately to all objects. On uninformative prior trials, choices are governed directly by visual singleton pop-out, distinctiveness, and feature complexity.

Model refined: `rsa_l2_singleton_feat_color_valence_l0`.
Differences from source:
- Recursion depth: Unchanged (depth 2, calling L2 in choice_probs).
- Parameters added: `cost_weight` (prior Normal(0.0, 1.0)) weighting utterance extension cost in simulated speaker utilities.
- Parameters removed: None.
- Speaker utility: Extended at both simulated speaker tiers S1 and S2 to penalize words with wider contextual extension (`- vec(cost, u)`), where cost is proportional to the fraction of display objects satisfying each feature (`cost_weight * (jnp.sum(real_lex, axis=-1) / real_lex.shape[-1])`).
- Referent prior: Unchanged (combining discrete singleton salience, continuous distinctiveness, feature count modulated by framing valence, and color contrast).
