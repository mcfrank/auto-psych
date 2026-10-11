RSA pragmatic listener at depth 2 with descriptive Question Under Discussion goal inference, competitor confusion avoidance, singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_confusion_singleton_l0 by incorporating the descriptive QUD component from descriptive_qud_listener: speakers balance discriminative referent identification against descriptive feature attribution, choosing between these communicative goals with a latent probability. Pragmatic listeners invert this goal-uncertain speaker at depth 2, inferring that when a speaker produces an ambiguous or ubiquitous word they were likely operating under a descriptive goal rather than uniquely identifying an object. This directly softens excessive pragmatic scalar exclusion and elevates selections of feature-rich referents on ubiquitous and two-object scalar displays, addressing posterior predictive underprediction of feature-rich selections while maintaining robust discriminative precision on distinctive displays.

Differences from rsa_l2_confusion_singleton_l0:
- Recursion depth remains depth 2 (choice_probs calls L2).
- Added parameter logit_p_ident (prior dist.Normal(0.0, 1.0)).
- Added global domain GOALS = jnp.arange(2) representing descriptive (0) and discriminative (1) communicative goals.
- In both S1 and S2, added a goal choice (speaker: chooses(goal in GOALS, wpp=p_ident if goal == 1 else (1.0 - p_ident))) and modulated the speaker utility by goal: goal * (alpha * log(L + {EPS}) - w_confusion * at(confusion, u, r)).
- In choice_probs, computed p_ident = jax.nn.sigmoid(params["logit_p_ident"]) and passed p_ident to L1 and L2.
- No parameters removed.
