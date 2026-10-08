Listeners focus visual attention on candidate referents matching the uttered description while attenuating non-matching distractor objects, but select referents using a power-law decision rule governed by a decision rationality parameter rather than strict probability matching (taken from rsa_l1_rational_referential_salience). When simulating the speaker within the attended display, listeners moderate extreme inferential probabilities across forced-choice displays through this decision exponent, preventing shared features of distractor objects from distorting reference resolution while accommodating sub-rational choice stochasticity.

Differences from attended_display_listener:
- Refined model: attended_display_listener
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged).
- Parameters added: beta (prior: dist.LogNormal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L1, listener chooses r with wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})) instead of wpp=Pr[speaker.r == r].
