This model refines valence_salience_listener by adding softmax decision rationality to the pragmatic listener: rather than strictly probability-matching their posterior beliefs about the speaker's intended referent, listeners select referents according to a softmax choice rule governed by an independent decision rationality parameter. This separates the speaker's communicative rationality from the listener's choice temperature, capturing systematic conservatism in human referent selection while preserving the evaluative stance framing where speaker valence modulates prior expectations over objects.

Differences from valence_salience_listener:
- Refined model: valence_salience_listener
- Recursion depth: depth 1 (choice_probs calls L1, unchanged).
- Parameters added: beta ~ LogNormal(0.0, 1.0) governing listener decision rationality.
- Parameters removed: none.
- Listener choice in L1: listener chooses r in OBJ with probability proportional to exp(beta * log(Pr[speaker.r == r] + EPS)) instead of matching Pr[speaker.r == r] directly.
- All other memo agents, distributions, and prior terms are unchanged.
