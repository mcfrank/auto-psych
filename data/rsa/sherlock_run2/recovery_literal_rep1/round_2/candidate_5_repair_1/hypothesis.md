We refine literal_salience_listener by elevating the listener from depth 0 to depth 1 to invert a naive, truthful speaker: rather than evaluating words in isolation, listeners model the speaker as randomly selecting one feature of the intended object, making boundedly rational softmax choices over candidate referents. Because an object with multiple features provides more descriptive options for the speaker, naming a shared feature provides weaker evidence for a complex referent than for a minimally described referent, naturally disfavoring over-specified objects without simulating counterfactual listeners. On prior trials with no informative word, choices are governed strictly by contextual object salience.

Differences from source (literal_salience_listener):
- Recursion depth: Changed from depth 0 (choice_probs calls L0) to depth 1 (choice_probs calls L1).
- Parameters added: beta (prior: LogNormal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: choice_probs calls L1 instead of L0. In L1, the listener simulates a naive speaker choosing truthfully among valid descriptors (`wpp = at(lex, u, r)`) given prior object distribution `vec(prior, r)`, and the listener chooses referents via softmax rationality `exp(beta * log(Pr[speaker.r == r] + {EPS}))`.
