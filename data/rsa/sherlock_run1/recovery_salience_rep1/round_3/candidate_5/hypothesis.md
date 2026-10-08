We refine baserate_salience_listener by integrating the referent prior directly into the literal listener's interpretation rule, as posited in classical Bayesian Rational Speech Act theory (Frank & Goodman, 2012). Rather than simulating a naive literal listener who evaluates candidate referents with uniform baseline probabilities, the speaker anticipates a listener who already weights candidate objects by their experiential base rates and visual feature complexity. This prior-informed literal interpretation sharpens the speaker's communicative incentives, driving speakers to choose distinctive expressions for rare or complex objects while relying on listeners' prior expectations for frequent, simple referents.

Differences from baserate_salience_listener:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from baserate_salience_listener).
- Parameters added: None (alpha, salience, baserate, and lapse retained unchanged).
- Parameters removed: None.
- Functional terms changed:
  - In L0, the listener's referent choice weight incorporates the referent prior alongside literal truth conditions, wpp=at(lex, u, r) * (vec(prior, r) + {EPS}), replacing the prior-agnostic choice rule.
  - In L1, the referent prior vector prior: ... is passed into L0.
