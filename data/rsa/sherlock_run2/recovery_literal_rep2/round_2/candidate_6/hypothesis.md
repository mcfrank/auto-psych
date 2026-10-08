Listeners evaluate candidate referents using a power-law choice rule governed by a decision rationality parameter rather than strict probability matching, while maintaining uniform guessing when no communicative word is given. When interpreting referring expressions under depth-1 pragmatic reasoning with a shared feature-salience prior, listeners' choice probabilities scale with the posterior referent probability raised to this decision exponent. This captures sub-rational decision stochasticity across forced-choice displays while preventing uninformative prior contexts from distorting communicative reference resolution.

Differences from rsa_l1_referential_salience:
- Refined model: rsa_l1_referential_salience
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged).
- Parameters added: beta (prior: dist.LogNormal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L1, listener chooses r with wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})) instead of wpp=Pr[speaker.r == r].
