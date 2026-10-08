Listeners select referents according to a power-law choice rule governed by a decision rationality parameter beta rather than strict probability matching. When interpreting referring expressions under depth-1 pragmatic reasoning with a shared salience prior, listeners' choice probabilities scale as the posterior referent probability raised to beta. This accommodates conservative decision stochasticity across forced-choice displays while preserving pragmatic inferences.

Differences from rsa_l1_shared_prior:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged).
- Parameters added: beta (prior: LogNormal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L1, listener chooses r with wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})) instead of wpp=Pr[speaker.r == r].
