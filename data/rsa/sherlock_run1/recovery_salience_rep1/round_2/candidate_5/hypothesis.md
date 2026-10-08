We refine feature_salience_listener by adding softmax decision rationality to the listener's referent choice rule, separating the listener's choice decisiveness from the speaker's communicative rationality. While listeners retain a feature-complexity salience prior favoring simpler referents, they do not merely probability-match their posterior beliefs; instead, they choose the referent with highest posterior support with greater decisiveness in forced-choice selection.

Differences from feature_salience_listener:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from feature_salience_listener).
- Parameters added: beta ~ LogNormal(0.0, 1.0), representing listener decision rationality.
- Parameters removed: None (alpha, salience, and lapse retained unchanged).
- Functional terms changed:
  - In L1, listener choice weights are exponentiated by beta times the log posterior probability of each referent given the utterance, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})), rather than matching posterior probabilities directly.
  - In choice_probs, params["beta"] is passed to L1. On prior trials, choices remain governed by the feature salience prior.
