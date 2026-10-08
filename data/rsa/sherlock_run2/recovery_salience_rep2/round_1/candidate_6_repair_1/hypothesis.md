Listeners evaluate candidate referents through a feature-complexity prior that favors simpler objects with fewer defining features, and reason recursively at depth 2 by inverting a speaker who simulates a depth-1 pragmatic listener. Rather than stopping at first-order pragmatic reasoning as in feature_salience_listener, depth-2 reasoning enables listeners to resolve complex higher-order implicatures while simultaneously incorporating cognitive parsimony expectations across both informative utterances and uninformative baseline trials.

Differences from feature_salience_listener:
- Recursion depth: depth 2 (choice_probs calls L2, whereas feature_salience_listener calls L1).
- Parameters added: none.
- Parameters removed: none (alpha, salience, and lapse are retained with identical priors).
- Other terms: introduced L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...) where a depth-2 speaker chooses utterances proportional to communicative informativeness under L1; L1 passes the feature-complexity prior to the base speaker; choice_probs evaluates L2 instead of L1 for heard utterances.
