Listeners reason recursively at depth 2 under a shared prior that is common knowledge across all levels of pragmatic mental simulation. Rather than assuming lower-level conversational partners ignore visual complexity and familiarization base rates, listeners expect even first-order speakers to anticipate these prior referent biases, allowing prior expectations to shape pragmatic reasoning at every depth of mutual recursion. When no informative speech is heard, listeners choose directly according to this shared Bayesian prior.

Refinement of bayesian_base_rate_l2:
- Base model: bayesian_base_rate_l2
- Recursion depth: depth 2 (calls L2 in choice_probs, unchanged from bayesian_base_rate_l2).
- Parameters added: none.
- Parameters modified: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, and lapse).
- Mathematical / structural changes: In L1, speaker S1 is given the Bayesian referent prior (speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS})) rather than an unconditional uniform prior (wpp=1), and L1 takes prior as an argument. Consequently, speaker S2 simulates an L1 listener whose inferences already integrate the Bayesian referent prior over candidate objects, establishing a shared prior across the entire pragmatic hierarchy.
