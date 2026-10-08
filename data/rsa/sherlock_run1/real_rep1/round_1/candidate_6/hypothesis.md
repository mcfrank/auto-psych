Listeners reason at depth 2 (inverting a speaker who simulates a depth-1 pragmatic listener), refining the rsa_l2 model by adopting a shared salience prior over objects across all simulated speaker and listener levels. This prior, formed from object feature counts and familiarization base rates, shapes both production and comprehension choices as common knowledge and directly governs choices when no informative word is spoken.

Refined model: rsa_l2
Differences from source:
- Recursion depth: depth 2 (calls L2 in choice_probs, unchanged).
- Parameters added: w_features (Normal(0.0, 1.0)) and w_familiar (Normal(0.0, 2.0)); no parameters removed.
- Terms changed: L0, L1, and L2 take prior as an argument; L0 weights objects by vec(prior, r) * at(lex, u, r); S1 and S2 sample objects with wpp=vec(prior, r); on prior trials, choice_probs returns prior instead of a uniform distribution.
