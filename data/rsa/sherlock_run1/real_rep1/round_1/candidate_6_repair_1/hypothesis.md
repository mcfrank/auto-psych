Listeners interpret evaluative descriptions by reasoning at depth 2 (inverting a speaker who anticipates a depth-1 pragmatic listener), refining the valence_salience_listener model by extending its recursive depth from depth 1 to depth 2 while preserving the common-knowledge evaluative salience prior across all levels of recursion. In complex reference games with evaluative framing (such as favorite vs. least favorite), depth-2 reasoning enables the speaker to select utterances based on pragmatic implicatures rather than just literal comprehension, resolving ambiguous references that depth-1 reasoning under-discriminates.

Refined model: valence_salience_listener
Differences from source:
- Recursion depth: depth 2 (choice_probs calls L2 instead of L1).
- Added memo L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...), where the listener inverts speaker S2 who chooses utterances according to L1's pragmatic recovery of the referent with the shared evaluative prior.
- Parameters added or removed: None (retains alpha, w_features, w_familiar, w_valence, lapse with identical priors).
- Terms changed: choice_probs calls L2 instead of L1.
