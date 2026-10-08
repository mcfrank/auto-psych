Listeners interpret referential descriptions by reasoning at depth 2 (inverting a speaker who anticipates a depth-1 pragmatic listener), refining costly_speaker_shared_prior by extending its recursive depth from depth 1 to depth 2 while preserving utterance production costs and the shared common-ground salience prior across all levels of recursion. In complex displays with both shared and distinctive features, depth-2 reasoning enables the speaker to select utterances based on pragmatic implicatures rather than just literal comprehension, while continuing to balance communicative informativeness against the production cost of overly specific terms.

Refined model: costly_speaker_shared_prior
Differences from source:
- Recursion depth: depth 2 (choice_probs calls L2 instead of L1).
- Added memo L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., weights: ...), where the listener inverts speaker S2 who chooses utterances according to L1's pragmatic recovery of the referent with the shared salience prior and utterance extension costs.
- Parameters added or removed: None (retains alpha, w_features, w_familiar, w_extension, lapse with identical priors).
- Terms changed: choice_probs calls L2 instead of L1.
