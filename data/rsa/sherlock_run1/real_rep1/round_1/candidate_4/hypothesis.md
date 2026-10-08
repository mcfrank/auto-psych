Listeners reason at depth 2 by inverting a speaker who anticipates the depth-1 pragmatic listener, maintaining the common-knowledge salience prior (feature count and familiarization base rate) across all levels of recursion. Reasoning two levels deep allows the speaker to select utterances based on pragmatic implicatures rather than just literal comprehension, resolving ambiguous references in complex displays where depth-1 reasoning under-discriminates.

Differences from rsa_l1_shared_prior:
- Recursion depth: choice_probs calls L2 instead of L1.
- Added memo L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...), where the listener inverts speaker S2 who chooses utterances according to L1's pragmatic recovery of the referent with common-knowledge prior over objects.
- Parameters added or removed: None (retains alpha, w_features, w_familiar, lapse with identical priors).
