Listeners apply a shared feature-salience prior during pragmatic reference resolution when interpreting spoken words, but fall back to unbiased uniform guessing when no communicative word is provided. In unprompted prior-elicitation settings, participants exhibit no intrinsic preference for feature-rich objects, so communicative salience only shapes choice when resolving an informative signal.

Differences from rsa_l1_shared_prior:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In choice_probs, responses on prior-elicitation trials (ctx.is_prior > 0) follow a uniform distribution across available referents (taken from rsa_l1) rather than the feature-count salience prior.
