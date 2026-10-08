Listeners interpret referential expressions through first-order pragmatic reasoning (inverting a speaker who evaluates literal listener informativeness penalized by cognitive ambiguity costs), while evaluating candidate referents through an inductive prior that integrates visual feature complexity alongside empirical familiarization base rates. Bounding recursive mentalizing at depth 1 prevents the over-concentration of pragmatic inferences observed at second-order depth on complex displays, better matching human performance where listeners do not recursively simulate an inner pragmatic listener. On uninformative baseline displays and familiarization displays, empirical exposure frequencies and simplicity preferences directly guide baseline referent expectations.

### Refinement of salience_base_rate_ambiguity_l2
We refine salience_base_rate_ambiguity_l2 with a single stated change: reducing recursion depth from depth 2 to depth 1, bounding Theory-of-Mind reasoning to first-order pragmatic speaker simulation while preserving the inductive referent prior and speaker ambiguity cost mechanisms.

### Differences from salience_base_rate_ambiguity_l2
- Recursion depth: depth 1 (choice_probs calls L1, whereas salience_base_rate_ambiguity_l2 calls L2).
- Parameters added: none.
- Parameters removed: none (alpha, cost, base_rate_weight, salience, and lapse retained with identical priors).
- Memo recursion: removed L2; L1 retains its exact definition and signature L1[u: UTT, r: OBJ](alpha, costs: ..., lex: ..., prior: ...).
- Choice probabilities: choice_probs evaluates L1 directly instead of L2 for heard utterances (heard = L1(params["alpha"], costs, ctx.lex, prior)[ctx.utterance]), returning with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]).
