Listeners evaluate referential expressions through depth-3 recursive Theory-of-Mind reasoning (inverting a speaker S3 who simulates a depth-2 pragmatic listener L2) while incorporating empirical familiarization base rates into their inductive prior over candidate referents. Reasoning at depth 3 sharpens pragmatic disambiguation on complex implicature displays where alternative descriptors are themselves partially ambiguous, while preserving empirical base-rate preferences on uninformative baseline displays.

### Refinement of base_rate_prior_l2
We refine base_rate_prior_l2 with a single stated change: increasing recursion depth from depth 2 to depth 3, allowing listeners to resolve higher-order communicative implicatures while maintaining empirical base-rate integration.

### Differences from base_rate_prior_l2
- Recursion depth: depth 3 (choice_probs calls L3, whereas base_rate_prior_l2 calls L2).
- Parameters added: none.
- Parameters removed: none (alpha, base_rate_weight, and lapse retained with identical priors).
- Other terms: introduced L3[u: UTT, r: OBJ](alpha, lex: ..., prior: ...) where a depth-3 speaker chooses utterances proportional to communicative informativeness under L2 (wpp = at(lex, u, r) * exp(alpha * log(L2[u, r](alpha, lex, prior) + {EPS}))), and choice_probs evaluates L3 instead of L2 for heard utterances.
