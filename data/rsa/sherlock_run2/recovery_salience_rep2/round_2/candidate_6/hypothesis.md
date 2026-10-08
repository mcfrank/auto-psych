Listeners evaluate referential expressions through depth-2 recursive Theory-of-Mind reasoning (inverting a speaker who simulates a pragmatic listener) while assuming that speakers penalize ambiguous features that apply to multiple referents in the visual display. By accounting for this distractor-dependent ambiguity cost across second-order recursive speaker simulation, depth-2 listeners expect speakers to actively avoid shared descriptions, providing stronger pragmatic evidence in favor of referents lacking alternative distinguishing features. Compared to depth-1 ambiguity-sensitive reasoning, depth-2 recursive simulation allows listeners to resolve higher-order implicatures where multiple objects share a descriptor but differ in their remaining features.

### Refinement of ambiguity_cost_speaker
We refine ambiguity_cost_speaker with a single stated change: increasing recursion depth from depth 1 to depth 2, allowing listeners to resolve higher-order implicatures while preserving the speaker ambiguity cost mechanism.

### Differences from ambiguity_cost_speaker
- Recursion depth: depth 2 (choice_probs calls L2, whereas ambiguity_cost_speaker calls L1).
- Parameters added: none.
- Parameters removed: none (alpha, cost, and lapse are retained with identical priors).
- Other terms: introduced L2[u: UTT, r: OBJ](alpha, costs: ..., lex: ...), where depth-2 speaker choices incorporate communicative informativeness under L1 penalized by utterance ambiguity cost vec(costs, u), and choice_probs evaluates L2 instead of L1 for heard utterances.
