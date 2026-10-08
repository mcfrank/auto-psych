This model refines bounded_capacity_listener by replacing the empirical visual salience prior with an uninformative uniform prior over candidate referents. Listeners possess bounded cognitive resources for pragmatic reasoning, anchoring on the literal interpretation of referring expressions and incorporating the speaker's communicative likelihood only to the extent permitted by their processing capacity. In the absence of an informative utterance, choices are uniform across available objects rather than biased by extraneous feature counts or familiarization base rates.

Differences from bounded_capacity_listener:
- Refined model: bounded_capacity_listener
- Recursion depth: depth 1 (choice_probs calls L_bounded, unchanged).
- Parameters added: none.
- Parameters removed: w_features, w_familiar.
- Literal listener L0: listener chooses r in OBJ with probability proportional to at(lex, u, r), replacing the empirical visual salience prior with an uninformative uniform prior over candidate referents.
- Speaker choice in S1: speaker simulates this uniform-prior literal listener L0, removing the prior array parameter.
- Pragmatic listener L_bounded: listener anchors on the uniform-prior literal baseline L0 and incorporates speaker likelihood S1 to the extent permitted by processing capacity, removing the prior array parameter.
- Prior trials in choice_probs: on prior trials (ctx.is_prior > 0), listener chooses uniformly across objects rather than according to feature-count and familiarization weights.
- All other memo agents, distributions, and prior terms are unchanged.
