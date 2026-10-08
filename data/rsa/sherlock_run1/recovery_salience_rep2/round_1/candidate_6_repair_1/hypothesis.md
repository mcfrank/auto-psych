Listeners combine depth-2 pragmatic recursive reasoning with a perceptual prior over referents governed by visual feature complexity. When interpreting referring expressions or guessing on uninformative trials, listeners reason about a speaker who anticipates pragmatic listener inferences, while scaling their baseline prior expectations according to the number of visual features each object possesses.

Refinement of feature_salience_listener:
- Base model: feature_salience_listener
- Recursion depth: depth 2 (calls L2 in choice_probs, whereas feature_salience_listener calls L1 at depth 1).
- Parameters added: none.
- Parameters removed: none (retains alpha, salience_weight, and lapse).
- Mathematical / structural changes: In choice_probs, the listener computes depth-2 pragmatic reference resolution (calling L2) rather than depth-1 (L1). L1 models a pragmatic speaker reasoning over literal listener L0 with uniform referent probabilities, and L2 models a pragmatic listener who inverts speaker S2 under the perceptual feature-salience prior.
