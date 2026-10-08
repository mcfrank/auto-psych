Listeners interpret utterances literally rather than pragmatically, but apply a descriptive parsimony penalty to candidate referents possessing superfluous unmentioned features. When a speaker names a single feature, listeners evaluate matching objects against contextual salience while discounting referents that have extra unmentioned attributes, favoring objects that are minimally and exhaustively described. On prior trials with no informative word, choices are governed strictly by contextual salience without any descriptive penalty.

Differences from source (literal_salience_listener):
- Recursion depth: Unchanged (choice_probs calls L0 at depth 0).
- Parameters added: w_excess (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs and L0, matching referents incur an excess-feature penalty `- w_excess * vec(excess, r)`, where `excess = jnp.maximum(0.0, ctx.feature_count - 1.0)` measures the number of unmentioned features beyond the uttered word.
