Listeners interpret utterances literally according to contextual salience, but also apply an exhaustivity preference that penalizes referents possessing unmentioned excess features beyond the named word. When choosing among objects of which the heard utterance is literally true, listeners discount referents that have additional unmentioned features, resolving the tension between prior visual salience (which favors feature-rich objects when no word is heard) and communicative specificity (which favors objects exhaustively described by the speaker's word).

Differences from source (literal_salience_listener):
- Recursion depth: Unchanged (choice_probs calls L0 at depth 0).
- Parameters added: exhaustivity (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L0, the listener's choice weight incorporates an exhaustivity penalty `exp(-exhaustivity * vec(excess, r))`, where `excess` measures the number of unmentioned features possessed by object `r` beyond the named descriptor (`jnp.maximum(0.0, ctx.feature_count - 1.0)`).
