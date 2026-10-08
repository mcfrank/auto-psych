Listeners interpret utterances literally according to contextual salience, but weight candidate referents by their descriptive specificity: the proportion of an object's features captured by the uttered word (the reciprocal of the object's feature count). When a speaker uses a single feature word, referents possessing only that named feature are fully and specifically described, whereas referents possessing additional unmentioned features are only partially described and discounted accordingly. On prior trials with no informative word, choices are governed strictly by contextual salience without any specificity weighting.

Differences from source (literal_salience_listener):
- Recursion depth: Unchanged (choice_probs calls L0 at depth 0).
- Parameters added: w_specificity (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In choice_probs and L0, matching referents receive a descriptive specificity weight `exp(w_specificity * vec(spec, r))`, where `spec = 1.0 / jnp.maximum(1.0, ctx.feature_count)` represents the fraction of object r's features accounted for by a single uttered descriptor.
