Listeners interpret referring expressions through depth-2 pragmatic reasoning biased by both referent distinctiveness and utterance extension costs. When resolving an ambiguous word, listeners invert a speaker who anticipates a pragmatic listener while expecting speakers to preferentially refer to visually distinctive objects that pop out from the display and to penalize features shared broadly across multiple visual objects.

This model refines `distinctive_object_l2` by incorporating contextual feature extension costs from `costly_feature_l2` into the recursive speaker utility functions.
Differences from `distinctive_object_l2`:
- Recursion depth: Remains 2 (`choice_probs` calls `L2`).
- Parameters added: `cost_weight` with prior `dist.Normal(0.0, 1.0)`. Parameters removed: None (retains `alpha`, `w_distinct`, and `lapse`).
- Other terms: Computes contextual extension for each utterance (`ext = jnp.sum(ctx.lex, axis=-1) / ctx.lex.shape[-1]`) and cost vector `cost = params["cost_weight"] * ext`. In `L1` and `L2`, the speaker's utility subtracts utterance cost `vec(cost, u)` from communicative informativeness before scaling by rationality `alpha`: `exp(alpha * (log(L... + {EPS}) - vec(cost, u)))`.
