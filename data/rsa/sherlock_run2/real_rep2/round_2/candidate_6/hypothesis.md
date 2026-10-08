Listeners interpret referring expressions through depth-2 pragmatic reasoning about speakers who penalize overextended descriptions while favoring referents that perceptually pop out from their visual context. When resolving an ambiguous word, listeners invert a speaker who simulates a pragmatic listener, discounts words shared across many visual objects, and preferentially refers to objects with higher contextual distinctiveness; on prior trials without an informative word, choices follow perceptual pop-out salience directly.

This model refines `costly_feature_l2` with a single change: incorporating the contextual distinctiveness referent prior from `distinctive_object_listener`.
Differences from `costly_feature_l2`:
- Recursion depth: Remains 2 (`choice_probs` calls `L2`).
- Parameters added: `w_distinct` with prior `dist.Normal(0.0, 1.0)`. Parameters removed: None (retains `alpha`, `cost_weight`, and `lapse`).
- Other terms: Computes contextual distinctiveness for each object (`dist_vec = object_distinctiveness(ctx.lex, ctx.is_sink)`) and distinctiveness prior (`prior = softmax_prior(params["w_distinct"] * dist_vec)`). In `L1` and `L2`, the speaker's prior over referents is weighted by `vec(prior, r)`: `speaker: given(r in OBJ, wpp=vec(prior, r))`. On prior trials (`ctx.is_prior > 0`), `choice_probs` returns `prior` instead of uniform guessing.
