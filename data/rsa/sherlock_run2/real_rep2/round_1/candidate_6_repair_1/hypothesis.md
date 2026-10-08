Listeners interpret referring expressions through depth-2 pragmatic reasoning biased by contextual distinctiveness: they invert a speaker who anticipates a pragmatic listener's interpretation while expecting speakers to refer preferentially to objects that visually pop out from their context. When hearing an ambiguous word, listeners resolve higher-order scalar implicatures while still favoring referents that exhibit greater perceptual contrast with display distractors.

This model refines `distinctive_object_listener` with a single change: increasing the recursion depth from depth 1 to depth 2.
Differences from `distinctive_object_listener`:
- Recursion depth: `choice_probs` calls `L2` instead of `L1`.
- Parameters added or removed: None (retains `alpha`, `w_distinct`, and `lapse`).
- Other terms: Adds the level-2 pragmatic listener `L2[u, r]` inverting a level-2 speaker `S2` who evaluates communicative utility against `L1[u, r]` with the contextual distinctiveness prior, while `L0`, `L1`, the distinctiveness calculation, and the lapse mixture remain identical.
