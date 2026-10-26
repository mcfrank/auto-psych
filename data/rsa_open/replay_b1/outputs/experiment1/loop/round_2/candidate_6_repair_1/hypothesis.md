Speakers choose referring expressions to maximize contrast against contextually similar competitors that lack the target feature, while listeners invert this speaker using an object prior combining discrete singleton salience and continuous contextual distinctiveness. On displays without an informative referring expression, listeners expect the speaker to intend the referent that maximizes overall feature distance from the rest of the display.

Refining `contrastive_speaker` by incorporating continuous contextual distinctiveness into the object prior addresses `prior_empty_vs_nonempty_singleton_unit22`, explaining why an empty singleton is preferred over an overpopulated singleton when distractors are feature-dense.

## Differences from contrastive_speaker

1. **Recursion depth**: Unchanged (depth 1, listener calls `L1`).
2. **Parameters added**: `w_distinct` with prior `dist.Normal(0.0, 1.0)`. No parameters removed.
3. **Object prior**: In `listener(params, ctx)`, `prior` now includes continuous contextual distinctiveness (`params["w_distinct"] * distinct`), where `distinct` computes the sum of Hamming distances from each referent to all other context objects. All speaker utility and contrast calculations are unchanged.
