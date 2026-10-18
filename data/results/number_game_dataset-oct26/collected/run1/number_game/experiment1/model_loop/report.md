# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **digit_similarity_with_common_factor** (posterior=0.943, elpd_loo=-1660.28)
- Trials: 2520
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| digit_similarity_with_common_factor | 0.9428 | -1660.28 |
| common_factor_digit_range_similarity | 0.0227 | -1664.01 |
| final_digit_similarity_gradient | 0.0214 | -1664.06 |
| digit_similarity_with_range_extension | 0.0061 | -1665.32 |
| digit_similarity_with_span_falloff | 0.0035 | -1665.87 |
| serial_position_digit_similarity | 0.0033 | -1665.93 |
| proximity_cued_digit_retrieval | 0.0002 | -1668.87 |
| place_value_grid_similarity | 0.0000 | -1671.55 |

## Hypotheses

- **digit_similarity_with_common_factor**: A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), but the one change is that they also notice the largest whole number (3 or more) that divides every example and treat "multiples of that number" as a candidate rule, so a target divisible by the examples' common factor gets an extra, inferred boost toward "yes" whatever its final digit. This addresses the critique that, among parity-matched targets, people accept those divisible by the examples' common divisor (e.g. multiples of 8 after 16, 8, 64) more than the final-digit model predicts.
- **common_factor_digit_range_similarity**: A refinement of `digit_similarity_with_range_extension`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), with an extra boost for targets inside the examples' span that fades beyond its edges, but the one change is that they also notice when several examples are all multiples of a common factor of 3 or more (their greatest common divisor, e.g. 8 for 16, 8, 64) and infer a "multiples of that factor" rule, giving an inferred extra push toward "yes" to targets that are multiples of it. This addresses the critique that, among parity-matched targets of sets sharing a common divisor, people accept divisible targets more than the digit model predicts.
- **final_digit_similarity_gradient**: People ignore magnitude and judge the target by how much its final digit resembles the final digits of the examples: a target ending in the same digit as an example is maximally similar to it, one ending in a different digit of the same parity (odd/even) is partly similar, and one of the opposite parity is least similar. They say "yes" in proportion to the target's average final-digit similarity to all the examples shown, so the answer is graded by the share of examples it resembles rather than by any rule that every example must obey, and numerically close targets get no credit for closeness.
- **digit_similarity_with_range_extension**: A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but they also treat the span from the smallest to the largest example as territory the program covers, so a target inside that span gets an extra, inferred boost toward "yes" that fades smoothly with how far it lies beyond the span's edges (for a single example, with its distance from that example). The one change is this range-membership component grafted from `range_extension_by_spacing`, addressing the critique that the magnitude-blind digit model under-predicts yes for in-range versus out-of-range targets.
- **digit_similarity_with_span_falloff**: A refinement of `final_digit_similarity_gradient`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same digit most similar, same parity partly, opposite parity least), but the one change is that they also treat the span from the smallest to the largest example as covered by the program, so their willingness to say "yes" falls off steadily with how far the target lies beyond that span (with no penalty anywhere inside it). This addresses the critique that people accept targets inside the examples' range more than the magnitude-blind digit model predicts.
- **serial_position_digit_similarity**: People compare the target's final digit with the final digits of the examples (same digit most similar, same parity partly, opposite parity least), but they do not hold every example equally strongly in memory: how much each example counts in the comparison depends on its serial position in the order shown, changing steadily from first to last (recency if later examples are held more strongly, primacy if earlier ones are). So "yes" grows with a position-weighted average of final-digit similarity, and the same numbers shown in a different order change which digit matches drive acceptance.
- **proximity_cued_digit_retrieval**: People compare the target's final digit with the final digits of the examples (same digit most similar, same parity partly, opposite parity least), but the target acts as a retrieval cue: examples close to it in magnitude come to mind readily, while numerically distant examples are only sometimes recalled for the comparison. So "yes" grows with the final-digit similarity to the examples that are actually retrieved, which makes a digit match with a nearby example count for more than the same match with a far-away one.
- **place_value_grid_similarity**: People represent each two-digit number by its place values, as a position on a tens-digit by units-digit grid, and judge the target by its similarity to each example on that grid: similarity is highest for the same final digit, partial for a different final digit of the same parity, and lowest for opposite parity, and it is multiplied down by how many decades apart the target and that example lie. They say "yes" in proportion to the target's average place-value similarity to the examples, so a final-digit match earns credit mainly when it falls in the same or a nearby decade as the matching example. This disagrees most sharply with the magnitude-blind final-digit model on digit-matched targets several decades away from every matching example (e.g. 93 after 13), which it predicts are rejected far more often than nearby digit-matched targets (e.g. 23).

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| digit_similarity_with_common_factor ←selected | 0.00 | 0.00 | — (best) | 0.943 | yes |
| common_factor_digit_range_similarity | 3.73 | 4.57 | no (within ~2·dse) | 0.023 | yes |
| final_digit_similarity_gradient | 3.78 | 5.44 | no (within ~2·dse) | 0.021 | yes |
| digit_similarity_with_range_extension | 5.04 | 7.02 | no (within ~2·dse) | 0.006 | yes |
| digit_similarity_with_span_falloff | 5.59 | 7.00 | no (within ~2·dse) | 0.004 | yes |
| serial_position_digit_similarity | 5.66 | 5.46 | no (within ~2·dse) | 0.003 | yes |
| proximity_cued_digit_retrieval | 8.60 | 6.86 | no (within ~2·dse) | 0.000 | yes |
| place_value_grid_similarity | 11.27 | 7.12 | no (within ~2·dse) | 0.000 | yes |
