# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **round_number_averaging_exemplar** (posterior=0.947, elpd_loo=-2698.63)
- Trials: 5040
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| round_number_averaging_exemplar | 0.9471 | -2698.63 |
| times_table_round_averaging | 0.0355 | -2701.91 |
| times_table_round_kinship_exemplar | 0.0105 | -2703.13 |
| weber_scaled_magnitude_exemplar | 0.0053 | -2703.81 |
| round_number_kinship_exemplar | 0.0014 | -2705.11 |
| times_table_averaging_exemplar | 0.0001 | -2708.29 |
| recency_gated_arithmetic_route | 0.0000 | -2708.71 |
| round_target_familiarity_bias | 0.0000 | -2710.25 |

## Hypotheses

- **round_number_averaging_exemplar**: Refinement of the incumbent `round_number_kinship_exemplar`, with one component grafted from `similarity_averaging_exemplar` (as used in `times_table_averaging_exemplar`): people pool a target's support across the examples by partly averaging rather than summing it. That support is the target's similarity to each example (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences) plus the distance-free routes for multiples, factor-kin and round-number kinship. So each further example dilutes the support instead of adding to it. A far target in a multi-example set (e.g. "25, 82", "50, 2") is not carried by several weak resemblances added together, while lone examples ("7", "5") keep generalizing broadly to their multiples and round kin (14, 49, 98; 30, 100).
- **times_table_round_averaging**: Refinement of `times_table_averaging_exemplar` (similarity to the examples on magnitude, final digit, parity and the set's common divisor, sharpened by suspicious coincidences, plus a distance-free multiples route only for times-table examples 2-12, with support partly averaged rather than summed over examples). The one change, grafted from `round_number_kinship_exemplar`: round numbers are a familiar class of their own, so a target that is a multiple of ten gains a fixed, distance-free amount of similarity to every example ending in 0 or 5. This predicts that round targets are endorsed well above magnitude and digit similarity after sets containing 0/5-ending examples (100 after "5", 50 after "45, 90"), while far multiples of large two-digit examples (94 after "47", 62 after "31") still get no multiples boost, unlike in the current best model.
- **times_table_round_kinship_exemplar**: Refinement of the incumbent `round_number_kinship_exemplar`: people still judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences), with distance-free routes for factor-kin of an example and for round targets after examples ending in 0 or 5. The one change is that the multiples route is retrieved times-table knowledge, so it exists only for examples that are themselves times-table numbers (2 to 12): after "7" people endorse 14, 49 and 98, but far multiples of two-digit examples (94 after "47", 62 after "31", 75 after "25") get no multiples boost and are judged by ordinary similarity, factor-kinship and roundness alone.
- **weber_scaled_magnitude_exemplar**: People judge a target by its summed similarity to the examples (final digit, parity, the set's common divisor, multiples, factor-kin and round-number kinship, all sharpened when the examples share a suspicious coincidence), but the magnitude part of that similarity is read off a compressed mental number line that obeys Weber's law: how different two numbers feel depends on their gap relative to their size, not on the raw gap. So a gap of 10 makes a target feel much less like a small example (5 against 15) than like a large one (85 against 95), and generalization by magnitude is narrow around small examples and broad around large ones, with the degree of compression inferred from the data.
- **round_number_kinship_exemplar**: People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, multiples and factor-kin of an example, all sharpened when the examples share a suspicious coincidence), but they also treat round numbers as a familiar class of their own: a target that is a multiple of ten resembles every example ending in 0 or 5 by a fixed amount, at any distance and whatever its final digit. So round targets are endorsed well above what magnitude and digit similarity allow when the examples include numbers ending in 0 or 5 (100 after "5", 50 after "45, 90"), while targets ending in 5, and round targets after examples with no 0 or 5 ending, get no such boost (30 after "63, 27, 99, 33").
- **times_table_averaging_exemplar**: Refinement of `times_table_multiple_route` (multiples route only for times-table examples 2-12), with one component grafted from `similarity_averaging_exemplar`: people pool their evidence across examples by (partly) averaging rather than summing it. A target's support — its similarity to each example (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences) plus the times-table multiples route — is diluted as more examples are shown, not accumulated. So a far target is not carried by adding up several weak resemblances in multi-example sets like "25, 82" or "50, 2", while lone examples like "7" or "5" keep generalizing broadly to their times-table multiples (14, 49, 98; 30, 100) and multiples of large examples (94 after "47") get no boost.
- **recency_gated_arithmetic_route**: People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, sharpened when the examples share a suspicious coincidence), but the distance-free arithmetic routes (being a multiple or a factor-kin of an example, and round-number kinship) need the example to be actively held in mind while the target is checked, and examples fade from working memory as later ones arrive. So an example's arithmetic support falls off steadily with how many examples were shown after it: the multiples of the last-shown example (21 after "3, 7", 36 after "5, 9") are endorsed much more than equally far multiples of an earlier example (99 after "3, 33" is not carried by the 3), while ordinary magnitude and digit similarity is unaffected by order.
- **round_target_familiarity_bias**: People judge a target by its summed similarity to the examples (magnitude, final digit, parity, the set's common divisor, multiples and factor-kin of an example, all sharpened when the examples share a suspicious coincidence), but the extra appeal of round numbers is a property of the target itself, not a resemblance to round-ish examples: multiples of ten feel like natural outputs of any number-producing program, so people are more willing to say yes to them by a fixed amount whatever the examples are. This disagrees with `round_number_kinship_exemplar` on round targets after sets with no example ending in 0 or 5 (e.g. 30 or 60 after "63, 27, 99, 33" are boosted here, not there), and on round targets after 0/5 sets, which here get no extra boost from having more round-ish examples.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| round_number_averaging_exemplar ←selected | 0.00 | 0.00 | — (best) | 0.947 | yes |
| times_table_round_averaging | 3.28 | 4.76 | no (within ~2·dse) | 0.036 | yes |
| times_table_round_kinship_exemplar | 4.50 | 5.95 | no (within ~2·dse) | 0.010 | yes |
| weber_scaled_magnitude_exemplar | 5.18 | 5.17 | no (within ~2·dse) | 0.005 | yes |
| round_number_kinship_exemplar | 6.48 | 5.00 | no (within ~2·dse) | 0.001 | yes |
| times_table_averaging_exemplar | 9.66 | 6.63 | no (within ~2·dse) | 0.000 | yes |
| recency_gated_arithmetic_route | 10.08 | 4.99 | yes | 0.000 | yes |
| round_target_familiarity_bias | 11.62 | 6.38 | no (within ~2·dse) | 0.000 | yes |
