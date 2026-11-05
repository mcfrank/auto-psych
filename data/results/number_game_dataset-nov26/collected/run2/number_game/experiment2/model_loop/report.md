# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **incremental_band_narrowing** (posterior=0.310, elpd_loo=-3152.56)
- Trials: 5040
- Models compared: 8

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| incremental_band_narrowing | 0.3095 | -3152.56 |
| first_impression_span_interior | 0.3092 | -3152.57 |
| range_tightened_magnitude_band | 0.1972 | -3153.02 |
| span_sharpened_magnitude_band | 0.0996 | -3153.70 |
| span_interval_similarity | 0.0841 | -3153.87 |
| progression_band_first_digit_primacy | 0.0002 | -3160.03 |
| first_impression_anchoring | 0.0002 | -3160.07 |
| coincidence_gated_first_digit_primacy | 0.0000 | -3164.11 |

## Hypotheses

- **incremental_band_narrowing**: People read the examples one at a time and keep a running guess about how widely the program's numbers spread in magnitude. After a single example the guess is broad, but each additional example revises it and tightens how far beyond the examples' range they are willing to generalize. So targets outside the range lose endorsement faster with distance the more examples have been shown, while targets inside the range keep full magnitude credit, and the rest of the judgment (the salience-weighted share of the examples' shared properties) is unchanged.
- **first_impression_span_interior**: Refinement of `first_impression_anchoring` by one change: when two or more examples are shown, people also take "a number lying between the smallest and the largest example" as a candidate property of the program (the range the examples fill in), with its own inferred salience, overshadowed like the other non-digit properties by a visible shared final digit. Everything else is unchanged (the first impression's contradicted properties keep a fraction of their weight, graded magnitude band, progression concept, seed multiples, the last example's lingering digit), so on multi-example sets targets strictly inside the examples' range are endorsed more than targets just outside it even when they share no digit, parity or seed-multiple with the examples, closing the inrange_nonshared_rate critique.
- **range_tightened_magnitude_band**: Refinement of `progression_band_first_digit_primacy` (the incumbent) by one change to its graded magnitude band: the band's width depends on whether the examples mark out a range. A single example gives no range, so magnitude credit falls off broadly around it, but once two or more examples are shown people take the span they cover as the program's range, and magnitude credit falls off more steeply (by an inferred fraction of the single-example width) below the smallest and above the largest example. Everything else is the incumbent unchanged, so in multi-example sets targets inside the examples' range that share no other property are endorsed more than targets outside it (the inrange_nonshared_rate critique), while single-example sets still generalize broadly.
- **span_sharpened_magnitude_band**: Refinement of the incumbent `progression_band_first_digit_primacy` by one change to its graded magnitude band: how sharply magnitude credit falls off outside the examples depends on whether the examples mark out a span. A lone example gives no edges, so people generalize by magnitude broadly; once two or more examples are shown, their smallest and largest values become real boundaries, and credit falls off much more steeply (its own inferred width) beyond them. Everything else is the incumbent unchanged, so in multi-example sets, targets inside the examples' range that share nothing else with them are endorsed more than equally nondescript targets outside it, closing the inrange_nonshared_rate critique, while single-example generalization stays broad.
- **span_interval_similarity**: When two or more examples are shown, people also treat the stretch of the number line between the smallest and largest example as a candidate concept in its own right ("numbers in this range"), and a target earns credit for it by its similarity to that interval: full credit anywhere inside it, decaying exponentially (Shepard's generalization gradient, at an inferred and fairly steep rate) with the target's distance outside it. This interval similarity enters, with its own salience, the salience-weighted share of the examples' shared profile that drives "yes" in the current best account, so targets lying between the examples are endorsed more than equally nondescript targets outside them, even when they share no digit, parity or seed-multiple with the examples.
- **progression_band_first_digit_primacy**: Refinement of `progression_graded_magnitude_band` (the incumbent) by one grafted component, taken from `graded_band_first_digit_primacy`: when several examples are shown without a shared final digit, the final digit of the FIRST example also lingers as a candidate pattern ("ends in 3"), with its own inferred salience, because the first example anchors people's initial guess before the later examples arrive. Everything else is the incumbent's judgment unchanged (salience-weighted share of the examples' shared profile, with the graded magnitude band, the "start at the smallest and add the gap" progression concept, multiples of the seed, final-digit overshadowing and the last example's lingering digit), so targets sharing the first-shown example's final digit are endorsed more than targets sharing no example's final digit, closing the first_example_digit_lift critique the incumbent fails.
- **first_impression_anchoring**: People read the examples in order and form their first guess about the program from the first example alone, entertaining every salient property it has (its parity, the numbers it is a multiple of, its digits, its final digit). Later examples should rule out the first example's properties they lack, but people adjust insufficiently from this first impression: contradicted first-example properties keep a fraction of their weight, so targets that share properties with the first-shown example — not only its final digit — are endorsed more than the examples' common profile alone implies, and the same examples in a different order are generalized differently.
- **coincidence_gated_first_digit_primacy**: Refinement of `coincidence_gated_pattern_belief` by one grafted component, taken from `graded_band_first_digit_primacy`: when people believe the program follows a rule, the final digit of the FIRST example shown also lingers as a candidate pattern ("ends in 3"), with its own inferred salience, because the first example anchors their initial guess before the later examples arrive. Everything else is unchanged: people first judge from how striking the examples' agreement is whether the program is patterned at all (a single example or broadly shared properties leave them suspecting it is random, so they lean toward "yes"), and otherwise say "yes" by the salience-weighted share of the examples' shared profile the target has, the last example's lingering digit included. So on multi-example sets without a shared final digit, targets sharing the first-shown example's final digit are endorsed more than targets sharing no example's final digit, which is the first_example_digit_lift discrepancy that the best model misses.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| incremental_band_narrowing ←selected | 0.00 | 0.00 | — (best) | 0.310 | yes |
| first_impression_span_interior | 0.00 | 6.72 | no (within ~2·dse) | 0.309 | yes |
| range_tightened_magnitude_band | 0.45 | 3.73 | no (within ~2·dse) | 0.197 | yes |
| span_sharpened_magnitude_band | 1.13 | 3.67 | no (within ~2·dse) | 0.100 | yes |
| span_interval_similarity | 1.30 | 4.59 | no (within ~2·dse) | 0.084 | yes |
| progression_band_first_digit_primacy | 7.47 | 5.95 | no (within ~2·dse) | 0.000 | yes |
| first_impression_anchoring | 7.50 | 7.68 | no (within ~2·dse) | 0.000 | yes |
| coincidence_gated_first_digit_primacy | 11.55 | 7.70 | no (within ~2·dse) | 0.000 | yes |
