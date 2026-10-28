# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **relational_analogy_transfer** (posterior=0.462, elpd_loo=-4665.08)
- Trials: 7560
- Models compared: 7

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| relational_analogy_transfer | 0.4624 | -4665.08 |
| leading_digit_even_rule_similarity | 0.4360 | -4665.14 |
| round_base_multiples_prior | 0.0905 | -4666.71 |
| range_inference_leading_digit | 0.0065 | -4669.35 |
| trivial_small_number_avoidance | 0.0045 | -4669.72 |
| leading_digit_string_similarity | 0.0001 | -4673.11 |
| even_number_rule_abstraction | 0.0000 | -4679.60 |

## Hypotheses

- **relational_analogy_transfer**: People reason about the examples relationally, by analogy, as well as by surface similarity. They notice how the examples relate to each other (the difference between two examples, or the ratio between them: 20 then 30 is "add 10", 8 then 16 is "double"). They then ask whether the target relates to some example in the same way, so a target reached by applying an observed relation to another example (40 after 20, 30; 32 after 8, 16) gets extra support for "yes" beyond its feature similarity. Everything else is as in leading_digit_string_similarity.
- **leading_digit_even_rule_similarity**: Refinement of leading_digit_string_similarity (all else unchanged: recency-weighted similarity on size, final digit, parity, graded shared factor and leading written digit with coherence-stretched attention, the "multiples of g", "ends in d" and fixed-list concepts, the filled-in span, sequence momentum, zigzag breadth, the saturating criterion relaxation and the upward-count start boundary). The single change, taken from even_number_rule_abstraction, is that "even numbers" is an explicit rule people can abstract: when every example is even they also consult the rule "the program makes even numbers" as separate evidence, weighted by how unlikely it is that that many examples would all be even by coincidence. "Odd" is the marked leftover category and evokes no such rule, so after all-even sets distant even targets are endorsed and nearby odd targets rejected more than parity similarity predicts, with no matching boost after all-odd sets (the critique's even-versus-odd set asymmetry).
- **round_base_multiples_prior**: Refinement of leading_digit_string_similarity (all else unchanged). The single change is to the "multiples of g" concepts people consider: the family now runs from g = 2 ("even numbers") upward, and the concepts are not equally plausible in advance. Skip-counting bases that are round in base ten or reached by doubling (2, 4, 5, 8, 10, 20, 25, 50 and the powers of two) have a higher prior plausibility than other bases (3, 6, 7, 9, 14 ...). So after examples that share a round base (e.g. 15, 85, 45 or 16, 8), multiples of that base are endorsed and non-multiples rejected more than coincidence evidence alone predicts. After examples that share a less natural base (e.g. 3, 33, 99 or 36, 45, 54, 81), that rule is applied more weakly.
- **range_inference_leading_digit**: Refinement of sample_size_neglect_upward_start (all else unchanged: how far a target lies beyond the examples is judged relative to the range they cover plus a prior expected range, ignoring how many examples were shown; the upward-count start boundary for ascending sets; recency-weighted similarity on final digit, parity and graded shared factor with coherence-stretched attention; the "multiples of g", "ends in d" and fixed-list concepts; sequence momentum, zigzag breadth and the saturating criterion relaxation). The single change, taken from leading_digit_string_similarity, is that people also compare numbers as written digit strings by their leading digit: a target whose first written digit matches an example's (34 after 31 or 3) feels more similar to that example, all-or-none, with attention to this match stretched when the examples share a leading digit. So inside the range-inference account, targets sharing the examples' leading digit are endorsed more, and those that do not are endorsed less, than range-relative distance alone predicts — e.g. after 3, 8 a two-digit 35 can feel close while 20 does not.
- **trivial_small_number_avoidance**: Refinement of leading_digit_string_similarity (all else unchanged: recency-weighted similarity on size, final digit, parity, graded shared factor and leading written digit with coherence-stretched attention, the "multiples of g", "ends in d" and fixed-list concepts, the filled-in span, sequence momentum, zigzag breadth, the saturating criterion relaxation and the upward-count start boundary). The single change is that people hold a prior belief that a number-generating program does not produce the trivial bottom of the counting sequence: the smallest numbers (1 most of all, then 2 and 3, fading quickly above that) feel like degenerate non-outputs, so they are rejected more than their similarity to the examples predicts, whatever the set. So 1 and 2 get fewer "yes" answers than equally similar targets elsewhere on the number line, after sets of large numbers and of small numbers alike.
- **leading_digit_string_similarity**: Refinement of upward_count_start_boundary (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g", "ends in d" and fixed-list concepts, the filled-in span, sequence momentum, zigzag breadth, the saturating criterion relaxation and the upward-count start boundary). The single change is that people also compare numbers as written digit strings by their leading digit: a target whose first written digit matches an example's (34 after 31 or 3, 72 after 79) feels more similar to that example, all-or-none, with attention to this match stretched when the examples themselves share a leading digit. So targets that share the examples' leading digit are endorsed more, and those that do not are endorsed less, than numerical distance alone predicts — even across decade-sized gaps between one- and two-digit numbers (3 and 35).
- **even_number_rule_abstraction**: Refinement of upward_count_start_boundary (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" (g of 3 or more), "ends in d" and fixed-list concepts, the filled-in span, sequence momentum, zigzag breadth, the saturating criterion relaxation and the upward-count start boundary). The single change is that "even numbers" is itself an explicit rule people can abstract: when every example is even they consult the rule "the program makes even numbers" as separate evidence at the decision, in proportion to how unlikely it is that that many examples would all be even by coincidence, so after all-even sets far-away even targets are endorsed and nearby odd targets rejected more than parity similarity predicts. "Odd" is not a rule in the same sense — it is the marked, leftover category of "not even" — so all-odd sets evoke no such rule and get no extra boost for odd targets.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| relational_analogy_transfer ←selected | 0.00 | 0.00 | — (best) | 0.462 | yes |
| leading_digit_even_rule_similarity | 0.06 | 6.42 | no (within ~2·dse) | 0.436 | yes |
| round_base_multiples_prior | 1.63 | 6.90 | no (within ~2·dse) | 0.091 | yes |
| range_inference_leading_digit | 4.27 | 4.83 | no (within ~2·dse) | 0.006 | yes |
| trivial_small_number_avoidance | 4.64 | 7.28 | no (within ~2·dse) | 0.004 | yes |
| leading_digit_string_similarity | 8.03 | 4.53 | no (within ~2·dse) | 0.000 | yes |
| even_number_rule_abstraction | 14.52 | 8.93 | no (within ~2·dse) | 0.000 | yes |
