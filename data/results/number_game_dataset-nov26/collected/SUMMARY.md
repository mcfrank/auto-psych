# Live (human) outer-loop — results summary

- Source run root: `/sailhome/wurgaft/auto-psych-runs/ngds-nov26-collect/assembled/collect_src`
- Runs: 3 (run1, run2, run3)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| run1 | experiment1 | coincidence_sharpened_exemplar | 0.551 | None | 2520 | digit_run_rule_attention_exemplar | 1.63 | 5.06 |
| run1 | experiment2 | round_number_averaging_exemplar | 0.947 | None | 5040 | times_table_round_averaging | 3.28 | 4.76 |
| run1 | experiment3 | span_fallback_rule_violation | 1.000 | None | 7560 | set_kin_consensus_violation_route | 16.33 | 9.44 |
| run2 | experiment1 | seed_multiple_salience_overlap | 1.000 | None | 2520 | final_digit_overshadowing_overlap | 14.73 | 6.11 |
| run2 | experiment2 | incremental_band_narrowing | 0.310 | None | 5040 | first_impression_span_interior | 0.00 | 6.72 |
| run2 | experiment3 | neighbourhood_doubt_three_example_memory | 0.463 | None | 7560 | parity_doubt_three_example_memory | 0.63 | 1.44 |
| run3 | experiment1 | extrapolation_verification_failure | 1.000 | None | 2520 | range_bounded_parity_concepts | 21.39 | 8.25 |
| run3 | experiment2 | consistent_examples_bounded_rules | 1.000 | None | 5040 | None | n/a | n/a |
| run3 | experiment3 | exception_weighted_rule_rejection | 1.000 | None | 7560 | None | n/a | n/a |

## Winning-model agreement across runs

- experiment1: agreement=no → coincidence_sharpened_exemplar, extrapolation_verification_failure, seed_multiple_salience_overlap
- experiment2: agreement=no → consistent_examples_bounded_rules, incremental_band_narrowing, round_number_averaging_exemplar
- experiment3: agreement=no → exception_weighted_rule_rejection, neighbourhood_doubt_three_example_memory, span_fallback_rule_violation
