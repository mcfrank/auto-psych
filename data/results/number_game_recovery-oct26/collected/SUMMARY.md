# Live (human) outer-loop — results summary

- Source run root: `/scr/wurgaft/auto-psych/ng_recovery_stage/trees`
- Runs: 2 (bigelow_piantadosi_interval, tenenbaum_rules_similarity)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| bigelow_piantadosi_interval | experiment1 | criterion_shifted_redundant_examples | 0.426 | None | 2250 | range_bounded_rule_concepts | 0.76 | 1.21 |
| bigelow_piantadosi_interval | experiment2 | volatile_changepoint_inflated_size | 0.872 | None | 4500 | volatile_concept_changepoint | 2.27 | 2.52 |
| bigelow_piantadosi_interval | experiment3 | uneven_familiarity_conservative_updating | 1.000 | None | 6750 | None | n/a | n/a |
| tenenbaum_rules_similarity | experiment1 | scale_end_avoiding_ranges | 0.487 | None | 2250 | weber_blurred_ranges_kind_pooled_rules | 1.06 | 1.82 |
| tenenbaum_rules_similarity | experiment2 | last_example_marks_range_top | 0.386 | None | 4500 | top_marking_uncertain_sampling | 0.69 | 3.16 |
| tenenbaum_rules_similarity | experiment3 | parity_representative_sequence_edges | 0.391 | None | 6750 | parity_representative_stray_example | 0.83 | 1.29 |

## Winning-model agreement across runs

- experiment1: agreement=no → criterion_shifted_redundant_examples, scale_end_avoiding_ranges
- experiment2: agreement=no → last_example_marks_range_top, volatile_changepoint_inflated_size
- experiment3: agreement=no → parity_representative_sequence_edges, uneven_familiarity_conservative_updating
