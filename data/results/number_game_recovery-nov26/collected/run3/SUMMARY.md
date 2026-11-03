# Live (human) outer-loop — results summary

- Source run root: `/sailhome/wurgaft/auto-psych-runs/nov26b-collect/assembled/collect_src/run3`
- Runs: 3 (bigelow_piantadosi_independent, bigelow_piantadosi_mixture, tenenbaum_rules_similarity)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| bigelow_piantadosi_independent | experiment1 | costly_squares_final_digit_concepts | 1.000 | None | 2250 | place_value_disjunctive_contains | 10.79 | 7.68 |
| bigelow_piantadosi_independent | experiment2 | irregular_threes_times_table | 0.857 | None | 4500 | prime_over_odd_reading | 1.88 | 5.20 |
| bigelow_piantadosi_independent | experiment3 | square_pair_concepts | 1.000 | None | 6750 | forward_continuation_spacing_sparsity | 12.70 | 7.26 |
| bigelow_piantadosi_mixture | experiment1 | iter4_candidate5 | 0.203 | None | 2250 | uncertain_reliability_noise_concepts | 0.02 | 1.35 |
| bigelow_piantadosi_mixture | experiment2 | graded_exemplar_equal_rule_prior | 0.476 | None | 4500 | prior_matched_familiar_strays | 0.31 | 2.71 |
| bigelow_piantadosi_mixture | experiment3 | approximate_near_miss_rules | 0.495 | None | 6750 | example_count_scaled_hull | 1.40 | 3.84 |
| tenenbaum_rules_similarity | experiment1 | gamma_width_parity_range_concepts | 0.384 | None | 2250 | parity_within_range_concepts | 0.35 | 0.58 |
| tenenbaum_rules_similarity | experiment2 | mixed_strong_weak_sampling_concepts | 0.416 | None | 4500 | range_size_misperception_concepts | 0.92 | 1.57 |
| tenenbaum_rules_similarity | experiment3 | representative_coverage_family_doubt | 0.353 | None | 6750 | rule_family_doubt_criterion | 0.83 | 1.30 |

## Winning-model agreement across runs

- experiment1: agreement=no → costly_squares_final_digit_concepts, gamma_width_parity_range_concepts, iter4_candidate5
- experiment2: agreement=no → graded_exemplar_equal_rule_prior, irregular_threes_times_table, mixed_strong_weak_sampling_concepts
- experiment3: agreement=no → approximate_near_miss_rules, representative_coverage_family_doubt, square_pair_concepts
