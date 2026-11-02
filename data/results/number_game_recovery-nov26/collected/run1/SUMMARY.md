# Live (human) outer-loop — results summary

- Source run root: `/sailhome/wurgaft/auto-psych-runs/nov26b-collect/assembled/collect_src/run1`
- Runs: 3 (bigelow_piantadosi_independent, bigelow_piantadosi_mixture, tenenbaum_rules_similarity)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| bigelow_piantadosi_independent | experiment1 | parity_restricted_range_conjunctions | 0.771 | None | 2250 | parity_restricted_compressed_intervals | 1.30 | 1.43 |
| bigelow_piantadosi_independent | experiment2 | rule_plus_exception_neighbourhood | 0.610 | None | 4500 | rule_plus_decade_exception | 0.80 | 3.98 |
| bigelow_piantadosi_independent | experiment3 | salient_decade_place_value_rules | 1.000 | None | 6750 | individual_rule_vs_range_style | 9.63 | 6.88 |
| bigelow_piantadosi_mixture | experiment1 | sticky_guess_power_law_ranges | 0.444 | None | 2250 | first_guess_win_stay_concepts | 0.62 | 1.85 |
| bigelow_piantadosi_mixture | experiment2 | coincidence_triggered_guess_rethink | 0.293 | None | 4500 | misfit_tolerant_coincidence_rethink | 0.03 | 0.47 |
| bigelow_piantadosi_mixture | experiment3 | range_reestimating_span_hesitant_guess | 0.718 | None | 6750 | span_extrapolation_hesitant_guess_2 | 1.73 | 1.33 |
| tenenbaum_rules_similarity | experiment1 | powers_own_family_erlang_interval | 0.229 | None | 2250 | power_base_family_prior | 0.05 | 1.16 |
| tenenbaum_rules_similarity | experiment2 | first_example_evidence_weight | 0.846 | None | 4500 | round_number_range_boundaries | 2.67 | 1.64 |
| tenenbaum_rules_similarity | experiment3 | digit_reversal_pair_concepts | 0.391 | None | 6750 | evidence_quantity_decisiveness | 0.63 | 2.51 |

## Winning-model agreement across runs

- experiment1: agreement=no → parity_restricted_range_conjunctions, powers_own_family_erlang_interval, sticky_guess_power_law_ranges
- experiment2: agreement=no → coincidence_triggered_guess_rethink, first_example_evidence_weight, rule_plus_exception_neighbourhood
- experiment3: agreement=no → digit_reversal_pair_concepts, range_reestimating_span_hesitant_guess, salient_decade_place_value_rules
