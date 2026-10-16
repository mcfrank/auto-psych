# Live (human) outer-loop — results summary

- Source run root: `/scratch/users/kushinm/auto-psych/outer_loop_live`
- Runs: 3 (run1, run2, run3)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| run1 | experiment1 | individual_streak_aversion_lapse | 1.000 | 40 | 2560 | lapse_individual_streak_aversion | 11.73 | 4.47 |
| run1 | experiment2 | most_lopsided_stretch_aversion | 0.999 | 40 | 5120 | block_rhythm_regularity | 7.28 | 4.62 |
| run1 | experiment3 | personal_pattern_detection_gain | 0.329 | 40 | 7680 | capacity_blur_pattern_gain | 0.16 | 0.96 |
| run2 | experiment1 | graded_periodicity_personal_ideal | 0.642 | 40 | 2560 | person_sensitivity_length_scaled_ideal | 0.91 | 1.49 |
| run2 | experiment2 | tally_span_switch_ideal_periodic_unit | 1.000 | 40 | 5120 | tally_span_switch_ideal_triplet_variety | 13.59 | 5.76 |
| run2 | experiment3 | edge_streak_primacy_recency_pattern_rule | 0.908 | 40 | 7680 | terminal_streak_recency_pattern_rule | 2.33 | 3.41 |
| run3 | experiment1 | length_normalised_chance_vs_motif | 0.999 | 40 | 2560 | bayesian_chance_vs_repeating_motif_2 | 7.86 | 3.00 |
| run3 | experiment2 | face_specific_streak_chance_coin | 0.274 | 40 | 5120 | second_order_motif_side_habit | 0.45 | 3.60 |
| run3 | experiment3 | mild_symmetric_trick_coin_streak_weary | 0.796 | 41 | 7744 | accelerating_streak_weary_chance | 1.38 | 4.11 |

## Winning-model agreement across runs

- experiment1: agreement=no → graded_periodicity_personal_ideal, individual_streak_aversion_lapse, length_normalised_chance_vs_motif
- experiment2: agreement=no → face_specific_streak_chance_coin, most_lopsided_stretch_aversion, tally_span_switch_ideal_periodic_unit
- experiment3: agreement=no → edge_streak_primacy_recency_pattern_rule, mild_symmetric_trick_coin_streak_weary, personal_pattern_detection_gain
