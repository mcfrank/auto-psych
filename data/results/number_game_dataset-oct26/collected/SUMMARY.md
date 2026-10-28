# Live (human) outer-loop — results summary

- Source run root: `/scr/wurgaft/auto-psych/ng_collect_stage_v2`
- Runs: 3 (run1, run2, run3)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| run1 | experiment1 | digit_similarity_with_common_factor | 0.943 | None | 2520 | common_factor_digit_range_similarity | 3.73 | 4.57 |
| run1 | experiment2 | mutual_similarity_contrast_standard | 0.741 | None | 5040 | coincidence_graded_rule_preemption | 1.18 | 4.00 |
| run1 | experiment3 | even_rule_digit_size_principle_counting | 0.973 | None | 7560 | bayesian_digit_veto_counting_up | 3.59 | 1.78 |
| run2 | experiment1 | coherence_sharpened_decision_exemplar | 0.461 | None | 2520 | shared_factor_rule_sharpened_exemplar | 0.00 | 4.03 |
| run2 | experiment2 | zigzag_breadth_sequence_exemplar | 0.381 | None | 5040 | sequence_momentum_span_exemplar | 0.08 | 0.76 |
| run2 | experiment3 | relational_analogy_transfer | 0.462 | None | 7560 | leading_digit_even_rule_similarity | 0.06 | 6.42 |
| run3 | experiment1 | session_growing_acquiescence_concepts | 0.938 | None | 2520 | log_compressed_sampling_concepts | 3.55 | 3.34 |
| run3 | experiment2 | outlier_tolerant_rule_concepts | 1.000 | None | 5040 | None | n/a | n/a |
| run3 | experiment3 | outlier_tolerant_rule_concepts | 0.818 | None | 7560 | late_session_acquiescence_concepts | 1.66 | 3.84 |

## Winning-model agreement across runs

- experiment1: agreement=no → coherence_sharpened_decision_exemplar, digit_similarity_with_common_factor, session_growing_acquiescence_concepts
- experiment2: agreement=no → mutual_similarity_contrast_standard, outlier_tolerant_rule_concepts, zigzag_breadth_sequence_exemplar
- experiment3: agreement=no → even_rule_digit_size_principle_counting, outlier_tolerant_rule_concepts, relational_analogy_transfer
