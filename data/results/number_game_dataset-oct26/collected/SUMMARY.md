# Live (human) outer-loop — results summary

- Source run root: `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1` (as run1/number_game)
- Runs: 1 (run1)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| run1 | experiment1 | digit_similarity_with_common_factor | 0.943 | None | 2520 | common_factor_digit_range_similarity | 3.73 | 4.57 |
| run1 | experiment2 | mutual_similarity_contrast_standard | 0.741 | None | 5040 | coincidence_graded_rule_preemption | 1.18 | 4.00 |
| run1 | experiment3 | even_rule_digit_size_principle_counting | 0.973 | None | 7560 | bayesian_digit_veto_counting_up | 3.59 | 1.78 |

## Winning-model agreement across runs

- experiment1: agreement=yes → digit_similarity_with_common_factor
- experiment2: agreement=yes → mutual_similarity_contrast_standard
- experiment3: agreement=yes → even_rule_digit_size_principle_counting
