# Live (human) outer-loop — results summary

- Source run root: `/scratch/users/benpry/auto-psych/outer_loop_live`
- Runs: 3 (run1, run2, run3)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| run1 | experiment1 | iter1_candidate0 | 1.000 | 40 | 1280 | iter0_candidate0 | 13.39 | 7.50 |
| run1 | experiment2 | inner_loop_model | 0.550 | 40 | 2560 | iter1_candidate1 | 0.46 | 0.10 |
| run1 | experiment3 | iter1_candidate0 | 0.969 | 40 | 3840 | linear_accumulated_typicality | 4.37 | 3.46 |
| run2 | experiment1 | iter1_candidate0 | 0.999 | 40 | 1280 | iter0_candidate0 | 7.49 | 4.01 |
| run2 | experiment2 | iter0_candidate1 | 0.620 | 40 | 2560 | iter1_candidate0 | 0.43 | 2.16 |
| run2 | experiment3 | inner_loop_model | 0.492 | 40 | 3840 | evidence_accumulation_per_run | 0.07 | 2.64 |
| run3 | experiment1 | iter1_candidate2 | 0.932 | 40 | 1280 | iter1_candidate0 | 3.28 | 4.21 |
| run3 | experiment2 | smoothed_prototype_distance | 0.547 | 40 | 2560 | iter1_candidate0 | 0.52 | 3.87 |
| run3 | experiment3 | iter1_candidate0 | 0.994 | 40 | 3840 | bayesian_diagnosticity | 5.24 | 3.57 |

## Winning-model agreement across runs

- experiment1: agreement=no → iter1_candidate0, iter1_candidate2
- experiment2: agreement=no → inner_loop_model, iter0_candidate1, smoothed_prototype_distance
- experiment3: agreement=no → inner_loop_model, iter1_candidate0
