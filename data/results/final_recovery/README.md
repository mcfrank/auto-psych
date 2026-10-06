# Final recovery runs

Collected 2026-10-06 09:32 by scripts/subjective_randomness/collect_final_results.sh.

Claude Opus 5.5 agents (API billing), faithful config. Repeats, by sweep:

- `run1/` <- `$SCRATCH/auto-psych/final_1/run1`
- `run2/` <- `$SCRATCH/auto-psych/final_23/run1`
- `run3/` <- `$SCRATCH/auto-psych/final_23/run2`
- `run4/` <- `$SCRATCH/auto-psych/final_45/run1`
- `run5/` <- `$SCRATCH/auto-psych/final_45/run2`
- `impossible/run1/` <- `$SCRATCH/auto-psych/final_impossible_1/run1`

`sweeps/<sweep>/` holds each sweep's launch record and its own test_retest summary
(over that sweep's repeats only).

## Cells without a result at collection time

- final_1/run1/finite_experience_occurrence -> run1/finite_experience_occurrence
- final_45/run1/falk_konold_dp -> run4/falk_konold_dp
- final_45/run2/local_representativeness -> run5/local_representativeness
- final_impossible_1/run1/fewer_heads_more_random -> impossible/run1/fewer_heads_more_random
- final_impossible_1/run1/longer_runs_more_random -> impossible/run1/longer_runs_more_random
- final_impossible_1/run1/more_heads_more_random -> impossible/run1/more_heads_more_random
- final_impossible_1/run1/more_imbalance_more_random -> impossible/run1/more_imbalance_more_random
