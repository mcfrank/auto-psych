---
name: exp3-iter3-critique-participant-structure
description: Exp3 responses.csv rows are 15 participants x 450 trials in order (trial_index restarts), 3 stimulus lists x 5 participants; harness keeps row order so participant/sequential stats work; iter_3 critique stats and MAP-null z values
metadata:
  type: project
---
Found 2026-11-01 while writing the iter_3 critique (incumbent equal_prior_familiar_rules_digit_sums):
- Participant id = cumsum(diff(trial_index) < 0). There are 15 blocks of 450 trials, and each list of 5 participants shares a stimulus list. Sets are interleaved, not blocked; 80% of same-set runs have length 1.
- The ppc harness keeps the human row order in replicates (rep = human_df.copy() with rating replaced), so participant and lag statistics are valid.
- Model is very well calibrated: the MAP p in iter_2/candidate_3/candidate.py.p.npy matches binned yes rates and per-participant rates. Participant residual SD 0.023 vs null 0.021.
- iter_3 stats, with z values against a MAP-only null (approximate): repeated-set exposure residual slope z=1.7, unanimous-cell fraction z=1.2, everything else |z|<0.7. iter_0 to iter_2 had 0 of 24 significant.
**How to apply:** read iter_3/critique/ppc_results.json. If participant and sequential stats are also null, the model is likely at the data's noise ceiling. See [[exp3-iter2-critique-stats]] and [[exp3-raw-response-patterns]].
