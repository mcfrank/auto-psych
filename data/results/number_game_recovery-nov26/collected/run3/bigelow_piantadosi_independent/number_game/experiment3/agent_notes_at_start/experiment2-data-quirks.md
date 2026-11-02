---
name: experiment2-data-quirks
description: cell_1/experiment2 responses.csv facts (4500 rows, 180 sets) useful for designing critique statistics
metadata:
  type: project
---
Found 2026-10-30 (experiment2 iter_0 critique):
- When the target is one of the examples (n=95), people say yes only 0.83 of the time.
- The yes rate falls across the session: about 0.52 for trials 0-89 and 0.25 for trials 360-449, with corr(trial_index, rating) = -0.14. There is no participant_id.
- 62% of (set, target) cells have mixed answers. Sets with one example: 265 rows.
- All-squares sets with 2+ examples (100,1,16; 81,9; 4,64,25,81; 64,4; 81,25) have only 25 square-target rows.
- After all-same-last-digit sets, matching targets get 0.96 yes and others 0.09.

**Why:** a different dataset from experiment1 ([[data-coverage-quirks]]), so don't reuse those counts.
**How to apply:** check slice sizes before building a statistic on experiment2.
