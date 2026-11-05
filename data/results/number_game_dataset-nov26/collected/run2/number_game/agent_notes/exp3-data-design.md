---
name: exp3-data-design
description: Experiment 3 responses differ from exp2 — ~150 mostly random-looking sets, 9 answers per (set,target), no participant_id column
metadata:
  type: project
---
Experiment 3 (cell_1) responses.csv has 7560 rows and columns number_set,target,trial_index,rating (no participant_id). It has ~150 sets, many with no structure (e.g. "92,82,27,42"), and each (set,target) pair is answered exactly 9 times. Pandas row `.apply`/`nunique` over the split set columns takes ~0.4 s, which is over the PPC budget of ~0.3 s per call; work over np.unique(sets) instead.

**Why:** stats copied from exp2 that rely on participant_id or on heavily structured sets won't carry over.
**How to apply:** in exp3 critique rounds, condition on per-unique-set flags mapped back with the inverse index. Related: [[critique-stat-no-imports]]
